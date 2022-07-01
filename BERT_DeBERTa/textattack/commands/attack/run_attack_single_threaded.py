"""

TextAttack Command Class for Attack Single Threaded
-----------------------------------------------------

A command line parser to run an attack in single thread from user specifications.

"""

from collections import deque
import os
import time

import tqdm
import dill

import textattack

from .attack_args_helpers import (
    parse_attack_from_args,
    parse_dataset_from_args,
    dataset_from_my_file,
    parse_logger_from_args,
)

from textattack.commands.train_model.run_training import load_examples

logger = textattack.shared.logger


def run(args, checkpoint=None):
    # Only use one GPU, if we have one.
    # TODO: Running Universal Sentence Encoder uses multiple GPUs
    # if "CUDA_VISIBLE_DEVICES" not in os.environ:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # Disable tensorflow logs, except in the case of an error.
    # if "TF_CPP_MIN_LOG_LEVEL" not in os.environ:
    #     os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    if args.gpu > -1:
        args.device = "cuda:{}".format(args.gpu_id)
    else:
        args.device = "cpu"


    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            tf.config.experimental.set_visible_devices(gpus[int(args.gpu_id)], 'GPU')
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

    if args.checkpoint_resume:
        num_remaining_attacks = checkpoint.num_remaining_attacks
        worklist = checkpoint.worklist
        worklist_tail = checkpoint.worklist_tail

        logger.info(
            "Recovered from checkpoint previously saved at {}".format(
                checkpoint.datetime
            )
        )
        print(checkpoint, "\n")
    else:
        if not args.interactive:
            current_dir = os.path.dirname(os.path.realpath(__file__))
            data_dir = os.path.join(
                current_dir, os.pardir, os.pardir, "my_datasets", args.dataset, args.dataset
            )
            args.data_dir = os.path.normpath(data_dir)
            test_text, test_labels = load_examples(args, args.dataset, type='test')
        if args.num_examples < len(test_labels):
            num_remaining_attacks = args.num_examples
        else:
            num_remaining_attacks = len(test_labels)
        worklist = deque(range(0, num_remaining_attacks))
        # worklist = deque(range(0, 10))
        worklist_tail = worklist[-1]

    start_time = time.time()

    # Attack
    attack = parse_attack_from_args(args)
    print(attack, "\n")

    # Logger
    if args.checkpoint_resume:
        attack_log_manager = checkpoint.log_manager
    else:
        attack_log_manager = parse_logger_from_args(args)

    load_time = time.time()
    textattack.shared.logger.info(f"Load time: {load_time - start_time}s")

    if args.interactive:
        print("Running in interactive mode")
        print("----------------------------")

        while True:
            print('Enter a sentence to attack or "q" to quit:')
            text = input()

            if text == "q":
                break

            if not text:
                continue

            print("Attacking...")

            attacked_text = textattack.shared.attacked_text.AttackedText(text)
            initial_result = attack.goal_function.get_output(attacked_text)
            result = next(attack.attack_dataset([(text, initial_result)]))
            print(result.__str__(color_method="ansi") + "\n")

    else:
        # Not interactive? Use default dataset.

        file_name = os.path.join(args.model, args.save_file)
        fileobject = open(file_name, 'w')

        pbar = tqdm.tqdm(total=num_remaining_attacks, smoothing=0)
        if args.checkpoint_resume:
            num_results = checkpoint.results_count
            num_failures = checkpoint.num_failed_attacks
            num_successes = checkpoint.num_successful_attacks
        else:
            num_results = 0
            num_failures = 0
            num_successes = 0

        for result in attack.attack_dataset(test_text, test_labels, indices=worklist):
            attack_log_manager.log_result(result)
            fileobject.write(result.original_result.attacked_text.text)
            fileobject.write(' >> ')
            fileobject.write(str(result.original_result.ground_truth_output))
            fileobject.write('\n')
            fileobject.write(result.perturbed_result.attacked_text.text)
            fileobject.write(' >> ')
            fileobject.write(str(result.perturbed_result.ground_truth_output))
            fileobject.write('\n')

            if not args.disable_stdout:
                print("\n")
            if (not args.attack_n) or (
                not isinstance(result, textattack.attack_results.SkippedAttackResult)
            ):
                pbar.update(1)
            else:
                # worklist_tail keeps track of highest idx that has been part of worklist
                # Used to get the next dataset element when attacking with `attack_n` = True.
                worklist_tail += 1
                worklist.append(worklist_tail)

            num_results += 1

            if (
                type(result) == textattack.attack_results.SuccessfulAttackResult
                or type(result) == textattack.attack_results.MaximizedAttackResult
            ):
                num_successes += 1
            if type(result) == textattack.attack_results.FailedAttackResult:
                num_failures += 1
            pbar.set_description(
                "[Succeeded / Failed / Total] {} / {} / {}".format(
                    num_successes, num_failures, num_results
                )
            )

            if (
                args.checkpoint_interval
                and len(attack_log_manager.results) % args.checkpoint_interval == 0
            ):
                new_checkpoint = textattack.shared.Checkpoint(
                    args, attack_log_manager, worklist, worklist_tail
                )
                new_checkpoint.save()
                attack_log_manager.flush()

        pbar.close()
        print()
        # Enable summary stdout
        if args.disable_stdout:
            attack_log_manager.enable_stdout()
        attack_log_manager.log_summary()
        attack_log_manager.flush()
        print()
        # finish_time = time.time()
        textattack.shared.logger.info(f"Attack time: {time.time() - load_time}s")

        return attack_log_manager.results
