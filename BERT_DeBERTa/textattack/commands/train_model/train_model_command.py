from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import datetime
import os

from textattack.commands import TextAttackCommand


class TrainModelCommand(TextAttackCommand):
    """The TextAttack train module:

    A command line parser to train a model from user specifications.
    """

    def run(self, args):
        date_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
        current_dir = os.path.dirname(os.path.realpath(__file__))
        outputs_dir = os.path.join(
            current_dir, os.pardir, os.pardir, os.pardir, "outputs", "training"
        )
        outputs_dir = os.path.normpath(outputs_dir)

        # args.output_dir = os.path.join(
        #     outputs_dir, f"{args.model}-{args.dataset}-{date_now}/"
        # )
        args.output_dir = os.path.join(
            outputs_dir, f"{args.model}-{args.dataset}-{args.task}-{args.run_num}/"
        )

        from .run_training import train_model

        train_model(args)

    @staticmethod
    def register_subcommand(main_parser: ArgumentParser):
        parser = main_parser.add_parser(
            "train",
            help="train a model for sequence classification",
            formatter_class=ArgumentDefaultsHelpFormatter,
        )
        parser.add_argument(
            "--model",
            type=str,
            required=True,
            help="directory of model to train",
        )
        parser.add_argument(
            "--dataset",
            type=str,
            required=True,
            default="sst2",
            help="dataset for training; will be loaded from "
            "`datasets` library. if dataset has a subset, separate with a colon. "
            " ex: `glue^sst2` or `rotten_tomatoes`",
        )
        parser.add_argument(
            "--task",
            type=str,
            required=True,
            default="base",
            help="task: base, adv",
        )
        parser.add_argument(
            "--pct-dataset",
            type=float,
            default=1.0,
            help="Fraction of dataset to use during training ([0., 1.])",
        )
        parser.add_argument(
            "--dataset-train-split",
            "--train-split",
            type=str,
            default="",
            help="train dataset split, if non-standard "
            "(can automatically detect 'train'",
        )
        parser.add_argument(
            "--dataset-dev-split",
            "--dataset-eval-split",
            "--dev-split",
            type=str,
            default="",
            help="val dataset split, if non-standard "
            "(can automatically detect 'dev', 'validation', 'eval')",
        )
        parser.add_argument(
            "--tb-writer-step",
            type=int,
            default=1,
            help="Number of steps before writing to tensorboard",
        )
        parser.add_argument(
            "--checkpoint-steps",
            type=int,
            default=-1,
            help="save model after this many steps (-1 for no checkpointing)",
        )
        parser.add_argument(
            "--checkpoint-every-epoch",
            action="store_true",
            default=False,
            help="save model checkpoint after each epoch",
        )
        parser.add_argument(
            "--save-last",
            action="store_true",
            default=False,
            help="Overwrite the saved model weights after the final epoch.",
        )
        parser.add_argument(
            "--num-train-epochs",
            "--epochs",
            type=int,
            default=100,
            help="Total number of epochs to train for",
        )
        parser.add_argument(
            "--attack",
            type=str,
            default=None,
            help="Attack recipe to use (enables adversarial training)",
        )
        parser.add_argument(
            "--check-robustness",
            default=False,
            action="store_true",
            help="run attack each epoch to measure robustness, but train normally",
        )
        parser.add_argument(
            "--num-clean-epochs",
            type=int,
            default=1,
            help="Number of epochs to train on the clean dataset before adversarial training (N/A if --attack unspecified)",
        )
        parser.add_argument(
            "--attack-period",
            type=int,
            default=10,
            help="How often (in epochs) to generate a new adversarial training set (N/A if --attack unspecified)",
        )
        parser.add_argument(
            "--augment",
            type=str,
            default=None,
            help="Augmentation recipe to use",
        )
        parser.add_argument(
            "--pct-words-to-swap",
            type=float,
            default=0.1,
            help="Percentage of words to modify when using data augmentation (--augment)",
        )
        parser.add_argument(
            "--transformations-per-example",
            type=int,
            default=4,
            help="Number of augmented versions to create from each example when using data augmentation (--augment)",
        )
        parser.add_argument(
            "--allowed-labels",
            type=int,
            nargs="*",
            default=[],
            help="Labels allowed for training (examples with other labels will be discarded)",
        )
        parser.add_argument(
            "--early-stopping-epochs",
            type=int,
            default=-1,
            help="Number of epochs validation must increase"
            " before stopping early (-1 for no early stopping)",
        )
        parser.add_argument(
            "--batch-size", type=int, default=128, help="Batch size for training"
        )
        parser.add_argument(
            "--max-length",
            type=int,
            default=512,
            help="Maximum length of a sequence (anything beyond this will "
            "be truncated)",
        )
        parser.add_argument(
            "--learning-rate",
            "--lr",
            type=float,
            default=2e-5,
            help="Learning rate for Adam Optimization",
        )
        parser.add_argument(
            "--grad-accum-steps",
            type=int,
            default=1,
            help="Number of steps to accumulate gradients before optimizing, "
            "advancing scheduler, etc.",
        )
        parser.add_argument(
            "--warmup-proportion",
            type=float,
            default=0.1,
            help="Warmup proportion for linear scheduling",
        )
        parser.add_argument(
            "--config-name",
            type=str,
            default="config.json",
            help="Filename to save BERT config as",
        )
        parser.add_argument(
            "--weights-name",
            type=str,
            default="pytorch_model.bin",
            help="Filename to save model weights as",
        )
        parser.add_argument(
            "--enable-wandb",
            default=False,
            action="store_true",
            help="log metrics to Weights & Biases",
        )
        parser.add_argument(
            "--low_freq",
            type=int,
            default=0,
            help="filter out low frequency words in vocab",
        )
        parser.add_argument(
            "--hidden_dim",
            type=int,
            default=300,
            help="number of hidden dimension",
        )
        parser.add_argument(
            "--embed_dim",
            type=int,
            default=768,
            help="number of embedding dimension",
        )
        parser.add_argument(
            "--dropout",
            type=float,
            default=0.2,
            help="dropout rate",
        )
        parser.add_argument(
            "--max_seq_length",
            type=int,
            default=50,
            help="maximum sequence length",
        )
        parser.add_argument(
            "--kernel-num",
            type=int,
            default=200,
            help="number of each kind of kernel",
        )
        parser.add_argument(
            "--kernel-sizes",
            type=str,
            default='3, 4, 5',
            help="comma-separated kernel size to use for convolution",
        )
        parser.add_argument('--attack-num', type=int, default=10000, help='number of hidden dimension')
        parser.add_argument('--run-num', type=str, default='', help='running number')
        parser.add_argument('--beta', type=float, default=1, help='beta')
        parser.add_argument('--gamma', type=float, default=1, help='gamma')
        parser.add_argument('--mask-hidden-dim', type=int, default=400, help='number of hidden dimension')
        parser.add_argument("--activation", type=str, dest="activation", default="tanh", help='the choice of \
                non-linearity transfer function')
        parser.add_argument("--random-seed", default=1111, type=int)
        parser.add_argument('--gpu', default=0, type=int, help='0:gpu, -1:cpu')
        parser.add_argument('--gpu_id', default='2', type=str, help='gpu id')

        # Add bert parameters
        parser.add_argument("--model_name_or_path", default='bert-base-uncased', type=str,
                        help="Path to pre-trained model or shortcut name selected in the list: ")
        parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
        parser.add_argument("--tokenizer_name", default="", type=str,
                            help="Pretrained tokenizer name or path if not the same as model_name")
        parser.add_argument("--cache_dir", default="", type=str,
                            help="Where do you want to store the pre-trained models downloaded from s3")
        parser.add_argument("--do_lower_case", default=True,
                            help="Set this flag if you are using an uncased model.")
        parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                            help="Batch size per GPU/CPU for training.")
        parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                            help="Batch size per GPU/CPU for evaluation.")
        parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                            help="Number of updates steps to accumulate before performing a backward/update pass.")
        parser.add_argument("--weight_decay", default=0.0, type=float,
                            help="Weight decay if we apply some.")
        parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                            help="Epsilon for Adam optimizer.")
        parser.add_argument("--max_grad_norm", default=10.0, type=float,
                            help="Max gradient norm.")
        parser.add_argument("--max_steps", default=-1, type=int,
                            help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
        parser.add_argument("--warmup_steps", default=0, type=int,
                            help="Linear warmup over warmup_steps.")
        parser.add_argument('--overwrite_cache', action='store_true',
                            help="Overwrite the cached training and evaluation sets")
        parser.add_argument('--savename', type=str, default='maskbert.pt',
                        help='path to save the final model')

        parser.set_defaults(func=TrainModelCommand())
