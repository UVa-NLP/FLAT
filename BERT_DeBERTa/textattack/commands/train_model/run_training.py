from __future__ import absolute_import, division, print_function

import collections
import json
import logging
import math
import os
import random

import numpy as np
import scipy
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
# import transformers

import textattack
import matplotlib.pyplot as plt
import torch.nn.functional as F

import argparse
import glob
import logging
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from textattack.commands.train_model.pytorch_transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForSequenceClassification, BertTokenizer)
from textattack.commands.train_model.transformers import (DebertaConfig,
                                  DebertaForSequenceClassification, DebertaTokenizer)

from textattack.commands.train_model.pytorch_transformers import AdamW, WarmupLinearSchedule

from textattack.commands.train_model.utils_glue import (convert_examples_to_features, convert_text_label_to_features,
                        output_modes, processors)
from textattack.commands.train_model.bert_mask_model import *


MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'deberta': (DebertaConfig, DebertaForSequenceClassification, DebertaTokenizer)
}


from .train_args_helpers import (
    attack_from_args,
    augmenter_from_args,
    dataset_from_args,
    dataset_from_my_file,
    model_from_args,
    model_from_my_file,
    write_readme,
)

# device = textattack.shared.utils.device
logger = textattack.shared.logger


def _save_args(args, save_path):
    """Dump args dictionary to a json.

    :param: args. Dictionary of arguments to save.
    :save_path: Path to json file to write args to.
    """
    final_args_dict = {k: v for k, v in vars(args).items() if _is_writable_type(v)}
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(final_args_dict, indent=2) + "\n")


def _get_sample_count(*lsts):
    """Get sample count of a dataset.

    :param *lsts: variable number of lists.
    :return: sample count of this dataset, if all lists match, else None.
    """
    if all(len(lst) == len(lsts[0]) for lst in lsts):
        sample_count = len(lsts[0])
    else:
        sample_count = None
    return sample_count


def _random_shuffle(*lsts):
    """Randomly shuffle a dataset. Applies the same permutation to each list
    (to preserve mapping between inputs and targets).

    :param *lsts: variable number of lists to shuffle.
    :return: shuffled lsts.
    """
    permutation = np.random.permutation(len(lsts[0]))
    shuffled = []
    for lst in lsts:
        shuffled.append((np.array(lst)[permutation]).tolist())
    return tuple(shuffled)


def _train_val_split(*lsts, split_val=0.2):
    """Split dataset into training and validation sets.

    :param *lsts: variable number of lists that make up a dataset (e.g. text, labels)
    :param split_val: float [0., 1.). Fraction of the dataset to reserve for evaluation.
    :return: (train split of list for list in lsts), (val split of list for list in lsts)
    """
    sample_count = _get_sample_count(*lsts)
    if not sample_count:
        raise Exception(
            "Batch Axis inconsistent. All input arrays must have first axis of equal length."
        )
    lsts = _random_shuffle(*lsts)
    split_idx = math.floor(sample_count * split_val)
    train_set = [lst[split_idx:] for lst in lsts]
    val_set = [lst[:split_idx] for lst in lsts]
    if len(train_set) == 1 and len(val_set) == 1:
        train_set = train_set[0]
        val_set = val_set[0]
    return train_set, val_set


def _filter_labels(text, labels, allowed_labels):
    """Keep examples with approved labels.

    :param text: list of text inputs.
    :param labels: list of corresponding labels.
    :param allowed_labels: list of approved label values.

    :return: (final_text, final_labels). Filtered version of text and labels
    """
    final_text, final_labels = [], []
    for text, label in zip(text, labels):
        if label in allowed_labels:
            final_text.append(text)
            final_labels.append(label)
    return final_text, final_labels


def _save_model_checkpoint(model, output_dir, global_step):
    """Save model checkpoint to disk.

    :param model: Model to save (pytorch)
    :param output_dir: Path to model save dir.
    :param global_step: Current global training step #. Used in ckpt filename.
    """
    # Save model checkpoint
    output_dir = os.path.join(output_dir, "checkpoint-{}".format(global_step))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Take care of distributed/parallel training
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(output_dir)


def _save_model(model, output_dir, weights_name, config_name):
    """Save model to disk.

    :param model: Model to save (pytorch)
    :param output_dir: Path to model save dir.
    :param weights_name: filename for model parameters.
    :param config_name: filename for config.
    """
    model_to_save = model.module if hasattr(model, "module") else model

    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(output_dir, weights_name)
    output_config_file = os.path.join(output_dir, config_name)

    torch.save(model_to_save.state_dict(), output_model_file)
    try:
        model_to_save.config.to_json_file(output_config_file)
    except AttributeError:
        # no config
        pass


def _make_directories(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def _is_writable_type(obj):
    for ok_type in [bool, int, str, float]:
        if isinstance(obj, ok_type):
            return True
    return False


def batch_encode(tokenizer, text_list):
    if hasattr(tokenizer, "batch_encode"):
        return tokenizer.batch_encode(text_list)
    else:
        return [tokenizer.encode(text_input) for text_input in text_list]


def _make_dataloader(tokenizer, text, labels, batch_size, shuffle=True):
    """Create torch DataLoader from list of input text and labels.

    :param tokenizer: Tokenizer to use for this text.
    :param text: list of input text.
    :param labels: list of corresponding labels.
    :param batch_size: batch size (int).
    :return: torch DataLoader for this training set.
    """
    text_ids = batch_encode(tokenizer, text)
    input_ids = np.array(text_ids)
    labels = np.array(labels)
    data = list((ids, label) for ids, label in zip(input_ids, labels))
    if shuffle:
        sampler = RandomSampler(data)
    else:
        sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    return dataloader


def _make_reg_dataloader(tokenizer, ori_text, adv_text, labels, batch_size):
    """Create torch DataLoader from list of input text and labels.

    :param tokenizer: Tokenizer to use for this text.
    :param ori_text: list of original input text.
    :param adv_text: list of adversarial input text.
    :param labels: list of corresponding labels.
    :param batch_size: batch size (int).
    :return: torch DataLoader for this training set.
    """
    ori_text_ids = batch_encode(tokenizer, ori_text)
    adv_text_ids = batch_encode(tokenizer, adv_text)
    ori_input_ids = np.array(ori_text_ids)
    adv_input_ids = np.array(adv_text_ids)
    labels = np.array(labels)
    data = list((ori_ids, adv_ids, label) for ori_ids, adv_ids, label in zip(ori_input_ids, adv_input_ids, labels))
    sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    return dataloader


def _data_augmentation(text, labels, augmenter):
    """Use an augmentation method to expand a training set.

    :param text: list of input text.
    :param labels: list of corresponding labels.
    :param augmenter: textattack.augmentation.Augmenter, augmentation scheme.

    :return: augmented_text, augmented_labels. list of (augmented) input text and labels.
    """
    aug_text = augmenter.augment_many(text)
    # flatten augmented examples and duplicate labels
    flat_aug_text = []
    flat_aug_labels = []
    for i, examples in enumerate(aug_text):
        for aug_ver in examples:
            flat_aug_text.append(aug_ver)
            flat_aug_labels.append(labels[i])
    return flat_aug_text, flat_aug_labels


def _generate_adversarial_examples(args, model, attack_class, dataset):
    """Create a dataset of adversarial examples based on perturbations of the
    existing dataset.

    :param model: Model to attack.
    :param attack_class: class name of attack recipe to run.
    :param dataset: iterable of (text, label) pairs.

    :return: list(AttackResult) of adversarial examples.
    """
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
    attack = attack_class.build(model)


    adv_attack_results = []
    for adv_ex in tqdm(
        attack.attack_train_dataset(dataset), desc="Attack", total=len(dataset)
    ):
        adv_attack_results.append(adv_ex)
    return adv_attack_results


def _deberta_cal_discrepance(model, tokenizer, ori_input_ids, adv_input_ids):
    bsz, seq = ori_input_ids.shape
    d_loss = 0
    for bs in range(bsz):
        ori_ids = [i for i in ori_input_ids[bs] if i != 1 and i != 2 and i != 0]
        adv_ids = [i for i in adv_input_ids[bs] if i != 1 and i != 2 and i != 0]
        ori_tokens = [tokenizer.convert_ids_to_tokens(int(i)) for i in ori_ids]
        adv_tokens = [tokenizer.convert_ids_to_tokens(int(i)) for i in adv_ids]
        ori_imp = model.get_importance_score(torch.tensor(ori_ids, device=ori_input_ids.device).unsqueeze(0)).squeeze(-1).squeeze(0)
        adv_imp = model.get_importance_score(torch.tensor(adv_ids, device=ori_input_ids.device).unsqueeze(0)).squeeze(-1).squeeze(0)
        ori_imp_clean = []
        adv_imp_clean = []
        for i, tok in enumerate(ori_tokens):
            if ori_imp_clean == []:
                ori_imp_clean.append(ori_imp[i])
            elif tok[:1] != 'Ġ':
                if ori_imp[i] > ori_imp_clean[-1]:
                    ori_imp_clean[-1] = ori_imp[i]
                else:
                    continue
            else:
                ori_imp_clean.append(ori_imp[i])
        for i, tok in enumerate(adv_tokens):
            if adv_imp_clean == []:
                    adv_imp_clean.append(adv_imp[i])
            elif tok[:1] != 'Ġ':
                if adv_imp[i] > adv_imp_clean[-1]:
                    adv_imp_clean[-1] = adv_imp[i]
                else:
                    continue
            else:
                adv_imp_clean.append(adv_imp[i])
        ori_imp_clean = torch.tensor(ori_imp_clean[:min(len(ori_imp_clean), len(adv_imp_clean))], device=ori_input_ids.device)
        adv_imp_clean = torch.tensor(adv_imp_clean[:min(len(ori_imp_clean), len(adv_imp_clean))], device=ori_input_ids.device)
        d_loss += torch.norm((ori_imp_clean - adv_imp_clean), 1)

    return d_loss/bsz


def _cal_discrepance(model, tokenizer, ori_input_ids, adv_input_ids):
    bsz, seq = ori_input_ids.shape
    d_loss = 0
    for bs in range(bsz):
        ori_ids = [i for i in ori_input_ids[bs] if i != 101 and i != 102 and i != 0]
        adv_ids = [i for i in adv_input_ids[bs] if i != 101 and i != 102 and i != 0]
        ori_tokens = [tokenizer.ids_to_tokens[int(i)] for i in ori_ids]
        adv_tokens = [tokenizer.ids_to_tokens[int(i)] for i in adv_ids]
        ori_imp = model.get_importance_score(torch.tensor(ori_ids, device=ori_input_ids.device).unsqueeze(0)).squeeze(-1).squeeze(0)
        adv_imp = model.get_importance_score(torch.tensor(adv_ids, device=ori_input_ids.device).unsqueeze(0)).squeeze(-1).squeeze(0)
        ori_imp_clean = []
        adv_imp_clean = []
        for i, tok in enumerate(ori_tokens):
            if tok[:2] == '##':
                if ori_imp_clean == []:
                    ori_imp_clean.append(ori_imp[i])
                elif ori_imp[i] > ori_imp_clean[-1]:
                    ori_imp_clean[-1] = ori_imp[i]
                else:
                    continue
            else:
                ori_imp_clean.append(ori_imp[i])
        for i, tok in enumerate(adv_tokens):
            if tok[:2] == '##':
                if adv_imp_clean == []:
                    adv_imp_clean.append(adv_imp[i])
                elif adv_imp[i] > adv_imp_clean[-1]:
                    adv_imp_clean[-1] = adv_imp[i]
                else:
                    continue
            else:
                adv_imp_clean.append(adv_imp[i])
        ori_imp_clean = torch.tensor(ori_imp_clean[:min(len(ori_imp_clean), len(adv_imp_clean))], device=ori_input_ids.device)
        adv_imp_clean = torch.tensor(adv_imp_clean[:min(len(ori_imp_clean), len(adv_imp_clean))], device=ori_input_ids.device)
        d_loss += torch.norm((ori_imp_clean - adv_imp_clean), 1)

    return d_loss/bsz


def load_examples(args, task, type):
    processor = processors[task]()
    if type == 'train':
        texts, labels = processor.get_text_label(args.data_dir, type)
    elif type == 'dev':
        texts, labels = processor.get_text_label(args.data_dir, type)
    else:
        texts, labels = processor.get_text_label(args.data_dir, type)
    return texts, labels


def set_dataloader(args, tokenizer, adv_train_text_temp, train_labels_temp):
    processor = processors[args.dataset]()
    output_mode = output_modes[args.dataset]
    label_list = processor.get_labels()
    features = convert_text_label_to_features(adv_train_text_temp, train_labels_temp, label_list, args.max_seq_length, tokenizer, output_mode,
                                                cls_token_at_end=bool(args.model_type in ['xlnet']),
                                                # xlnet has a cls token at the end
                                                cls_token=tokenizer.cls_token,
                                                cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
                                                sep_token=tokenizer.sep_token,
                                                sep_token_extra=bool(args.model_type in ['roberta']),
                                                # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                                                pad_on_left=bool(args.model_type in ['xlnet']),
                                                # pad on the left for xlnet
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
                                                )
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    data_sampler = RandomSampler(dataset)
    daloader = DataLoader(dataset, sampler=data_sampler, batch_size=args.train_batch_size)
    return daloader


def set_dataloader_reg(args, tokenizer, train_text_temp, adv_train_text_temp, train_labels_temp):
    processor = processors[args.dataset]()
    output_mode = output_modes[args.dataset]
    label_list = processor.get_labels()
    features = convert_text_label_to_features(adv_train_text_temp, train_labels_temp, label_list, args.max_seq_length, tokenizer, output_mode,
                                                train_text_temp,
                                                cls_token_at_end=bool(args.model_type in ['xlnet']),
                                                # xlnet has a cls token at the end
                                                cls_token=tokenizer.cls_token,
                                                cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
                                                sep_token=tokenizer.sep_token,
                                                sep_token_extra=bool(args.model_type in ['roberta']),
                                                # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                                                pad_on_left=bool(args.model_type in ['xlnet']),
                                                # pad on the left for xlnet
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
                                                )
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_ids_ori = torch.tensor([f.input_ids_ori for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_input_ids_ori)
    data_sampler = RandomSampler(dataset)
    daloader = DataLoader(dataset, sampler=data_sampler, batch_size=args.train_batch_size)
    return daloader


def load_and_cache_examples(args, task, tokenizer, type):
    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}_{}'.format(
        type,
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task),
        args.task))
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if type == 'train':
            examples = processor.get_train_examples(args.data_dir)
        elif type == 'dev':
            examples = processor.get_dev_examples(args.data_dir)
        else:
            examples = processor.get_test_examples(args.data_dir)
        features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer, output_mode,
                                                cls_token_at_end=bool(args.model_type in ['xlnet']),
                                                # xlnet has a cls token at the end
                                                cls_token=tokenizer.cls_token,
                                                cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
                                                sep_token=tokenizer.sep_token,
                                                sep_token_extra=bool(args.model_type in ['roberta']),
                                                # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                                                pad_on_left=bool(args.model_type in ['xlnet']),
                                                # pad on the left for xlnet
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
                                                )
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset


def evaluate(args, model, eval_dataset):
    eval_output_dir = args.output_dir
    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],  # XLM and RoBERTa don't use segment_ids
                      'labels':         batch[3]}
            try:
                outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'],\
                    token_type_ids=inputs['token_type_ids'], labels=inputs['labels'])
            except:
                outputs = model(inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs['labels'].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=1)
    eval_acc = (preds == out_label_ids).mean()

    return eval_loss, eval_acc


def train_model(args):
    if args.gpu > -1:
        args.device = "cuda:{}".format(args.gpu_id)
    else:
        args.device = "cpu"
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    textattack.shared.utils.set_seed(args.random_seed)

    logger.warn(
        "WARNING: TextAttack's model training feature is in beta. Please report any issues on our Github page, https://github.com/QData/TextAttack/issues."
    )
    _make_directories(args.output_dir)

    # num_gpus = torch.cuda.device_count()
    args.n_gpu = 1

    # Save logger writes to file
    log_txt_path = os.path.join(args.output_dir, "log.txt")
    fh = logging.FileHandler(log_txt_path)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.info(f"Writing logs to {log_txt_path}.")

    # Prepare GLUE task
    args.task_name = args.dataset
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)
    args.model_type = args.model.split('_')[0]
    current_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(
        current_dir, os.pardir, os.pardir, "my_datasets", args.dataset, args.dataset
    )
    args.data_dir = os.path.normpath(data_dir)
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    if len(args.model.split('_')) == 1:
        args.config_name = ""
        config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)
        model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config)
    else:
        # Load pretrained bert model
        pre_model_dir = os.path.join(
            args.output_dir, os.pardir, args.model.split('_')[0]+'-'+args.dataset+'-base-'
        )
        pre_model_dir = os.path.normpath(pre_model_dir)
        prebert = model_class.from_pretrained(pre_model_dir)
        tokenizer = tokenizer_class.from_pretrained(pre_model_dir, do_lower_case=args.do_lower_case)
        prebert.to(args.device)

        model = MASK_BERT(args, prebert)

        # fix embeddings
        try:
            parameters = filter(lambda p: p.requires_grad, model.bertmodel.bert.embeddings.parameters())
        except:
            parameters = filter(lambda p: p.requires_grad, model.bertmodel.deberta.embeddings.parameters())
        for param in parameters:
            param.requires_grad = False

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    model_wrapper = textattack.models.wrappers.PyTorchModelWrapper(model, tokenizer, args.train_batch_size)
    model = model_wrapper.model
    model.to(args.device)
    tokenizer = model_wrapper.tokenizer
    train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, type='train')
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    train_text, train_labels = load_examples(args, args.task_name, type='train')

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    attack_class = attack_from_args(args)
    # We are adversarial training if the user specified an attack along with
    # the training args.
    adversarial_training = (attack_class is not None) and (not args.check_robustness) and (args.task.split('_')[0] == 'adv')

    # multi-gpu training
    # if num_gpus > 1:
    #     model = torch.nn.DataParallel(model)
    #     logger.info("Using torch.nn.DataParallel.")
    # logger.info(f"Training model across {num_gpus} GPUs")

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    # Start Tensorboard and log hyperparams.
    # from torch.utils.tensorboard import SummaryWriter

    # tb_writer = SummaryWriter(args.output_dir)

    # Use Weights & Biases, if enabled.
    if args.enable_wandb:
        global wandb
        wandb = textattack.shared.utils.LazyLoader("wandb", globals(), "wandb")

        wandb.init(sync_tensorboard=True)

    # Save original args to file
    args_save_path = os.path.join(args.output_dir, "train_args.json")
    _save_args(args, args_save_path)
    logger.info(f"Wrote original training args to {args_save_path}.")

    global_step = 0

    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    epochnum = 0
    best_val_acc = 0
    for _ in train_iterator:
        logging_steps = 0
        preds = None
        out_label_ids = None
        epochnum += 1
        if adversarial_training:
            if epochnum >= args.num_clean_epochs:
                if (epochnum - args.num_clean_epochs) % args.attack_period == 0:
                    # only generate a new adversarial training set every args.attack_period epochs
                    # after the clean epochs
                    if args.task == "adv":
                        logger.info("Attacking model to generate new training set...")

                        train_text1 = train_text.copy()
                        train_labels1 = train_labels.copy()
                        listpack = list(zip(train_text1, train_labels1))
                        random.shuffle(listpack)
                        train_text1[:], train_labels1[:] = zip(*listpack)
                        adv_attack_results = _generate_adversarial_examples(
                            args, model_wrapper, attack_class, list(zip(train_text1[:args.attack_num], train_labels1[:args.attack_num]))
                        )
                        adv_train_text = [r.perturbed_text() for r in adv_attack_results]
                        adv_train_text_temp = train_text + adv_train_text
                        train_labels_temp = train_labels + train_labels1[:args.attack_num]
                        train_dataloader = set_dataloader(
                            args, tokenizer, adv_train_text_temp, train_labels_temp
                        )
                    else:
                        logger.info("Attacking model to generate new training set...")

                        train_text1 = train_text.copy()
                        train_labels1 = train_labels.copy()
                        listpack = list(zip(train_text1, train_labels1))
                        random.shuffle(listpack)
                        train_text1[:], train_labels1[:] = zip(*listpack)
                        adv_attack_results = _generate_adversarial_examples(
                            args, model_wrapper, attack_class, list(zip(train_text1[:args.attack_num], train_labels1[:args.attack_num]))
                        )
                        adv_train_text = [r.perturbed_text() for r in adv_attack_results]
                        train_text_temp = train_text + train_text1[:args.attack_num]
                        adv_train_text_temp = train_text + adv_train_text
                        train_labels_temp = train_labels + train_labels1[:args.attack_num]
                        train_dataloader = set_dataloader_reg(
                            args, tokenizer, train_text_temp, adv_train_text_temp, train_labels_temp
                        )
            else:
                logger.info(f"Running clean epoch {epochnum+1}/{args.num_clean_epochs}")
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        trn_count = 0
        trn_loss, trn_pred_loss, trn_bl_loss, trn_de_loss = 0, 0, 0, 0
        for step, batch in enumerate(epoch_iterator):
            trn_count += 1
            model.train()
            batch = tuple(t.to(args.device) for t in batch)              
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],  # XLM and RoBERTa don't use segment_ids
                      'labels':         batch[3]}
            try:
                outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'],\
                    token_type_ids=inputs['token_type_ids'], labels=inputs['labels'])
            except:
                outputs = model(inputs)
            loss, logits = outputs[:2]

            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            trn_pred_loss += loss.item()

            if len(args.model.split('_')) == 2:
                loss += args.beta * model.infor_loss
                trn_bl_loss += model.infor_loss.item()

            if epochnum >= args.num_clean_epochs and args.task == "adv_reg":
                ori_input_ids = batch[4]
                if args.model_type == 'bert':
                    reg_loss = _cal_discrepance(model, tokenizer, ori_input_ids, inputs['input_ids'])
                else:
                    reg_loss = _deberta_cal_discrepance(model, tokenizer, ori_input_ids, inputs['input_ids'])
                loss += args.gamma * reg_loss
                trn_de_loss += reg_loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            trn_loss += loss.item()
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            global_step += 1
            logging_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        preds = np.argmax(preds, axis=1)
        tr_acc = (preds == out_label_ids).mean()

        # evaluate model
        eval_dataset = load_and_cache_examples(args, args.task_name, tokenizer, type='dev')
        eval_loss, eval_acc = evaluate(args, model, eval_dataset)
        if eval_acc > best_val_acc:
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)
            if len(args.model.split('_')) == 2:
                with open(os.path.join(args.output_dir, args.savename), 'wb') as f:
                    torch.save(model, f)
            else:
                model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                model_to_save.save_pretrained(args.output_dir)
            
            tokenizer.save_pretrained(args.output_dir)
            torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))
            best_val_acc = eval_acc

        print('epoch {} | train_loss {:.6f} | train_acc {:.6f} | dev_loss {:.6f} | dev_acc {:.6f}'.format(epochnum, trn_loss/trn_count, tr_acc, eval_loss, eval_acc))
        print('best_val_acc: {:.6f}'.format(best_val_acc))

    
    # Load a trained model and vocabulary that you have fine-tuned
    del model

    # Load a trained model and vocabulary that you have fine-tuned
    if len(args.model.split('_')) == 2:
        with open(os.path.join(args.output_dir, args.savename), 'rb') as f:
            model = torch.load(f)
    else:
        model = model_class.from_pretrained(args.output_dir)
    tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
    model.to(args.device)

    # Test
    test_dataset = load_and_cache_examples(args, args.task_name, tokenizer, type='test')
    test_loss, test_acc = evaluate(args, model, test_dataset)
    print('\ntest_loss {:.6f} | test_acc {:.6f}'.format(test_loss, test_acc))