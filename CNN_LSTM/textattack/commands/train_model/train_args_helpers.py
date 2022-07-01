import os
import re

import textattack
from textattack.commands.attack.attack_args import ATTACK_RECIPE_NAMES
from textattack.commands.attack.attack_args_helpers import ARGS_SPLIT_TOKEN
from textattack.commands.augment import AUGMENTATION_RECIPE_NAMES

logger = textattack.shared.logger


def prepare_dataset_for_training(datasets_dataset):
    """Changes an `datasets` dataset into the proper format for
    tokenization."""

    def prepare_example_dict(ex):
        """Returns the values in order corresponding to the data.

        ex:
            'Some text input'
        or in the case of multi-sequence inputs:
            ('The premise', 'the hypothesis',)
        etc.
        """
        values = list(ex.values())
        if len(values) == 1:
            return values[0]
        return tuple(values)

    text, outputs = zip(*((prepare_example_dict(x[0]), x[1]) for x in datasets_dataset))
    return list(text), list(outputs)


def dataset_from_args(args):
    """Returns a tuple of ``HuggingFaceDataset`` for the train and test
    datasets for ``args.dataset``."""
    dataset_args = args.dataset.split(ARGS_SPLIT_TOKEN)
    # TODO `HuggingFaceDataset` -> `HuggingFaceDataset`
    if args.dataset_train_split:
        train_dataset = textattack.datasets.HuggingFaceDataset(
            *dataset_args, split=args.dataset_train_split
        )
    else:
        try:
            train_dataset = textattack.datasets.HuggingFaceDataset(
                *dataset_args, split="train"
            )
            args.dataset_train_split = "train"
        except KeyError:
            raise KeyError(f"Error: no `train` split found in `{args.dataset}` dataset")
    train_text, train_labels = prepare_dataset_for_training(train_dataset)

    if args.dataset_dev_split:
        eval_dataset = textattack.datasets.HuggingFaceDataset(
            *dataset_args, split=args.dataset_dev_split
        )
    else:
        # try common dev split names
        try:
            eval_dataset = textattack.datasets.HuggingFaceDataset(
                *dataset_args, split="dev"
            )
            args.dataset_dev_split = "dev"
        except KeyError:
            try:
                eval_dataset = textattack.datasets.HuggingFaceDataset(
                    *dataset_args, split="eval"
                )
                args.dataset_dev_split = "eval"
            except KeyError:
                try:
                    eval_dataset = textattack.datasets.HuggingFaceDataset(
                        *dataset_args, split="validation"
                    )
                    args.dataset_dev_split = "validation"
                except KeyError:
                    try:
                        eval_dataset = textattack.datasets.HuggingFaceDataset(
                            *dataset_args, split="test"
                        )
                        args.dataset_dev_split = "test"
                    except KeyError:
                        raise KeyError(
                            f"Could not find `dev`, `eval`, `validation`, or `test` split in dataset {args.dataset}."
                        )
    eval_text, eval_labels = prepare_dataset_for_training(eval_dataset)

    return train_text, train_labels, eval_text, eval_labels


class My_DATA(object):
    def __init__(self, args, file_path):
        self.train_text, self.train_label, self.dev_text, self.dev_label, self.test_text, self.test_label = self.read_data(args, file_path)

    def read_data(self, args, file_path):
        file_train = os.path.join(file_path, "train.tsv")
        file_dev = os.path.join(file_path, "dev.tsv")
        file_test = os.path.join(file_path, "test.tsv")

        train_text = []
        train_labels = []
        with open(file_train, "r", encoding="utf-8-sig") as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                a = line.split('\t')
                train_text.append(a[0][:-1])
                train_labels.append(int(a[1][:-1]))
        
        eval_text = []
        eval_labels = []
        with open(file_dev, "r", encoding="utf-8-sig") as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                a = line.split('\t')
                eval_text.append(a[0][:-1])
                eval_labels.append(int(a[1][:-1]))

        test_text = []
        test_labels = []
        with open(file_test, "r", encoding="utf-8-sig") as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                a = line.split('\t')
                test_text.append(a[0][:-1])
                test_labels.append(int(a[1][:-1]))

        print('train num: {}'.format(len(train_labels)))
        print('dev num: {}'.format(len(eval_labels)))
        print('test num: {}'.format(len(test_labels)))

        self.wordvocab = self.build_vocab(args, train_text)

        return train_text, train_labels, eval_text, eval_labels, test_text, test_labels

    
    def preprocess(self, s):
        s = s.lower()
        # s = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", s)
        s = re.sub(r"[^A-Za-z0-9\'\`]", " ", s)
        s = re.sub(r"\'s", " \'s", s)
        s = re.sub(r"\'ve", " \'ve", s)
        s = re.sub(r"n\'t", " n\'t", s)
        s = re.sub(r"\'re", " \'re", s)
        s = re.sub(r"\'d", " \'d", s)
        s = re.sub(r"\'ll", " \'ll", s)
        s = re.sub(r",", " , ", s)
        s = re.sub(r"!", " ! ", s)
        s = re.sub(r"\(", " \( ", s)
        s = re.sub(r"\)", " \) ", s)
        s = re.sub(r"\?", " ? ", s)
        s = re.sub(r"\s{2,}", " ", s)
        s = s.split()
        return s
    
    
    def build_vocab(self, args, traintext):
        dic = {}
        low_frequency = args.low_freq
        for txt in traintext:
            txt = self.preprocess(txt)
            for w in txt:
                if w not in dic.keys():
                    dic[w] = 1
                else:
                    dic[w] += 1
        dic_sort = sorted(dic.items(), key=lambda d: d[1], reverse=True)
        vocab = [k[0] for k in dic_sort if k[1] > low_frequency]
        vocab.insert(0, '<unk>')
        vocab.insert(0, '<pad>')
        print('vocab:{}'.format(len(vocab)))
        return vocab


def dataset_from_my_file(args):
    """Read data from my file"""
    output_dir = args.output_dir
    mydata_dir = os.path.join(
            output_dir, os.pardir, os.pardir, os.pardir, "textattack", "my_datasets", args.dataset
        )
    file_path = os.path.normpath(mydata_dir)
    data = My_DATA(args, file_path)

    return data


def model_from_args(train_args, num_labels, model_path=None):
    """Constructs a model from its `train_args.json`.

    If huggingface model, loads from model hub address. If TextAttack
    lstm/cnn, loads from disk (and `model_path` provides the path to the
    model).
    """
    if train_args.model == "lstm":
        textattack.shared.logger.info("Loading textattack model: LSTMForClassification")
        model = textattack.models.helpers.LSTMForClassification(
            max_seq_length=train_args.max_length,
            num_labels=num_labels,
            emb_layer_trainable=False,
        )
        if model_path:
            model.load_from_disk(model_path)

        model = textattack.models.wrappers.PyTorchModelWrapper(model, model.tokenizer)
    elif train_args.model == "cnn":
        textattack.shared.logger.info(
            "Loading textattack model: WordCNNForClassification"
        )
        model = textattack.models.helpers.WordCNNForClassification(
            max_seq_length=train_args.max_length,
            num_labels=num_labels,
            emb_layer_trainable=False,
        )
        if model_path:
            model.load_from_disk(model_path)

        model = textattack.models.wrappers.PyTorchModelWrapper(model, model.tokenizer)
    else:
        import transformers

        textattack.shared.logger.info(
            f"Loading transformers AutoModelForSequenceClassification: {train_args.model}"
        )
        config = transformers.AutoConfig.from_pretrained(
            train_args.model, num_labels=num_labels, finetuning_task=train_args.dataset
        )
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            train_args.model,
            config=config,
        )
        tokenizer = textattack.models.tokenizers.AutoTokenizer(
            train_args.model, use_fast=True, max_length=train_args.max_length
        )

        model = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)

    return model


def model_from_my_file(train_args, wordvocab, num_labels, model_path=None): 
    """Constructs a model from my file
    """
    if train_args.model == "lstm":
        textattack.shared.logger.info("Loading model: MyLSTM")
        model = textattack.models.helpers.MyLSTM(
            train_args,
            num_labels=num_labels,
            emb_layer_trainable=False,
            wordvocab = wordvocab,
        )
        model = textattack.models.wrappers.PyTorchModelWrapper(model, model.tokenizer)

    elif train_args.model == "lstm_mask":
        textattack.shared.logger.info("Loading model: MyLSTM_VMASK")
        model = textattack.models.helpers.MyLSTMVmask(
            train_args,
            num_labels=num_labels,
            emb_layer_trainable=False,
            wordvocab = wordvocab,
        )
        model = textattack.models.wrappers.PyTorchModelWrapper(model, model.lstmmodel.tokenizer)

    elif train_args.model == "cnn":
        textattack.shared.logger.info("Loading model: MyCNN")
        model = textattack.models.helpers.MyCNN(
            train_args,
            num_labels=num_labels,
            emb_layer_trainable=False,
            wordvocab = wordvocab,
        )
        model = textattack.models.wrappers.PyTorchModelWrapper(model, model.tokenizer)

    elif train_args.model == "cnn_mask":
        textattack.shared.logger.info("Loading model: MyCNN_VMASK")
        model = textattack.models.helpers.MyCNNVmask(
            train_args,
            num_labels=num_labels,
            emb_layer_trainable=False,
            wordvocab = wordvocab,
        )
        model = textattack.models.wrappers.PyTorchModelWrapper(model, model.cnnmodel.tokenizer)

    elif train_args.model == "cnn_group":
        textattack.shared.logger.info("Loading model: MyCNN_GROUP")
        model = textattack.models.helpers.MyCNNGroup(
            train_args,
            num_labels=num_labels,
            emb_layer_trainable=False,
            wordvocab = wordvocab,
        )
        model = textattack.models.wrappers.PyTorchModelWrapper(model, model.cnnmodel.tokenizer)

    elif train_args.model == "logre":
        textattack.shared.logger.info("Loading model: LogisticRegression")
        model = textattack.models.helpers.LogisticRegression(
            train_args,
            wordvocab=wordvocab,
            num_classes=num_labels,
        )
        model = textattack.models.wrappers.PyTorchModelWrapper(model, model.tokenizer)

    elif train_args.model == "logre_mask":
        textattack.shared.logger.info("Loading model: MyLRVmask")
        model = textattack.models.helpers.MyLRVmask(
            train_args,
            num_labels=num_labels,
            emb_layer_trainable=False,
            wordvocab = wordvocab,
        )
        model = textattack.models.wrappers.PyTorchModelWrapper(model, model.lrmodel.tokenizer)
        
    else:
        import transformers

        textattack.shared.logger.info(
            f"Loading transformers AutoModelForSequenceClassification: {train_args.model}"
        )
        config = transformers.AutoConfig.from_pretrained(
            train_args.model, num_labels=num_labels, finetuning_task=train_args.dataset
        )
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            train_args.model,
            config=config,
        )
        tokenizer = textattack.models.tokenizers.AutoTokenizer(
            train_args.model, use_fast=True, max_length=train_args.max_length
        )

        model = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)

    return model


def attack_from_args(args):
    # note that this returns a recipe type, not an object
    # (we need to wait to have access to the model to initialize)
    attack_class = None
    if args.attack:
        if args.attack in ATTACK_RECIPE_NAMES:
            attack_class = eval(ATTACK_RECIPE_NAMES[args.attack])
        else:
            raise ValueError(f"Unrecognized attack recipe: {args.attack}")

    # check attack-related args
    assert args.num_clean_epochs > 0, "--num-clean-epochs must be > 0"
    assert not (
        args.check_robustness and not (args.attack)
    ), "--check_robustness must be used with --attack"

    return attack_class


def augmenter_from_args(args):
    augmenter = None
    if args.augment:
        if args.augment in AUGMENTATION_RECIPE_NAMES:
            augmenter = eval(AUGMENTATION_RECIPE_NAMES[args.augment])(
                pct_words_to_swap=args.pct_words_to_swap,
                transformations_per_example=args.transformations_per_example,
            )
        else:
            raise ValueError(f"Unrecognized augmentation recipe: {args.augment}")
    return augmenter


def write_readme(args, best_eval_score, best_eval_score_epoch):
    # Save args to file
    readme_save_path = os.path.join(args.output_dir, "README.md")
    dataset_name = (
        args.dataset.split(ARGS_SPLIT_TOKEN)[0]
        if ARGS_SPLIT_TOKEN in args.dataset
        else args.dataset
    )
    task_name = "regression" if args.do_regression else "classification"
    loss_func = "mean squared error" if args.do_regression else "cross-entropy"
    metric_name = "pearson correlation" if args.do_regression else "accuracy"
    epoch_info = f"{best_eval_score_epoch} epoch" + (
        "s" if best_eval_score_epoch > 1 else ""
    )
    readme_text = f"""
## TextAttack Model Card

This `{args.model}` model was fine-tuned for sequence classification using TextAttack
and the {dataset_name} dataset loaded using the `datasets` library. The model was fine-tuned
for {args.num_train_epochs} epochs with a batch size of {args.batch_size}, a learning
rate of {args.learning_rate}, and a maximum sequence length of {args.max_length}.
Since this was a {task_name} task, the model was trained with a {loss_func} loss function.
The best score the model achieved on this task was {best_eval_score}, as measured by the
eval set {metric_name}, found after {epoch_info}.

For more information, check out [TextAttack on Github](https://github.com/QData/TextAttack).

"""

    with open(readme_save_path, "w", encoding="utf-8") as f:
        f.write(readme_text.strip() + "\n")
    logger.info(f"Wrote README to {readme_save_path}.")
