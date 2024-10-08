""" Finetuning the library models for sequence classification on GLUE."""
import os
import logging
import os
import random
import sys
import math
from dataclasses import dataclass, field
from typing import Optional
from accelerate import Accelerator

import datasets
import numpy as np
from datasets import load_dataset, load_metric
import torch
import json
from torch.nn import CrossEntropyLoss, MSELoss


import transformers
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    TrainingArguments,
    default_data_collator,
    get_scheduler,
    set_seed,
    Trainer
)
# from transformers import BertForSequenceClassification
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from utils_glue import LGTMTeacher,cal_loss
from mpobert import BertModelCustom


import sys

sys.path.append('/home/name/Project/DistillingMPO/OPF')

from compress_tools.MPOtorch import LinearDecomMPO
from compress_tools.Matrix2MPO import MPO
import re
# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.17.0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

# input3072_size = [64,1,1,1,1,1,1,1,1,48]
# input768_size = [32,1,1,1,1,1,1,1,1,24]
# input3072_size2 = [64,1,1,1,1,1,1,1,1,48]
# input768_size2 = [32,1,1,1,1,1,1,1,1,24]

# input3072_size = [64,1,1,1,48]
# input768_size = [32,1,1,1,24]
# input3072_size2 = [64,1,1,1,48]
# input768_size2 = [32,1,1,1,24]

input3072_size = [64,1,1,1,1,1,48]
input768_size = [32,1,1,1,1,1,24]
input3072_size2 = [64,1,1,1,1,1,48]
input768_size2 = [32,1,1,1,1,1,24]

# input3072_size = [64,1,1,48]
# input768_size = [32,1,1,24]
# input3072_size2 = [64,1,1,48]
# input768_size2 = [32,1,1,24]


# input4096_size = [64,64]
# input1024_size = [32,32]

# input512_size = [32,1,1,1,1,1,1,1,1,1,1,1,1,1,16]
# input2048_size = [16,2,1,1,1,1,1,1,1,1,1,1,1,2,32]

SHAPE_CONFIG = {
    "attention":(input768_size,input768_size),
    "FFN_1":(input3072_size,input768_size),
    "FFN_2":(input768_size,input3072_size)
}
SHAPE_CONFIG2 = {
    "attention":(input768_size2,input768_size2),
    "FFN_1":(input3072_size2,input768_size2),
    "FFN_2":(input768_size2,input3072_size2)
}


def _determine_type(name):
    if 'intermediate' in name:
        return 'FFN_1'
    elif 'output' in name and 'attention' not in name:
        return 'FFN_2'
    elif 'query' in name or 'key' in name or 'value' in name or ('attention' in name and 'output' in name):
        return 'attention'
    
    
def loss_layer_fn(model, teacher_model, choose_name):
    loss_layer = 0
    for module_name in choose_name:
        # get_module_from_name 
        ind = re.findall(r"\d+",module_name)[0]
        ind_teacher = str(int(re.findall(r"\d+",module_name)[0]) *2 + 1)

        module_name_teacher = module_name.replace(f"{ind}",f"[{ind_teacher}]")
        module_name = module_name.replace(f"{ind}",f"[{ind}]")
        
        module_name_teacher = module_name_teacher.replace(".weight","")
        module_name = module_name.replace(".weight","")

        pattern = r'bert.encoder.layer.\[\d+\]'


        module_name_teacher = re.search(f'{pattern}(.*)', module_name_teacher).group(1)

        module_name = re.search(f'{pattern}(.*)', module_name).group(1)

  

        # (1) TODO: mpo decomposition for MPO module
        # layer_module.from_pretrained(None, None, mpo_tensor_set, layer_module.bias)
        # (2) mpo decomposition for ori module
        obj_name_teacher, weight_name_teacher = module_name_teacher.rsplit('.',1)
        obj_name, weight_name = module_name.rsplit('.',1)
        
        # 3layer
        for i in [0,1,3,4]:
            module_name_teacher = module_name_teacher + '.tensor_set.' + str(i)
            # # 8layer 
            # module_name_teacher = module_name_teacher + '.tensor_set.4'

            module_name_teacher = module_name_teacher.split('.', 1)[1]
            module_name = module_name.split('.', 1)[1]

            layer_teacher = teacher_model.bert.encoder.layer[int(ind_teacher)].state_dict()[module_name_teacher]
            layer_student = model.bert.encoder.layer[int(ind)].state_dict()[module_name_teacher]

            loss_layer = loss_layer + MSELoss()(layer_teacher, layer_student)
    return loss_layer/len(choose_name)



def fine_grained_decomposition(model,module_name):
    # get_module_from_name 
    ind = re.findall(r"\d+",module_name)[0]
    module_name = module_name.replace(f".{ind}",f"[{ind}]")
    module_name = module_name.replace(".weight","")
    layer_module = eval("model."+module_name)

    type_name = _determine_type(module_name)
    FINE_INPUT_SHAPE, FINE_OUTPUT_SHAPE = SHAPE_CONFIG[type_name]
    mpo = MPO(FINE_INPUT_SHAPE, FINE_OUTPUT_SHAPE, None)
    device = layer_module.weight.device
    # mpo_tensor_set, _, _ = mpo.matrix2mpo(layer_module.get_weight().cpu().detach().numpy()) # .query_mpo

    mpo_tensor_set, _, _ = mpo.matrix2mpo(layer_module.weight.cpu().detach().numpy()) # .query
    bias = layer_module.bias
    # (1) TODO: mpo decomposition for MPO module
    # layer_module.from_pretrained(None, None, mpo_tensor_set, layer_module.bias)
    # (2) mpo decomposition for ori module
    obj_name, weight_name = module_name.rsplit('.',1)
    obj = eval("model."+obj_name)
    setattr(obj, weight_name, LinearDecomMPO(FINE_INPUT_SHAPE, FINE_OUTPUT_SHAPE, None))
    layer_module_new = eval("model."+module_name)

    layer_module_new.from_pretrained(None, None, mpo_tensor_set, bias, device=device)


def fine_grained_decomposition2(model,module_name):
    # get_module_from_name 
    ind = re.findall(r"\d+",module_name)[0]
    module_name = module_name.replace(f".{ind}",f"[{ind}]")
    module_name = module_name.replace(".weight","")
    layer_module = eval("model."+module_name)

    type_name = _determine_type(module_name)
    FINE_INPUT_SHAPE, FINE_OUTPUT_SHAPE = SHAPE_CONFIG2[type_name]
    mpo = MPO(FINE_INPUT_SHAPE, FINE_OUTPUT_SHAPE, None)
    device = layer_module.weight.device
    # mpo_tensor_set, _, _ = mpo.matrix2mpo(layer_module.get_weight().cpu().detach().numpy()) # .query_mpo

    mpo_tensor_set, _, _ = mpo.matrix2mpo(layer_module.weight.cpu().detach().numpy()) # .query
    bias = layer_module.bias
    # (1) TODO: mpo decomposition for MPO module
    # layer_module.from_pretrained(None, None, mpo_tensor_set, layer_module.bias)
    # (2) mpo decomposition for ori module
    obj_name, weight_name = module_name.rsplit('.',1)
    obj = eval("model."+obj_name)
    setattr(obj, weight_name, LinearDecomMPO(FINE_INPUT_SHAPE, FINE_OUTPUT_SHAPE, None))
    layer_module_new = eval("model."+module_name)

    layer_module_new.from_pretrained(None, None, mpo_tensor_set, bias, device=device)





task_to_keys = {
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "stsb": ("sentence1", "sentence2"),
    "cola": ("sentence", None),
    "sst2": ("sentence", None),
}

acc_tasks = ["mnli","qnli", "rte", "sst2"]
f1_tasks = ["mrpc", "qqp"]

logger = logging.getLogger(__name__)

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.dataset_name is not None:
            pass
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task, a training/validation file or a dataset name.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

    # kd setting
    alpha_kd: float = field(
        default=1.0,
        metadata={
            "help": "The weight of kd loss"
        },
    )
    # mode: str = field(
    #     default="kd",
    #     metadata={"help": "The type of kd loss"},
    # )
    temperature: float = field(
        default=1,
        metadata={
            "help": "The temperature."
        },
    )

    # teacher model setting
    teacher_model: str = field(
        default=None,
        metadata={
            "help": "Path of teacher model."
        },
    )
    train_teacher: bool = field(
        default=False,
        metadata={
            "help": "Train teacher or not."
        },
    )
    t_alpha_kd: float = field(
        default=0.4,
        metadata={
            "help": "The weight of kd loss if train_teacher is True."
        },
    )
    t_learning_rate: float = field(
        default=3e-5,
        metadata={
            "help": "The learning rate of teacher."
        },
    )

    # lgtm setting
    use_lgtm: bool = field(
        default=False,
        metadata={
            "help": "Use LGTM or not."
        },
    )
    init_classifier_to_zero: bool = field(
        default=False,
        metadata={
            "help": "Initialize the classifier of the teacher and student to zero."
        },
    )
    num_layers: int = field(
        default=6,
        metadata={
            "help": "The layer number of the student model."
        },
    )

import inspect

def _remove_unused_columns(model, dataset: "datasets.Dataset", description: Optional[str] = None):
    # if not self.args.remove_unused_columns:
    #     return dataset
    # if _signature_columns is None:
    # Inspect model forward signature to keep only the arguments it accepts.
    signature = inspect.signature(model.forward)
    _signature_columns = list(signature.parameters.keys())
    # Labels may be named label or label_ids, the default data collator handles that.
    _signature_columns += ["label", "label_ids"]
    columns = [k for k in _signature_columns if k in dataset.column_names]
    ignored_columns = list(set(dataset['train'].column_names) - set(_signature_columns))
    if len(ignored_columns) > 0:
        dset_description = "" if description is None else f"in the {description} set "
        logger.info(
            f"The following columns {dset_description} don't have a corresponding argument in "
            f"`{model.__class__.__name__}.forward` and have been ignored: {', '.join(ignored_columns)}."
        )
    return dataset.remove_columns(ignored_columns)

def init_classifier_as_zero(model):
    for params in model.classifier.parameters():
        params.data.fill_(0.0)
    
def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    accelerator = Accelerator()
    os.makedirs(training_args.output_dir, exist_ok=True)
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    #
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset("glue", data_args.task_name, cache_dir=model_args.cache_dir)
    elif data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir
        )
    else:
        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.
        data_files = {"train": data_args.train_file, "validation": data_args.validation_file}

        # Get the test dataset: you can provide your own CSV/JSON test file (see below)
        # when you use `do_predict` without specifying a GLUE benchmark task.
        if training_args.do_predict:
            if data_args.test_file is not None:
                train_extension = data_args.train_file.split(".")[-1]
                test_extension = data_args.test_file.split(".")[-1]
                assert (
                    test_extension == train_extension
                ), "`test_file` should have the same extension (csv or json) as `train_file`."
                data_files["test"] = data_args.test_file
            else:
                raise ValueError("Need either a GLUE task or a test file for `do_predict`.")

        for key in data_files.keys():
            logger.info(f"load a local file for {key}: {data_files[key]}")

        if data_args.train_file.endswith(".csv"):
            # Loading a dataset from local csv files
            raw_datasets = load_dataset("csv", data_files=data_files, cache_dir=model_args.cache_dir)
        else:
            # Loading a dataset from local json files
            raw_datasets = load_dataset("json", data_files=data_files, cache_dir=model_args.cache_dir)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    # student model
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        num_hidden_layers=model_args.num_layers,
        output_hidden_states = True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    
    # teacher model(only used in training)
    if training_args.do_train:
        t_config = AutoConfig.from_pretrained(model_args.teacher_model, num_labels=num_labels, finetuning_task=data_args.task_name, output_hidden_states = True)
        teacher_model = AutoModelForSequenceClassification.from_pretrained(
            model_args.teacher_model,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=t_config,
        )



    #####################################################################################进行MPO#########################################################################################################################
#######################################################################################################################################################################################

    # Preprocessing the raw_datasets
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    if model.config.id2label:
        label_to_id = {v: k for k, v in model.config.id2label.items()}
        model.config.label2id = label_to_id

    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and data_args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif data_args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        return result

    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
        raw_datasets = _remove_unused_columns(model, raw_datasets)
        
    if training_args.do_train:
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
            
    if training_args.do_eval:
        if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

    if training_args.do_predict or data_args.task_name is not None or data_args.test_file is not None:
        if "test" not in raw_datasets and "test_matched" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test_matched" if data_args.task_name == "mnli" else "test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Get the metric function
    if data_args.task_name is not None:
        metric = load_metric("glue", data_args.task_name)
    else:
        metric = load_metric("accuracy")

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if data_args.task_name is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None
    ####################################################################################进行MPO#########################################################################################################################
    # 对student进行mpo
    print("#########################################################################################################")
    print(str(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print("#########################################################################################################")
    # 获取name
    basic_choose_name = {}
    gradient_mask = dict()
    gradient_name_mask = dict()
    for name, params in model.named_parameters():
        if 'layer' in name and _determine_type(name) in 'FFN_1,FFN_2,attention' and name not in basic_choose_name:
            gradient_mask[params] = params.new_ones(params.size())
            gradient_name_mask[name] = 0
    choose_name = set()
    for target_name in ['intermediate.dense.weight','output.dense.weight','attention.self.query.weight','attention.self.key.weight','attention.self.value.weight','attention.output.dense.weight']:
        target_group = {k:v for k,v in gradient_name_mask.items() if (target_name in k) and ("bias" not in k) and (target_name not in basic_choose_name)}
        sort_g = dict(sorted(target_group.items(), key=lambda x :x[1], reverse=True))
        choose_name.update(sort_g)

    # 分解
    for name, params in model.named_parameters():
        if name in choose_name:
            print(name)
            fine_grained_decomposition(model, name)
    #对student进行mpo
    print("#########################################################################################################")
    print("分解后参数量："+str(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print("#########################################################################################################")
   ######################################################################################################################################################################################
    choose_name_student = choose_name
    ###################################################################################进行MPO#########################################################################################################################
    #之对teacher进行mpo
    print("#########################################################################################################")
    print("分解前参数量："+str(sum(p.numel() for p in teacher_model.parameters() if p.requires_grad)))
    print("#########################################################################################################")
    # 获取name
    basic_choose_name = {}
    gradient_mask = dict()
    gradient_name_mask = dict()
    for name, params in teacher_model.named_parameters():
        if 'layer' in name and _determine_type(name) in 'FFN_1,FFN_2,attention' and name not in basic_choose_name:
            gradient_mask[params] = params.new_ones(params.size())
            gradient_name_mask[name] = 0
    choose_name = set()
    for target_name in ['intermediate.dense.weight','output.dense.weight','attention.self.query.weight','attention.self.key.weight','attention.self.value.weight','attention.output.dense.weight']:
        target_group = {k:v for k,v in gradient_name_mask.items() if (target_name in k) and ("bias" not in k) and (target_name not in basic_choose_name)}
        sort_g = dict(sorted(target_group.items(), key=lambda x :x[1], reverse=True))
        choose_name.update(sort_g)

    # 分解
    for name, params in teacher_model.named_parameters():
        if name in choose_name:
            print(name)
            fine_grained_decomposition2(teacher_model, name)
    #之对student进行mpo
    print("#########################################################################################################")
    print("分解后参数量："+str(sum(p.numel() for p in teacher_model.parameters() if p.requires_grad)))
    print("#########################################################################################################")
    ##############################################################################################################################################################################################################

    print("######################################student_model##########################################################")
    for name, params in model.named_parameters():
        print(name)
        print(params.shape)
    print("######################################teacher_model##########################################################")
    for name, params in teacher_model.named_parameters():
        print(name)
        print(params.shape)


    # prediction
    if training_args.do_predict:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            data_collator=data_collator
        )

        logger.info("*** Predict ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        predict_datasets = [predict_dataset]
        
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            predict_datasets.append(raw_datasets["test_mismatched"])

        for predict_dataset, task in zip(predict_datasets, tasks):
            # Removing the `label` columns because it contains -1 and Trainer won't like that.
            
            predict_dataset = predict_dataset.remove_columns("label")
            predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
            predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)
            
            output_predict_file = os.path.join(training_args.output_dir, f"{task}.tsv")
            if trainer.is_world_process_zero():
                with open(output_predict_file, "w") as writer:
                    logger.info(f"***** Predict results {task} *****")
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if is_regression:
                            writer.write(f"{index}\t{item:3.3f}\n")
                        else:
                            item = label_list[item]
                            writer.write(f"{index}\t{item}\n")
        return

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=training_args.per_device_train_batch_size
    )
    eval_dataset_size = len(eval_dataset)
    split_point = int(eval_dataset_size // 2)
    dev_dataset = eval_dataset.select(range(split_point))
    test_dataset = eval_dataset.select(range(split_point, eval_dataset_size))

    # 创建新的数据加载器
    eval_dataloader = DataLoader(dev_dataset, collate_fn=data_collator, batch_size=training_args.per_device_eval_batch_size)
    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=training_args.per_device_eval_batch_size)
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=training_args.learning_rate)

    if model_args.train_teacher:
        t_optimizer = AdamW(teacher_model.parameters(), lr=model_args.t_learning_rate)
    
        if model_args.init_classifier_to_zero:
            init_classifier_as_zero(teacher_model)
            init_classifier_as_zero(model)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / training_args.gradient_accumulation_steps)
    max_train_steps = int(training_args.num_train_epochs * num_update_steps_per_epoch)
    num_warmup_steps = max_train_steps*training_args.warmup_ratio

    lr_scheduler = get_scheduler(
        name=training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_train_steps,
    )

    if model_args.train_teacher:
        t_lr_scheduler = get_scheduler(
            name=training_args.lr_scheduler_type,
            optimizer=t_optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=max_train_steps,
        )
        
        model, teacher_model, optimizer, t_optimizer, train_dataloader, eval_dataloader, test_dataloader = accelerator.prepare(
            model, teacher_model, optimizer, t_optimizer, train_dataloader, eval_dataloader, test_dataloader
        )

        if model_args.use_lgtm:
            held_iter = iter(eval_dataloader)
            model_total = LGTMTeacher(teacher_model, model, model_args.alpha_kd, model_args.t_alpha_kd,
                                    t_optimizer, t_lr_scheduler, model_args.temperature)
            
    else:
        model, teacher_model, optimizer, train_dataloader, eval_dataloader, test_dataloader = accelerator.prepare(
                model, teacher_model, optimizer, train_dataloader, eval_dataloader, test_dataloader
            )
     
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / training_args.gradient_accumulation_steps)
    total_batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {int(training_args.num_train_epochs)}")
    logger.info(f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {training_args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(max_train_steps))
    completed_steps = 0
    best_metric = 0.0
    t_best_metric = 0.0
    best_dev = {}
    best_test = {}

    for epoch in range(int(training_args.num_train_epochs)):
        for step, batch in enumerate(train_dataloader):
            if model_args.train_teacher:
                if model_args.use_lgtm:
                    model.train()
                    teacher_model.train()
                    # fetch the batch for calculating student's feedback 
                    try:
                        held_inputs = next(held_iter)
                    except:
                        held_iter = iter(eval_dataloader)
                        held_inputs = next(held_iter)
                    # update the teacher
                    model_total.step(batch, held_inputs, optimizer)
                else:
                    model.eval()
                    teacher_model.train()
                    with torch.no_grad():
                        outputs = model(**batch)
                        logits = outputs.logits
                    teacher_outputs = teacher_model(**batch)
                    t_loss, t_logits = teacher_outputs.loss, teacher_outputs.logits
                    t_loss = model_args.t_alpha_kd * cal_loss(t_logits,logits, model_args.temperature) + (1 - model_args.t_alpha_kd) * t_loss
    #####################################################################################对齐#########################################################################################################################
                    loss_layer = 0

                    for i in range(len(outputs.hidden_states)-1):
                        loss_layer = loss_layer + torch.nn.MSELoss()(outputs.hidden_states[i+1],teacher_outputs.hidden_states[2*i+1+1])
                    # t_loss = loss_layer/(len(outputs.hidden_states)-1) + t_loss
    ###############################################################################################################################################################################################################

                    # update the teacher
                    t_loss.backward()
                
                    if step % training_args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                        t_optimizer.step()
                        t_lr_scheduler.step()
                        t_optimizer.zero_grad()

                # use teacher logits as soft labels
                teacher_model.eval()
                model.train()
                            
                with torch.no_grad():
                    teacher_outputs = teacher_model(**batch)
                    t_logits = teacher_outputs.logits

                outputs = model(**batch)
                loss, logits = outputs.loss, outputs.logits
                loss = model_args.alpha_kd * cal_loss(logits, t_logits, model_args.temperature) + (1-model_args.alpha_kd) * loss
            else: # no teacher update
                # student loss
                model.train()
                outputs = model(**batch)
                loss, logits = outputs.loss, outputs.logits
                teacher_model.eval()
                with torch.no_grad():
                    teacher_outputs = teacher_model(**batch)
                    t_logits = teacher_outputs.logits

                loss = model_args.alpha_kd * cal_loss(logits, t_logits, model_args.temperature) + (1 - model_args.alpha_kd) * loss
    #####################################################################################对齐#########################################################################################################################
            # update the student 
            loss_layer = 0
            for i in range(1,int(len(outputs.hidden_states))):
                loss_layer = loss_layer + torch.nn.MSELoss()(outputs.hidden_states[i],teacher_outputs.hidden_states[2*i])
     ################################################################################################################################################################################################################
            loss_layer = loss_layer_fn(model, teacher_model, choose_name_student)
            loss = loss / int(training_args.gradient_accumulation_steps)
            loss = loss +  0.0001*loss_layer/(len(outputs.hidden_states)-1)
            print("loss:")
            print(loss)
            loss.backward()

            if step % training_args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            # We keep track of the loss at each epoch
            if completed_steps % training_args.eval_steps == 0 or completed_steps == max_train_steps:
                model.eval()
                samples_seen = 0


                
                # test結果
                for step, batch in enumerate(test_dataloader):
                    with torch.no_grad():
                        outputs = model(**batch)
                    predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
                    predictions, references = accelerator.gather((predictions, batch["labels"]))
                    # If we are in a multiprocess environment, the last batch has duplicates
                    if accelerator.num_processes > 1:
                        if step == len(test_dataloader) - 1:
                            predictions = predictions[: len(test_dataloader.dataset) - samples_seen]
                            references = references[: len(test_dataloader.dataset) - samples_seen]
                        else:
                            samples_seen += references.shape[0]
                    metric.add_batch(
                        predictions=predictions,
                        references=references,
                    )           
                    # eval_metric = metric.compute(predictions=predictions, references=batch["labels"])
                    # acc_metric += eval_metric['accuracy']
                    # f1_metric += eval_metric['f1']
                test_metric = metric.compute()
                
                logger.info("***** test Results*****")
                logger.info(f"  Training step = {completed_steps}")
                for key, value in test_metric.items():
                    logger.info(f" test_{key}:{value} ")
           
                if data_args.task_name in acc_tasks:
                    metric_key = "accuracy"
                elif data_args.task_name in f1_tasks:
                    metric_key = "f1"
                if data_args.task_name == 'stsb':
                    metric_key = "spearmanr"
                if data_args.task_name =='cola':
                    metric_key = 'matthews_correlation'




                # evaluation
                for step, batch in enumerate(eval_dataloader):
                    with torch.no_grad():
                        outputs = model(**batch)
                    predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
                    predictions, references = accelerator.gather((predictions, batch["labels"]))
                    # If we are in a multiprocess environment, the last batch has duplicates
                    if accelerator.num_processes > 1:
                        if step == len(eval_dataloader) - 1:
                            predictions = predictions[: len(eval_dataloader.dataset) - samples_seen]
                            references = references[: len(eval_dataloader.dataset) - samples_seen]
                        else:
                            samples_seen += references.shape[0]
                    metric.add_batch(
                        predictions=predictions,
                        references=references,
                    )           
                    # eval_metric = metric.compute(predictions=predictions, references=batch["labels"])
                    # acc_metric += eval_metric['accuracy']
                    # f1_metric += eval_metric['f1']
                eval_metric = metric.compute()
                
                logger.info("***** Evaluation Results*****")
                logger.info(f"  Training step = {completed_steps}")
                for key, value in eval_metric.items():
                    logger.info(f" eval_{key}:{value} ")
           
                if data_args.task_name in acc_tasks:
                    metric_key = "accuracy"
                elif data_args.task_name in f1_tasks:
                    metric_key = "f1"
                if data_args.task_name == 'stsb':
                    metric_key = "spearmanr"
                if data_args.task_name =='cola':
                    metric_key = 'matthews_correlation'


                if eval_metric[metric_key] > best_metric:
                    best_metric = eval_metric[metric_key]
                    tokenizer.save_pretrained(training_args.output_dir)
                    model.save_pretrained(training_args.output_dir)
                    path = os.path.join(training_args.output_dir, "eval_results.json")
                    with open(path, "w") as f:
                        json.dump(eval_metric, f, indent=4, sort_keys=True)
                    best_dev = eval_metric
                    best_test = test_metric
                print("dev:")
                print(best_dev)
                print("test:")
                print(best_test)
                # 存储
                data = {'dev': best_dev, 'test': best_test}
                # 将数据存储为文本文件
                file_path = f"/mnt/name/data/checkpoint/nlp/lgtm/evalue/{data_args.task_name}_{training_args.learning_rate}_{model_args.t_learning_rate}_len{len(input3072_size)}_len2{len(input3072_size2)}.txt"
                with open(file_path, 'w') as file:
                    for key, value in data.items():
                        file.write(f"{key}: {value}\n")
                # teacher evaluation 
                teacher_model.eval()
                samples_seen = 0
                for step, batch in enumerate(eval_dataloader):
                    with torch.no_grad():
                        outputs = teacher_model(**batch)
                    predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
                    predictions, references = accelerator.gather((predictions, batch["labels"]))
                    if accelerator.num_processes > 1:
                        if step == len(eval_dataloader) - 1:
                            predictions = predictions[: len(eval_dataloader.dataset) - samples_seen]
                            references = references[: len(eval_dataloader.dataset) - samples_seen]
                        else:
                            samples_seen += references.shape[0]
                    metric.add_batch(
                        predictions=predictions,
                        references=references,
                    )



                eval_metric = metric.compute()
                logger.info("***** Teacher Evaluation Results*****")
                logger.info(f"  Training step = {completed_steps}")
                for key, value in eval_metric.items():
                    logger.info(f" eval_{key}:{value} ")  
                
                if eval_metric[metric_key] > t_best_metric:
                    t_best_metric = eval_metric[metric_key]
                    path = os.path.join(training_args.output_dir, "teacher_eval_results.json")
                    with open(path, "w") as f:
                        json.dump(eval_metric, f, indent=4, sort_keys=True) 

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()

if __name__ == "__main__":
    main()
