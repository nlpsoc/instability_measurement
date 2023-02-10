import random

from datasets import load_dataset
from torch.utils.data.dataloader import DataLoader

from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    default_data_collator,
)
import paths


TASK2KEYS = {
    "cola": ("sentence", None),  # 8.5k; acceptability; Matthews corr.
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),  # 3.7k; paraphrase; Acc./F1
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),  # 2.5k; NLI; Acc.
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),  # 634; coreference/NLI; Acc.
}


def load_glue_from_raw(task_name, model_name_or_path, train_bs=16, eval_bs=64,
                       padding=False, max_length=128, val_test_split=False):

    def preprocess_function(examples):
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=padding, max_length=max_length, truncation=True)
        result["labels"] = examples["label"]
        return result

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True, cache_dir=paths.hg_tkn_dir)
    raw_datasets = load_dataset("glue", task_name, cache_dir=paths.hg_data_dir)
    sentence1_key, sentence2_key = TASK2KEYS[task_name]

    processed_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        desc="Running tokenizer on dataset",
    )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"]

    if val_test_split:
        val_test = eval_dataset.train_test_split(test_size=0.5)
        eval_dataset, test_dataset = val_test['train'], val_test['test']

    # DataLoaders creation:
    data_collator = default_data_collator if padding else DataCollatorWithPadding(tokenizer)
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=train_bs)
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=eval_bs)

    if val_test_split:
        test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=eval_bs)
        eval_dataloader = [eval_dataloader, test_dataloader]

    return train_dataloader, eval_dataloader
