# coding=utf-8

# This file has been modified by Yupei Du. The original file is licensed under the Apache License Version 2.0.
# The modifications by Yupei Du are licensed under the MIT license.

# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Fine-tuning a Transformers model for sequence classification on GLUE.
This script cannot fine-tune on MNLI."""
import os

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

from loguru import logger
import argparse
import math

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from datasets import load_metric
from tqdm.auto import tqdm

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    set_seed,
    SchedulerType,
)
from transformers.utils.versions import require_version
import paths
import data_loader as dl
from model_utils import construct_adam_w_optimizer, reinit_bert, reinit_roberta
from mixout import MixLinear

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")
torch.use_deterministic_algorithms(True)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

VALIDATION_METRIC = {'rte': 'accuracy', 'mrpc': 'f1', 'cola': 'matthews_correlation', 'sst2': 'accuracy'}


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a transformers model on a GLUE task")
    parser.add_argument("--log_filename", type=str, default=None, help="Filename for logger.")
    parser.add_argument(
        "--task_name",
        type=str,
        default='sst2',
        help="The name of the glue task to train on.",
        choices=["cola", "mrpc", "rte", "sst2"],
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_length` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        default='bert-large-uncased',
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay to use.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clip to use.")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Total number of training epochs to perform.")
    parser.add_argument("--eval_per_epoch", type=int, default=1, help="Number of evaluation per epoch to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument("--num_warmup_steps", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=123, help="Seed for reproducible training.")
    parser.add_argument("--eval_data_dir", type=str, default=paths.fix_eval_data_dir,
                        help="directory for saving fixed evaluation data.")
    parser.add_argument("--num_reinit_layers", type=int, default=5)
    parser.add_argument("--mixout_p", type=float, default=0.1)
    parser.add_argument("--llrd_rate", type=float, default=0.95)
    parser.add_argument("--mitigation", type=str, default=None,
                        choices=[None, 'wd_pre', 'mixout', 'reinit', 'llrd'])
    parser.add_argument("--save_final_model", action='store_true')
    args = parser.parse_args()

    return args


def evaluate_model(model, eval_dataloader, metric, validation_metric, is_regression=False):
    device = model.device

    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(eval_dataloader):
            batch = batch.to(device)
            outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
            metric.add_batch(predictions=predictions, references=batch["labels"])

    eval_metric = metric.compute()
    return eval_metric, eval_metric[validation_metric]


def train(model, optimizer, scheduler, train_dataloader, eval_dataloader, test_dataloader,
          eval_metric, validation_metric, save_path, final_save_path, eval_per_epoch=3,
          num_train_epochs=5, train_bs=16, is_regression=False, max_grad_norm=1.0,
          max_train_steps=None, writer=None, save_final_model=False):
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {len(train_dataloader)}")
    logger.info(f"  Train batch size = {train_bs}")
    max_train_steps = len(train_dataloader[0]) * num_train_epochs if max_train_steps is None else max_train_steps
    logger.info(f"  Total optimization steps = {max_train_steps}")

    progress_bar = tqdm(range(max_train_steps))
    completed_steps = 0

    loss_records = []
    average_loss = 0
    best_validation_res = float('-inf')

    for epoch in range(num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            model.train()

            # Forward and backward propagation
            batch = batch.to(DEVICE)
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            loss_records.append(loss.item())

            # Update
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            completed_steps += 1
            average_loss = (completed_steps - 1) / completed_steps * average_loss + loss_records[-1] / completed_steps

            if writer is not None:
                writer.add_scalar('Train/loss', loss_records[-1], completed_steps)
                writer.add_scalar('Train/avg_loss', average_loss, completed_steps)

            # Evaluate
            if (step + 1) % int(len(train_dataloader) / eval_per_epoch) == 0:
                logger.info(f'Evaluating after {completed_steps} steps')
                eval_res, validation_res = evaluate_model(
                    model, eval_dataloader, eval_metric, validation_metric, is_regression=is_regression)
                test_res, _ = evaluate_model(
                    model, test_dataloader, eval_metric, validation_metric, is_regression=is_regression)
                logger.info(f'Validation @ epoch {epoch}, step {step}: {eval_res}')
                logger.info(f'Test @ epoch {epoch}, step {step}: {test_res}')
                for metric_term in eval_res:
                    writer.add_scalar(f'Evaluation/{metric_term}', eval_res[metric_term], completed_steps)
                for metric_term in test_res:
                    writer.add_scalar(f'Test/{metric_term}', test_res[metric_term], completed_steps)
                if validation_res > best_validation_res:
                    logger.info(f'Saving model @ epoch {epoch}: {eval_res}')
                    model.save_pretrained(save_path)
                    best_validation_res = validation_res

            if completed_steps >= max_train_steps:
                break

    if save_final_model:
        model.save_pretrained(final_save_path)

    return model


def main():

    args = parse_args()
    if args.log_filename is not None:
        logger.add(args.log_filename)

    # Set random seeds
    if args.seed is not None:
        set_seed(args.seed)

    # model name
    model_name_dict = {'bert-large-uncased': 'bert', 'roberta-large': 'roberta'}
    model_name = model_name_dict[args.model_name_or_path]

    # Load data from disk
    logger.info(f'Fine-tuning {args.model_name_or_path} on GLUE task {args.task_name}\n'
                f'learning rate: {args.learning_rate}')
    # Specify number of labels
    is_regression = args.task_name == "stsb"
    num_labels = 1 if is_regression else 2

    # Load data to dataloader object
    padding = "max_length" if args.pad_to_max_length else False  # store_true
    train_dataloader, _ = dl.load_glue_from_raw(
        args.task_name, args.model_name_or_path, train_bs=args.train_batch_size,
        padding=padding, max_length=args.max_length)
    eval_data_dir = os.path.join(args.eval_data_dir, args.task_name, args.model_name_or_path)
    eval_dataloader = torch.load(os.path.join(eval_data_dir, 'validation.pt'), map_location='cpu')
    test_dataloader = torch.load(os.path.join(eval_data_dir, 'test.pt'), map_location='cpu')

    # Load pretrained model and optimizer
    # Model
    config = AutoConfig.from_pretrained(
        args.model_name_or_path, num_labels=num_labels,
        finetuning_task=args.task_name, cache_dir=paths.hg_config_dir)
    # It's worth noting that BERT has a "pooler" layer itself, applying a 768*768 linear transformation to the first token
    # See https://github.com/google-research/bert/issues/43
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config, cache_dir=paths.hg_ptms_dir)
    if args.mitigation == 'reinit':
        if model_name == 'bert':
            num_layers = len(model.bert.encoder.layer)
            layers_to_reinit = ['pooler'] + list(range(num_layers - 1, num_layers - args.num_reinit_layers - 1, -1))
            logger.info(f'Re-initializing {args.num_reinit_layers} layers')
            model = reinit_bert(model, layers_to_reinit)
        elif model_name == 'roberta':
            num_layers = len(model.roberta.encoder.layer)
            layers_to_reinit = ['pooler'] + list(range(num_layers - 1, num_layers - args.num_reinit_layers - 1, -1))
            logger.info(f'Re-initializing {args.num_reinit_layers} layers')
            model = reinit_roberta(model, layers_to_reinit)
    # from https://github.com/asappresearch/revisit-bert-finetuning/blob/0aa4f4e117ee4422f7cb9355158203e01d6730db/run_glue.py#L800
    elif args.mitigation == 'mixout':
        for sup_module in model.modules():
            for name, module in sup_module.named_children():
                if isinstance(module, nn.Dropout):
                    module.p = 0.0
                if isinstance(module, nn.Linear):
                    target_state_dict = module.state_dict()
                    bias = True if module.bias is not None else False
                    new_module = MixLinear(
                        module.in_features, module.out_features, bias, target_state_dict["weight"], args.mixout_p
                    )
                    new_module.load_state_dict(target_state_dict)
                    setattr(sup_module, name, new_module)
    model = model.to(DEVICE)

    # Construct optimizer
    num_update_steps_per_epoch = len(train_dataloader)  # Update after n steps
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if args.mitigation in ['wd_pre', 'llrd']:
        optimizer, lr_scheduler = construct_adam_w_optimizer(
            model, args.max_train_steps, learning_rate=args.learning_rate,
            weight_decay=args.weight_decay, lr_scheduler_type=args.lr_scheduler_type,
            num_warmup_steps=args.num_warmup_steps, algorithm=args.mitigation, model_name=model_name)
    else:
        optimizer, lr_scheduler = construct_adam_w_optimizer(
            model, args.max_train_steps, learning_rate=args.learning_rate,
            weight_decay=args.weight_decay, lr_scheduler_type=args.lr_scheduler_type,
            num_warmup_steps=args.num_warmup_steps, model_name=model_name)

    # Get the metric function
    metric = load_metric("glue", args.task_name)
    mitigation = 'original' if args.mitigation is None else args.mitigation
    ckpt_dir = os.path.join(f'results/{model_name}/best_validation_ckpt', 'models', args.task_name,
                            str(args.learning_rate), mitigation, f'{args.seed}/')
    tb_path = os.path.join(f'results/{model_name}/best_validation_ckpt', 'tensorboard', args.task_name,
                            str(args.learning_rate), mitigation, f'{args.seed}/')
    os.makedirs(ckpt_dir, exist_ok=True)
    with SummaryWriter(tb_path) as writer:
        _ = train(
            model, optimizer, lr_scheduler, train_dataloader, eval_dataloader, test_dataloader,
            metric, VALIDATION_METRIC[args.task_name],
            save_path=os.path.join(ckpt_dir, 'best_model/'),
            final_save_path=os.path.join(ckpt_dir, 'final_model/'),
            eval_per_epoch=args.eval_per_epoch,
            num_train_epochs=args.num_train_epochs, train_bs=args.train_batch_size,
            is_regression=is_regression, max_grad_norm=args.max_grad_norm,
            max_train_steps=args.max_train_steps,  writer=writer,
            save_final_model=args.save_final_model
        )


if __name__ == "__main__":
    main()
