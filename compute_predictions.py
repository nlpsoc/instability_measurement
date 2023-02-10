import os
import argparse

import torch
from loguru import logger
import deepdish as dd
from transformers import set_seed

from model_utils import load_model, compute_pred
from datasets import load_metric
import paths

set_seed(123)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_names', nargs='+', default=['rte', 'mrpc', 'cola'])
    parser.add_argument('--model_names', nargs='+', default=['bert-large-uncased', 'roberta-large'])
    parser.add_argument('--lrs', nargs='+', default=['2e-05'])
    parser.add_argument('--mitigation_methods', nargs='+', default=['original'])
    parser.add_argument('--seeds', nargs='+', default=[str(num) for num in range(20)])
    args = parser.parse_args()
    return args


def main():
    model_name_dict = {'bert-large-uncased': 'bert', 'roberta-large': 'roberta'}

    args = parse_args()
    task_names = args.task_names
    lrs = args.lrs
    mitigation_methods = args.mitigation_methods
    seeds = [int(seed) for seed in args.seeds]
    model_names = args.model_names

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # measure_comparison exploration
    for model_name in model_names:
        logger.info(f'Processing {model_name}')
        pm_save_dir = f'results/{model_name_dict[model_name]}/instability_eval/predictions'
        os.makedirs(pm_save_dir, exist_ok=True)
        for task_name in task_names:
            metric = load_metric("glue", task_name)

            val_data_path = os.path.join(paths.fix_eval_data_dir, task_name, f'{model_name}/test.pt')
            val_dl = torch.load(val_data_path, map_location=device)
            test_data_path = os.path.join(paths.fix_eval_data_dir, task_name, f'{model_name}/test.pt')
            test_dl = torch.load(test_data_path, map_location=device)

            for lr in lrs:
                logger.info(f'task: {task_name}, lr: {lr}')
                for mitigation_method in mitigation_methods:
                    val_logits_dict = {}
                    val_metric_dict = {}
                    val_label_dict = {}
                    test_logits_dict = {}
                    test_metric_dict = {}
                    test_label_dict = {}

                    for seed in seeds:
                        model_path = os.path.join(
                            f'results/{model_name_dict[model_name]}/best_validation_ckpt/models', task_name, lr,
                            mitigation_method, str(seed), 'best_model/')
                        model = load_model(model_path).to(device)
                        model.eval()

                        # dict(metric: val); n_samples, n_classes
                        val_metric, val_logits, val_labels = compute_pred(model, val_dl, metric, return_label=True)
                        test_metric, test_logits, test_labels = compute_pred(model, test_dl, metric, return_label=True)

                        val_metric_dict[seed], val_logits_dict[seed], val_label_dict[seed] = \
                            val_metric, val_logits.detach().cpu().numpy(), val_labels.detach().cpu().numpy()
                        test_metric_dict[seed], test_logits_dict[seed], test_label_dict[seed] = \
                            test_metric, test_logits.detach().cpu().numpy(), test_labels.detach().cpu().numpy()

                    save_data = {
                        'validation': {'eval_metric': val_metric_dict, 'logits': val_logits_dict, 'labels': val_label_dict},
                        'test': {'eval_metric': test_metric_dict, 'logits': test_logits_dict, 'labels': test_label_dict},
                    }
                    save_path = os.path.join(pm_save_dir, f'{task_name}_{lr}_{mitigation_method}.h5')
                    # dd.io.save(save_path, save_data)


if __name__ == '__main__':
    main()