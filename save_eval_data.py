import os
import argparse
import torch
import data_loader as dl
from transformers import set_seed
import paths

set_seed(123)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default=paths.fix_eval_data_dir)
    parser.add_argument('--model_names', nargs='+', default=['bert-large-uncased', 'roberta-large'])
    parser.add_argument('--tasks', nargs='+', default=['cola', 'mrpc', 'rte', 'sst2'])
    args = parser.parse_args()
    return args


def retrieve_eval_data(model_name, task_name):
    # Load data
    _, [val_dl, test_dl] = dl.load_glue_from_raw(task_name, model_name, eval_bs=64, val_test_split=True)
    val_dl = [batch for batch in val_dl]
    test_dl = [batch for batch in test_dl]

    return val_dl, test_dl


def main():
    args = parse_args()
    save_dir = args.save_dir
    model_names = args.model_names
    tasks = args.tasks

    for task_name in tasks:
        print(f'Task: {task_name}')
        for model_name in model_names:
            print(f'Model: {model_name}')
            val_dl, test_dl = retrieve_eval_data(model_name, task_name)

            specific_save_dir = os.path.join(save_dir, task_name, model_name)
            os.makedirs(specific_save_dir, exist_ok=True)

            val_path = os.path.join(specific_save_dir, 'validation.pt')
            test_path = os.path.join(specific_save_dir, 'test.pt')

            torch.save(val_dl, val_path)
            torch.save(test_dl, test_path)


if __name__ == "__main__":
    main()
