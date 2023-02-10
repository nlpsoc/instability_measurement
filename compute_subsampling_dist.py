import os
import argparse
from collections import defaultdict

import torch
import numpy as np
from loguru import logger
import deepdish as dd
from transformers import set_seed

from model_utils import load_model, compute_bert_cls_rep, compute_roberta_cls_rep
import paths
import representation_measure as rm

set_seed(123)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_names', nargs='+', default=['rte', 'mrpc', 'cola'])
    parser.add_argument('--lrs', nargs='+', default=['2e-05'])
    parser.add_argument('--model_names', nargs='+', default=['bert-large-uncased', 'roberta-large'])
    parser.add_argument('--seeds', nargs='+', default=[str(num) for num in range(20)])
    parser.add_argument('--sample_rate', type=float, default=0.5)
    parser.add_argument('--sample_times', type=int, default=4)
    args = parser.parse_args()
    return args


def main():
    model_name_dict = {'bert-large-uncased': 'bert', 'roberta-large': 'roberta'}
    cls_func_dict = {'bert-large-uncased': compute_bert_cls_rep, 'roberta-large': compute_roberta_cls_rep}

    args = parse_args()
    task_names = args.task_names
    sample_rate = args.sample_rate
    sample_times = args.sample_times
    seeds = [int(seed) for seed in args.seeds]

    lr = '2e-05'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # measure_comparison exploration
    for model_name in args.model_names:
        cls_func = cls_func_dict[model_name]
        pm_save_dir = f'results/{model_name_dict[model_name]}/instability_eval/representation_measures/subsample'
        os.makedirs(pm_save_dir, exist_ok=True)
        for task_name in task_names:

            eval_data_path = os.path.join(paths.fix_eval_data_dir, task_name, f'{model_name}/test.pt')
            eval_dl = torch.load(eval_data_path, map_location=device)

            # compute predictions/logits/representations
            logger.info(f'task: {task_name}')

            rep_dict = {}
            for seed in seeds:
                model_path = os.path.join(
                    f'results/{model_name_dict[model_name]}/best_validation_ckpt/models', task_name, lr,
                    'original', str(seed), 'best_model/')
                model = load_model(model_path).to(device)
                model.eval()
                rep_dict[seed] = cls_func(model, eval_dl).detach().cpu().numpy()  # 24 (layers), n_samples, hs

            num_layers, num_examples, _ = rep_dict[seeds[0]].shape
            rep_dist_path = os.path.join(pm_save_dir, f'{task_name}_{lr}_{sample_rate}_{sample_times}.h5')
            rep_dist_dict = defaultdict(lambda: np.zeros((len(seeds), len(seeds), sample_times, num_layers)))

            for sample_time in range(sample_times):
                logger.info(f'Computing sample rate {sample_rate}')
                sample_size = int(num_examples * sample_rate)
                sample_ids = np.random.choice(num_examples, size=sample_size, replace=False)
                for seed_0_idx, seed_0 in enumerate(seeds):
                    rep_0 = torch.FloatTensor(rep_dict[seed_0][:, sample_ids]).to(device)
                    for seed_1_idx, seed_1 in enumerate(seeds[seed_0_idx + 1:]):
                        rep_1 = torch.FloatTensor(rep_dict[seed_1][:, sample_ids]).to(device)

                        logger.info(f'Computing seed pair: {seed_0}-{seed_1}')
                        svcca_dist = rm.svcca_dist(rep_0, rep_1).detach().cpu().numpy()
                        cka_dist = rm.unbiased_linear_cka_dist(rep_0, rep_1).detach().cpu().numpy()
                        op_dist = rm.op_dist(rep_0, rep_1).detach().cpu().numpy()
                        rep_dist_dict['svcca_dists'][seed_0, seed_1, sample_time] = svcca_dist
                        rep_dist_dict['cka_dists'][seed_0, seed_1, sample_time] = cka_dist
                        rep_dist_dict['op_dists'][seed_0, seed_1, sample_time] = op_dist

                    dd.io.save(rep_dist_path, rep_dist_dict)


if __name__ == '__main__':
    main()
