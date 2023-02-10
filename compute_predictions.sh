#!/bin/bash

python compute_predictions.py --task_names rte mrpc cola --model_names bert-large-uncased roberta-large\
 --seeds $(echo {0..19}) --mitigation_methods original --lrs 2e-05

# BERT failed runs
python compute_predictions.py --task_names rte mrpc cola --model_names bert-large-uncased\
 --seeds $(echo {0..19}) --mitigation_methods original --lrs 5e-05

# RoBERTa failed runs
python compute_predictions.py --task_names rte mrpc cola --model_names roberta-large\
 --seeds $(echo {0..19}) --mitigation_methods original --lrs 3e-05

# Apply IMMs
python compute_predictions.py --task_names rte mrpc cola --model_names bert-large-uncased roberta-large\
 --seeds $(echo {0..9}) --mitigation_methods wd_pre mixout reinit llrd --lrs 2e-05
