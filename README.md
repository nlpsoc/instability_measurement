# Measuring the Instability of Fine-Tuning

**Python file structure description**

- ```save_eval_data.py``` for downloading and splitting data.
- ```ft_save_model.py``` for fine-tuning models. 
- ```compute_predictions.py``` for computing prediction measures.
- ```compute_representation_measures.py``` for computing representation measures.
- ```compute_subsampling_dist.py``` for computing representation measures of different subsamples.
- ```analyze_measure.ipynb``` for analyses and all Figures in the paper. 

**Requirements** 

Our experiments are performed using Python 3.7. 
```
torch==1.10.1
Transformers==4.14.1
datasets==1.9.0
tensorboardX
pingouin
loguru
deepdish
```


**Step 1: Data pre-processing** 

Download and split validation data into new validation and test sets. 
```shell
python save_eval_data.py --model_names bert-large-uncased roberta-large --tasks rte mrpc cola sst2 
```

**Step 2: Train models**

In total, we need to perform fine-tuning for 480 times, 
producing 600 models (final models are saved for the analyses of successful/failed runs), 
which will use ~1TB of disk space. 
```shell
bash ./train_model.sh
```

**Step 3: Compute predictions**

Compute predictions of all models for further analyses.
```shell
bash ./compute_predictions.sh
```

**Step 4: Compute representation instability** 

Compute (sub-sampling) representation instability of all models for further analyses. 
```shell
bash ./compute_representation_instability.sh
python compute_subsampling_dist.py --task_names rte mrpc cola --model_names bert-large-uncased roberta-large\
 --seeds $(echo {0..19}) --mitigation_methods original --lrs 2e-05 --sample_rate 0.5
python compute_subsampling_dist.py --task_names rte mrpc cola --model_names bert-large-uncased roberta-large\
 --seeds $(echo {0..19}) --mitigation_methods original --lrs 2e-05 --sample_rate 0.1
```

**Step 5: Run analyses in Section 5 and Section 6 of the paper**

See ```analyze_measures.ipynb```

If you find this repository useful, 
please consider cite our paper
```
@article{du-nguyen-2023-measuring,
  author     = {Yupei Du and
                Dong Nguyen},
  title      = {Measuring the Instability of Fine-Tuning},
  journal    = {CoRR},
  volume     = {abs/23xx.xxxxx},
  year       = {2023},
  url        = {https://arxiv.org/abs/23xx.xxxxx},
  eprinttype = {arXiv},
  eprint     = {23xx.xxxxx},
}
```