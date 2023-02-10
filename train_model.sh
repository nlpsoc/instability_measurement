#!/bin/bash

# train models without IMMs (120 configurations => 120 models)
for model in bert-large-uncased roberta-large
do
  for task in rte mrpc cola
  do
    for rnds in {0..19}
    do
      echo "Fine-tuning $model on $task using random seed $rnds"
      python ft_save_model.py --model_name_or_path $model --task_name $task --seed $rnds
    done
  done
done

# produce failed runs (120 configurations => 240 models)
for model in bert-large-uncased roberta-large
do
  if [[ "$model" == "bert-large-uncased" ]]
  then
    lr=5e-05
  else
    lr=3e-05
  fi
  for task in rte mrpc cola
  do
    for rnds in {0..19}
    do
      echo "Producing failed runs for $model on $task using random seed $rnds and learning rate $lr"
      python ft_save_model.py --model_name_or_path $model --task_name $task --seed $rnds --learning_rate $lr --save_final_model
    done
  done
done

# train models with IMMs (240 configurations => 240 models)
for task in rte mrpc cola
do
  for imm in wd_pre mixout reinit llrd
  do
    for model in bert-large-uncased roberta-large
    do
      for rnds in {0..9}
      do
        echo "Fine-tuning $model on $task with $imm using random seed $rnds"
        python ft_save_model.py --model_name_or_path $model --task_name $task --seed $rnds --mitigation $imm
      done
    done
  done
done
