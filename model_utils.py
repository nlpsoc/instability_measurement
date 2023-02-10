import os
import pandas as pd
import numpy as np
from tensorflow.python.summary.summary_iterator import summary_iterator
import torch
import torch.nn as nn
from torch.optim import Optimizer
from transformers import AutoModelForSequenceClassification, AdamW, get_scheduler
from loguru import logger


def reinit_bert(bert_model, layers_to_reinit):
    r"""See also https://github.com/asappresearch/revisit-bert-finetuning/blob/0aa4f4e117ee4422f7cb9355158203e01d6730db/run_glue.py#L748"""
    if 'pooler' in layers_to_reinit:
        bert_model.bert.pooler.dense.weight.data.normal_(mean=0.0, std=bert_model.config.initializer_range)
        bert_model.bert.pooler.dense.bias.data.zero_()
        for p in bert_model.bert.pooler.parameters():
            p.requires_grad = True
        layers_to_reinit.remove('pooler')

    if len(layers_to_reinit) > 0:
        for layer_no in layers_to_reinit:
            for module in bert_model.bert.encoder.layer[layer_no].modules():
                if isinstance(module, (nn.Linear, nn.Embedding)):
                    module.weight.data.normal_(mean=0.0, std=bert_model.config.initializer_range)
                elif isinstance(module, nn.LayerNorm):
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
    return bert_model


def reinit_roberta(roberta_model, layers_to_reinit):
    r"""See also https://github.com/asappresearch/revisit-bert-finetuning/blob/0aa4f4e117ee4422f7cb9355158203e01d6730db/run_glue.py#L748"""
    if 'pooler' in layers_to_reinit:
        roberta_model.classifier.dense.weight.data.normal_(mean=0.0, std=roberta_model.config.initializer_range)
        roberta_model.classifier.dense.bias.data.zero_()
        for p in roberta_model.classifier.parameters():
            p.requires_grad = True
        layers_to_reinit.remove('pooler')

    if len(layers_to_reinit) > 0:
        for layer_no in layers_to_reinit:
            for module in roberta_model.roberta.encoder.layer[layer_no].modules():
                if isinstance(module, (nn.Linear, nn.Embedding)):
                    module.weight.data.normal_(mean=0.0, std=roberta_model.config.initializer_range)
                elif isinstance(module, nn.LayerNorm):
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
    return roberta_model



def get_wd_pre_optimizer_grouped_parameters(weight_decay, learning_rate, model):
    no_decay = ["bias", "LayerNorm.weight"]
    no_prior = ['pooler', 'classifier']
    optimizer_grouped_parameters = [
        {
            "params": [],
            "weight_decay": weight_decay,
            "lr": learning_rate,
            "prior": True,
        },
        {
            "params": [],
            "weight_decay": weight_decay,
            "lr": learning_rate,
            "prior": False,
        },
        {
            "params": [],
            "weight_decay": 0.0,
            "lr": learning_rate,
            "prior": False,
        }
    ]

    for name, param in model.named_parameters():
        prior = True
        decay = True
        for kw_decay in no_decay:
            if kw_decay in name:
                decay = False
                break
        if not decay:
            optimizer_grouped_parameters[2]['params'].append(param)  # no decay
            continue
        for kw_prior in no_prior:
            if kw_prior in name:
                prior = False
                break
        if prior:
            optimizer_grouped_parameters[0]['params'].append(param)  # decay to prior
        else:
            optimizer_grouped_parameters[1]['params'].append(param)  # decay to zero

    return optimizer_grouped_parameters


def get_llrd_optimizer_grouped_parameters(weight_decay, learning_rate, model, llrd=0.95, model_name='bert'):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if "classifier" in n or "pooler" in n],
            "weight_decay": 0.0,
            "lr": learning_rate,
        },
    ]
    if model_name == 'bert':
        layers = [model.bert.embeddings] + list(model.bert.encoder.layer)
    elif model_name == 'roberta':
        layers = [model.roberta.embeddings] + list(model.roberta.encoder.layer)
    layers.reverse()
    for layer in layers:
        learning_rate *= llrd
        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
                "lr": learning_rate,
            },
            {
                "params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": learning_rate,
            },
        ]
    return optimizer_grouped_parameters


def get_optimizer_grouped_parameters(weight_decay, learning_rate, model):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,  # Weight decay group
            "lr": learning_rate,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,  # No weight decay group
            "lr": learning_rate,
        },
    ]
    return optimizer_grouped_parameters


class PriorWD(Optimizer):
    def __init__(self, optim):
        super(PriorWD, self).__init__(optim.param_groups, optim.defaults)

        # python dictionary does not copy by default
        self.param_groups = optim.param_groups
        self.optim = optim

        self.weight_decay_by_group = []
        self.prior_params = {}
        for i, group in enumerate(self.param_groups):
            self.weight_decay_by_group.append(group["weight_decay"])
            if group['prior']:
                for p in group["params"]:
                    self.prior_params[id(p)] = p.detach().clone()
            group["weight_decay"] = 0

    def step(self, closure=None):
        loss = self.optim.step(closure)
        for i, group in enumerate(self.param_groups):
            if group['prior']:
                for p in group["params"]:
                    p.data.add_(-group["lr"] * self.weight_decay_by_group[i], p.data - self.prior_params[id(p)])
            else:
                for p in group["params"]:
                    p.data.add_(-group["lr"] * self.weight_decay_by_group[i], p.data)
        return loss


def construct_adam_w_optimizer(model, max_train_steps, learning_rate=2e-5, weight_decay=0.01,
                               lr_scheduler_type='linear', num_warmup_steps=0.1, algorithm=None, model_name='bert'):
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    if algorithm == 'wd_pre':
        optimizer_grouped_parameters = get_wd_pre_optimizer_grouped_parameters(weight_decay, learning_rate, model)
        optimizer = PriorWD(AdamW(optimizer_grouped_parameters, lr=learning_rate))
    elif algorithm == 'llrd':
        optimizer_grouped_parameters = get_llrd_optimizer_grouped_parameters(
            weight_decay, learning_rate, model, model_name=model_name)
        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    else:
        optimizer_grouped_parameters = get_optimizer_grouped_parameters(weight_decay, learning_rate, model)
        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    # Scheduler
    num_warmup_steps = int(num_warmup_steps) if (
            num_warmup_steps >= 1 or num_warmup_steps == 0) else num_warmup_steps * max_train_steps
    lr_scheduler = get_scheduler(
        name=lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_train_steps,
    )
    return optimizer, lr_scheduler


def retrieve_bert_cls_representations(bert_model, dataloader):
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(bert_model.device)
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            token_type_ids = batch['token_type_ids']
            bert_outputs = bert_model.bert(
                input_ids=input_ids, attention_mask=attention_mask,
                token_type_ids=token_type_ids, output_hidden_states=True)
            cls_representations = [hs[:, 0] for hs in bert_outputs.hidden_states] + [bert_outputs.pooler_output]
            yield cls_representations  # num_layers * [bs, hidden_size (1024)]


def write_pt(data, save_path):
    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)
    torch.save(data, save_path)
    return save_path


def convert_tb_data(root_dir, sort_by=None):
    """Convert local TensorBoard data into Pandas DataFrame.

    Function takes the root directory path and recursively parses
    all events data.
    If the `sort_by` value is provided then it will use that column
    to sort values; typically `wall_time` or `step`.

    *Note* that the whole data is converted into a DataFrame.
    Depending on the data size this might take a while. If it takes
    too long then narrow it to some sub-directories.

    Paramters:
        root_dir: (str) path to root dir with tensorboard data.
        sort_by: (optional str) column name to sort by.

    Returns:
        pandas.DataFrame with [wall_time, name, step, value] columns.

    """
    def convert_tfevent(filepath):
        return pd.DataFrame([
            parse_tfevent(e) for e in summary_iterator(filepath) if len(e.summary.value)
        ])

    def parse_tfevent(tfevent):
        return dict(
            wall_time=tfevent.wall_time,
            name=tfevent.summary.value[0].tag,
            step=tfevent.step,
            value=float(tfevent.summary.value[0].simple_value),
        )

    columns_order = ['wall_time', 'name', 'step', 'value']

    out = []
    for (root, _, filenames) in os.walk(root_dir):
        for filename in filenames:
            if "events.out.tfevents" not in filename:
                continue
            file_full_path = os.path.join(root, filename)
            out.append(convert_tfevent(file_full_path))

    # Concatenate (and sort) all partial individual dataframes
    all_df = pd.concat(out)[columns_order]
    if sort_by is not None:
        all_df = all_df.sort_values(sort_by)

    return all_df.reset_index(drop=True)


def average_grad_norm(model, norm_type=2):
    r"""norm_type: sum(abs(x)**ord)**(1./ord)"""
    grad_norms = [torch.norm(p.grad.detach(), norm_type) for p in model.parameters() if p.grad is not None]
    return (sum(grad_norms) / len(grad_norms)).item()


def get_bert_grad_norm(bert_model, norm_type=2):
    grad_norm_dict = {}
    # get grad norms
    bert_layers = bert_model.bert.encoder.layer
    for layer_no, bert_layer in enumerate(bert_layers):
        attn = bert_layer.attention.self
        grad_norm_dict[f'layer_{layer_no}'] = {
            'query': average_grad_norm(attn.query, norm_type),
            'key': average_grad_norm(attn.key, norm_type),
            'value': average_grad_norm(attn.value, norm_type)
        }
    grad_norm_dict['pooler'] = average_grad_norm(bert_model.bert.pooler.dense, norm_type)
    grad_norm_dict['classifier'] = average_grad_norm(bert_model.classifier, norm_type)
    return grad_norm_dict


def load_model(model_path):
    return AutoModelForSequenceClassification.from_pretrained(model_path)


def sample_from_np_array(x, axis=0, p=0.5):
    # generate prob
    axis_size = x.shape[axis]
    sample_size = int(axis_size * p)
    if axis != 0:
        x = np.moveaxis(x, axis, 0)
    sample_ids = np.random.choice(axis_size, size=sample_size, replace=False)
    x = x[sample_ids]
    if axis != 0:
        x = np.moveaxis(x, 0, axis)
    return x, sample_ids


def compute_bert_cls_rep(model, eval_dl):
    logger.info('Computing BERT CLS representations')
    cls_reps = []
    with torch.no_grad():
        for bs_idx, batch in enumerate(eval_dl):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            token_type_ids = batch['token_type_ids']
            bert_outputs = model.bert(
                input_ids=input_ids, attention_mask=attention_mask,
                token_type_ids=token_type_ids, output_hidden_states=True)
            cls_hs_list = [hs[:, 0].detach() for hs_idx, hs in enumerate(bert_outputs.hidden_states)]
            # cls_reps.append(torch.stack(cls_hs_list + [bert_outputs.pooler_output.detach()]))
            cls_reps.append(torch.stack(cls_hs_list[1:]))
    cls_reps = torch.cat(cls_reps, dim=1)
    return cls_reps


def compute_roberta_cls_rep(model, eval_dl):
    logger.info('Computing RoBERTa CLS representations')
    cls_reps = []
    with torch.no_grad():
        for bs_idx, batch in enumerate(eval_dl):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            roberta_outputs = model.roberta(
                input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            cls_hs_list = [hs[:, 0].detach() for hs_idx, hs in enumerate(roberta_outputs[1])]
            cls_reps.append(torch.stack(cls_hs_list[1:]))
    cls_reps = torch.cat(cls_reps, dim=1)
    return cls_reps


def compute_pred(model, eval_dl, metric, return_label=False):
    logger.info('Computing model predictions')
    logit_list = []
    label_list = []
    with torch.no_grad():
        for bs_idx, batch in enumerate(eval_dl):
            logits = model(**batch).logits
            metric.add_batch(predictions=logits.argmax(dim=-1), references=batch["labels"])
            logit_list.append(logits)
            label_list.append(batch['labels'])

    eval_metric = metric.compute()
    if return_label:
        return eval_metric, torch.cat(logit_list, dim=0), torch.cat(label_list, dim=0)
    else:
        return eval_metric, torch.cat(logit_list, dim=0)
