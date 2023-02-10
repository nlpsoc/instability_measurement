import os


data_dir = 'data/'
hg_data_dir = os.path.join(data_dir, 'hg_datasets/')
hg_config_dir = os.path.join(data_dir, 'hg_cfgs/')
hg_tkn_dir = os.path.join(data_dir, 'hg_tkns/')
hg_ptms_dir = os.path.join(data_dir, 'hg_ptms/')
hg_preprocess_dir = os.path.join(hg_data_dir, 'preprocessed/')

ckpt_dir = os.path.join(data_dir, 'ckpt/')

fix_eval_data_dir = os.path.join(data_dir, 'fixed_eval_data/')

log_dir = 'log/'

result_dir = 'results/'
pred_dir = os.path.join(result_dir, 'pred/')
tensor_board_dir = os.path.join(result_dir, 'tensor_board/')
hidden_state_dir = os.path.join(result_dir, 'hidden_state/')
