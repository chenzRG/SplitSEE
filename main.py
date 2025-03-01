import torch

import os
import numpy as np
from datetime import datetime
import argparse
from utils import _logger, set_requires_grad
from dataloader.dataloader import data_generator
from trainer.trainer import Trainer, model_evaluate
from models.TC import TC
from models.FC import FC
from utils import _calc_metrics, copy_Files
from models.model import TCN, classification_model, FFT_CNN
import torch.nn as nn
# Args selections
start_time = datetime.now()


parser = argparse.ArgumentParser()

######################## Model parameters ########################
home_dir = os.getcwd()
parser.add_argument('--experiment_description', default='default', type=str,
                    help='Experiment Description')
parser.add_argument('--run_description', default='default', type=str,
                    help='Run Description')
parser.add_argument('--seed', default=0, type=int,
                    help='seed value')
parser.add_argument('--mode', default='self_supervised', type=str,
                    help='Modes of choice: random_init, supervised, self_supervised, fine_tune, train_linear')
parser.add_argument('--dataset', default='CHBMIT', type=str,
                    help='Dataset of choice: sleepEDF, CHBMIT, HUH, SHHS')
parser.add_argument('--logs_save_dir', default='experiments_logs', type=str,
                    help='saving directory')
parser.add_argument('--device', default='cuda', type=str,
                    help='cpu or cuda')
parser.add_argument('--home_path', default=home_dir, type=str,
                    help='Project home directory')
parser.add_argument('--epoch', default='default', type=str,
                    help='Number of epoch')
parser.add_argument('--channel', default=0, type=str,
                    help='CHBMIT  or HUH channel of choice')
args = parser.parse_args()


device = torch.device(args.device)
experiment_description = args.experiment_description
data_type = args.dataset
method = 'SplitSEE'
mode = args.mode
run_description = args.run_description
channel = args.channel


logs_save_dir = args.logs_save_dir
os.makedirs(logs_save_dir, exist_ok=True)


exec(f'from config_files.{data_type}_Configs import Config as Configs')
configs = Configs()
if args.experiment_description == 'default':
    experiment_description = args.dataset
if args.run_description == 'default':
    run_description = 'run'
if args.epoch != 'default':
    configs.num_epoch = int(args.epoch)


# ##### fix random seeds for reproducibility ########
SEED = args.seed
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
#####################################################

experiment_log_dir = os.path.join(logs_save_dir, experiment_description, run_description, mode + f"_seed_{SEED}")
os.makedirs(experiment_log_dir, exist_ok=True)

# loop through domains
counter = 0
src_counter = 0

# Logging
log_file_name = os.path.join(experiment_log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
logger = _logger(log_file_name)
logger.debug("=" * 45)
logger.debug(f'Dataset: {data_type}')
logger.debug(f'Method:  {method}')
logger.debug(f'Mode:    {mode}')
if (data_type == "CHBMIT") or (data_type == "HUH"):
    logger.debug(f'Channel: {channel}')
logger.debug("=" * 45)

# Load datasets
if (data_type != "CHBMIT") and (data_type != "HUH"):
    data_path = f"data/{data_type}"
else:
    data_path = f"data/{data_type}/{channel}"
train_dl, valid_dl, test_dl = data_generator(data_path, configs, mode)
logger.debug("Data loaded ...")

# Load Model
temporal_model = TCN(configs).to(device)
frequency_model = FFT_CNN(configs).to(device)
temporal_contr_model = TC(configs, device).to(device)
frequency_contr_model = FC(configs, device).to(device)
c_model = classification_model(configs).to(device)

if mode == "fine_tune":
    # load saved model of this experiment
    load_from = os.path.join(os.path.join(logs_save_dir, experiment_description, run_description, f"self_supervised_seed_{SEED}", "saved_models"))
    chkpoint = torch.load(os.path.join(load_from, "ckp_last.pt"), map_location=device)
    pretrained_dict = chkpoint["model_state_dict"]
    model_dict = temporal_model.state_dict()
    model_dict.update(pretrained_dict)
    temporal_model.load_state_dict(model_dict)
    pretrained_dict = chkpoint["frequency_model_state_dict"]
    frequency_model_dict = frequency_model.state_dict()
    frequency_model_dict.update(pretrained_dict)
    frequency_model.load_state_dict(frequency_model_dict)
    pretrained_dict = chkpoint["temporal_contr_model_state_dict"]
    temporal_contr_model_dict = temporal_contr_model.state_dict()
    temporal_contr_model_dict.update(pretrained_dict)
    temporal_contr_model.load_state_dict(temporal_contr_model_dict)
    pretrained_dict = chkpoint["frequency_contr_model_state_dict"]
    frequency_contr_model_dict = frequency_contr_model.state_dict()
    frequency_contr_model_dict.update(pretrained_dict)
    frequency_contr_model.load_state_dict(frequency_contr_model_dict)

if mode == "test":
    # load saved model of this experiment
    load_from = os.path.join(os.path.join(logs_save_dir, experiment_description, run_description, f"fine_tune_seed_{SEED}", "saved_models"))
    chkpoint = torch.load(os.path.join(load_from, "ckp_last.pt"), map_location=device)
    pretrained_dict = chkpoint["model_state_dict"]
    model_dict = temporal_model.state_dict()
    model_dict.update(pretrained_dict)
    temporal_model.load_state_dict(model_dict)
    pretrained_dict = chkpoint["frequency_model_state_dict"]
    frequency_model_dict = frequency_model.state_dict()
    frequency_model_dict.update(pretrained_dict)
    frequency_model.load_state_dict(frequency_model_dict)
    pretrained_dict = chkpoint["temporal_contr_model_state_dict"]
    temporal_contr_model_dict = temporal_contr_model.state_dict()
    temporal_contr_model_dict.update(pretrained_dict)
    temporal_contr_model.load_state_dict(temporal_contr_model_dict)
    pretrained_dict = chkpoint["frequency_contr_model_state_dict"]
    frequency_contr_model_dict = frequency_contr_model.state_dict()
    frequency_contr_model_dict.update(pretrained_dict)
    frequency_contr_model.load_state_dict(frequency_contr_model_dict)

    # delete all the parameters except for logits
    del_list = ['logits']
    pretrained_dict_copy = model_dict.copy()
    for i in pretrained_dict_copy.keys():
        for j in del_list:
            if j in i:
                del model_dict[i]
    set_requires_grad(temporal_model, model_dict, requires_grad=False)  # Freeze everything except last layer.

model_optimizer = torch.optim.Adam(temporal_model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)
frequency_model_optimizer = torch.optim.Adam(frequency_model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)
temporal_contr_optimizer = torch.optim.Adam(temporal_contr_model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)
frequency_contr_optimizer = torch.optim.Adam(frequency_contr_model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)
classification_optimizer = torch.optim.Adam(c_model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)

if mode == "self_supervised":  # to do it only once
    copy_Files(os.path.join(logs_save_dir, experiment_description, run_description), data_type)

if mode != "test":
    # Trainer
    Trainer(temporal_model, frequency_model, temporal_contr_model, frequency_contr_model, c_model, model_optimizer, frequency_model_optimizer, temporal_contr_optimizer, frequency_contr_optimizer, classification_optimizer, train_dl, valid_dl, test_dl, device, logger, configs, experiment_log_dir, mode)

if mode != "self_supervised":
    # Testing
    outs = model_evaluate(temporal_model, frequency_model, temporal_contr_model, frequency_contr_model, c_model, test_dl, configs, device, mode)
    total_loss, total_acc, pred_labels, true_labels = outs
    _calc_metrics(pred_labels, true_labels, experiment_log_dir, args.home_path)

logger.debug(f"Processing time is : {datetime.now()-start_time}")
