#!/usr/bin/env python
import logging
import os
import random
import sys
from datetime import datetime
from typing import Optional
import signal
from urllib.parse import urlencode

from flcore.servers.serverDTPFL import DTPFL

sys.path.append(os.path.split(os.path.abspath(os.path.dirname(__file__)))[0])
sys.path.append(os.path.split(os.path.split(os.path.abspath(os.path.dirname(__file__)))[0])[0])
import requests
from omegaconf import OmegaConf

from flcore.servers.serverMAP import FedMAP
from flcore.servers.serverNTD import FedNTD
from flcore.servers.serveras import FedAS
from flcore.servers.serverdbe import FedDBE
from flcore.servers.servergpfl import GPFL
from flcore.servers.serverpac import FedPAC
from flcore.servers.serverpavg import FedPAvg
from flcore.servers.serverpbase import FedPBase
from flcore.servers.serverscaffold import SCAFFOLD
from flcore.servers.servertpfl import TPFL
from trainmodel.resnet import resnet18_gn


import omegaconf

from torchvision.models import vit_b_16

BASE_DIR = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
sys.path.insert(0, BASE_DIR)
_current_file_handler: Optional[logging.FileHandler] = None

import argparse
import time
import warnings
import numpy as np
import logging

from flcore.servers.severONE import FedONE
from flcore.servers.serveravg import FedAvg
from flcore.servers.serverprox import FedProx
from flcore.servers.serverditto import Ditto
from flcore.servers.serverrep import FedRep
from flcore.servers.serverrod import FedROD
from flcore.servers.serverdyn import FedDyn
from flcore.servers.servermoon import MOON
from flcore.trainmodel.models import *

# from flcore.trainmodel.resnet import resnet18 as resnet
from utils.result_utils import average_data, setup_logger
from utils.mem_utils import MemReporter

warnings.simplefilter("ignore")
torch.manual_seed(0)

# hyper-params for Text tasks
vocab_size = 98635
max_len = 200
hidden_dim = 32
signal_save_path: Optional[str] = None
signal_fault_path: Optional[str] = None


def send_serverchan(title, content):
    logging.info('推送中')
    key = 'SCT273804Tqr6P7WPRZnggU0w5LECSFYv8'
    url = f'https://sctapi.ftqq.com/{key}.send'
    params = {
        "title": (title),
        "desp": (content)
    }
    requests.post(url, params=params, verify=False)

def handle_signal(signum, frame):
    """信号处理函数，在接收到信号时执行清理并退出"""
    global signal_save_path, signal_fault_path
    logging.info(f'signal_save_path: {signal_save_path}')
    logging.error(f"接收到信号 {signum}，正在清理...")
    send_serverchan('发生错误', f'实验标号为{signal_save_path}')
    if signal_save_path is not None and signal_fault_path is not None:
        clean(signal_save_path, signal_fault_path)
    else:
        logging.warning("清理路径未初始化，跳过清理。")
    exit(999)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mk_exp_dirs(config):
    dataset = config.train.dataset_name
    local_steps = config.train.local_epochs
    join_ratio = config.train.participation_ratio
    algorithm = config.train.algorithm
    attack_flag = config.attack.flag
    defense_flag = config.defense.flag
    defense_name = config.defense.name
    model_str = config.train.model_str
    attack_strategy = config.attack.name
    try:
        pr_mode = config.train.participation_mode
    except AttributeError:
        pr_mode = 'static'
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not attack_flag:
        exp_str = f'{date_str}_{model_str}_{pr_mode}_pr_{join_ratio}_ls_{local_steps}_att_{int(attack_flag)}'
    elif not defense_flag:
        exp_str = f'{date_str}_{model_str}_{pr_mode}_pr_{join_ratio}_ls_{local_steps}_{int(attack_flag)}_{attack_strategy}_nodef'
    else:
        exp_str = f'{date_str}_{model_str}_{pr_mode}_pr_{join_ratio}_ls_{local_steps}_{int(attack_flag)}_{attack_strategy}_{defense_name}'



    default_path = config.general.default_path

    # base_path = f'/root/shared-nvme/experiments/{algorithm}/{dataset}/{exp_str}'  # 并行云
    base_path = os.path.join(default_path, f'{algorithm}/{dataset}/{exp_str}')    # 组里服务器
    fault_path = f'experiments_{algorithm}_{dataset}_{exp_str}'

    if not os.path.exists(base_path):
        logging.info('初始化实验保存目录')
        os.makedirs(base_path)
    return base_path, fault_path


def report_details_param(args):
    logging.info(
        '+++++++++++++++++ CAUTION: THE FOLLOWING PARAMS MAY NOT BE USED IN THIS PROCEDURE ++++++++++++++++++++++')
    logging.info('THIS IS THE FULL LIST OF DEFAULT AND PRE-SET VALUES OF ALL PARAMETERS VALUES')
    for arg_name in vars(args):
        logging.info(f"{arg_name}: {getattr(args, arg_name)}")


def initialize_model(config):
    train_model = None
    model_str = config.train.model
    if model_str == "cnn":
        if "MNIST" in config.train.dataset_name or "fmnist" in config.train.dataset_name:
            train_model = FedAvgCNN(in_features=1, num_classes=config.train.num_classes, dim=1024).to(
                config.general.device)
        elif "Cifar" in config.train.dataset_name:
            train_model = FedAvgCNN(in_features=3, num_classes=config.train.num_classes, dim=1600).to(
                config.general.device)
        elif config.train.dataset_name[:13] == "Tiny-imagenet" or config.train.dataset_name[
                                                                  :8] == "Imagenet" or 'TinyImageNet' in config.train.dataset_name:
            train_model = FedAvgCNN(in_features=3, num_classes=config.train.num_classes, dim=10816).to(
                config.general.device)
        else:
            train_model = FedAvgCNN(in_features=3, num_classes=config.train.num_classes, dim=1600).to(
                config.general.device)

    # elif model_str == "resnet50":
    #     train_model = torchvision.models.resnet50(pretrained=False, num_classes=config.train.num_classes).to(
    #         config.general.device)

    elif model_str == "resnet18":
        train_model = torchvision.models.resnet18(pretrained=False, num_classes=config.train.num_classes).to(
            config.general.device)
        # train_model = torchvision.models.resnet18(pretrained=True).to(config.general.device)
        # feature_dim = list(train_model.fc.parameters())[0].shape[1]
        # train_model.fc = nn.Linear(feature_dim, config.train.num_classes).to(config.general.device)

        # train_model = resnet18(num_classes=config.train.num_classes, has_bn=True, bn_block_num=4).to(config.general.device)
    elif model_str == 'resnet18m':
        logging.info('正在使用Resnet18m，仅使用Cifar 32x32尺寸')
        train_model = resnet18_gn(num_classes=config.train.num_classes).to(config.general.device)

    elif model_str == 'drop_resnet':
        train_model = torchvision.models.resnet50(pretrained=False, num_classes=config.train.num_classes).to(
            config.general.device)
        num_ftrs = train_model.fc.in_features

        # Define a new fc layer with a dropout layer before it
        train_model.fc = nn.Sequential(
            nn.Dropout(0.2),  # Dropout with 50% probability
            nn.Linear(num_ftrs, config.train.num_classes)  # 1000 is the number of classes in the original ResNet50
        ).to(config.general.device)

    elif model_str == 'harcnn':
        if config.train.dataset_name == 'har':
            train_model = HARCNN(9, dim_hidden=1664, num_classes=config.train.num_classes,
                                 conv_kernel_size=(1, 9),
                                 pool_kernel_size=(1, 2)).to(config.general.device)
        elif config.train.dataset_name == 'pamap':
            train_model = HARCNN(9, dim_hidden=3712, num_classes=config.train.num_classes,
                                 conv_kernel_size=(1, 9),
                                 pool_kernel_size=(1, 2)).to(config.general.device)

    elif model_str == "dropcnn":
        if config.train.dataset_name == "mnist" or config.train.dataset_name == "fmnist":
            train_model = DropCNN(in_features=1, num_classes=config.train.num_classes, dim=1024).to(
                config.general.device)
        elif config.train.dataset_name == "Cifar10" or config.train.dataset_name == "Cifar100":
            train_model = DropCNN(in_features=3, num_classes=config.train.num_classes, dim=1600).to(
                config.general.device)
        elif config.train.dataset_name[:13] == "Tiny-imagenet" or config.train.dataset_name[:8] == "Imagenet":
            train_model = DropCNN(in_features=3, num_classes=config.train.num_classes, dim=10816).to(
                config.general.device)
        else:
            train_model = DropCNN(in_features=3, num_classes=config.train.num_classes, dim=1600).to(
                config.general.device)

    elif model_str == 'vit':
        if "Cifar10" in config.train.dataset_name:
            train_model = vit_b_16(image_size=32, weights=None).to(config.general.device)
            train_model.heads[-1] = torch.nn.Linear(train_model.heads[-1].in_features,
                                                    config.train.num_classes).to(config.general.device)
        else:
            raise NotImplementedError

    elif model_str == 'textcnn':
        if "AGNews" in config.train.dataset_name:
            train_model = TextCNN(hidden_dim=config.train.model_config.feature_dim, max_len=config.train.model_config.max_len, vocab_size=config.train.model_config.vocab_size,
                                 num_classes=config.train.num_classes).to(config.general.device)
    else:
        raise NotImplementedError

    return train_model


def initialize_server(config, i, model_str, init_model):
    # select algorithm
    if config.train.algorithm == "FedAvg":
        server = FedAvg(config, i, init_model)

    elif config.train.algorithm == "FedProx":
        server = FedProx(config, i, init_model)

    elif config.train.algorithm == "Ditto":
        server = Ditto(config, i, init_model)

    elif config.train.algorithm == "FedRep":
        split_model = rearrange_model(init_model)
        server = FedRep(config, i, split_model)

    elif config.train.algorithm == "FedRoD":
        split_model = rearrange_model(init_model)
        server = FedROD(config, i, split_model)

    elif config.train.algorithm == "FedDyn":
        server = FedDyn(config, i, init_model)

    elif config.train.algorithm == "SCAFFOLD":
        server = SCAFFOLD(config, i, init_model)

    elif config.train.algorithm == "MOON":
        split_model = rearrange_model(init_model)
        server = MOON(config, i, split_model)

    elif config.train.algorithm == "FedONE":
        server = FedONE(config, i, init_model)

    elif config.train.algorithm == "FedNTD":
        server = FedNTD(config, i, init_model)

    elif config.train.algorithm == "FedPAC":
        split_model = rearrange_model(init_model)
        server = FedPAC(config, i, split_model)

    # elif config.train.algorithm == 'FedPAvg':
    #     split_model = rearrange_model(init_model)
    #     server = FedPAvg(config, i, split_model)
    #
    # elif config.train.algorithm == 'FedPC':
    #     split_model = rearrange_model(init_model)
    #     init_model = split_model
    #     server = FedPC(config, i)
    #
    # elif config.train.algorithm == 'PBase':
    #     split_model = rearrange_model(init_model)
    #     init_model = split_model
    #     server = FedPBase(config, i)

    elif config.train.algorithm == 'TPFL':
        split_model = rearrange_model(init_model)
        server = TPFL(config, i, split_model)

    elif config.train.algorithm == 'DTPFL':
        split_model = rearrange_model(init_model)
        server = DTPFL(config, i, split_model)

    elif config.train.algorithm == 'FedMAP':
        model = rearrange_model(init_model)
        dual_model = DualEncoder(model.base, model.head, data_str=config.train.dataset_name, model_str=model_str)
        init_model = dual_model
        server = FedMAP(config, i, init_model)

    elif config.train.algorithm == "GPFL":
        split_model = rearrange_model(init_model)
        server = GPFL(config, i, split_model)

    elif config.train.algorithm == "DBE":
        split_model = rearrange_model(init_model)
        server = FedDBE(config, i, split_model)

    elif config.train.algorithm == "FedAS":
        split_model = rearrange_model(init_model)
        server = FedAS(config, i, split_model)

    else:
        raise NotImplementedError

    return server


def run(config):
    time_list = []
    reporter = MemReporter()
    model_str = config.train.model
    config.train.model_str = model_str
    seed_list = config.general.seed
    for i in range(config.general.prev, config.general.times):
        set_seed(seed_list[i])
        logging.info(f"\n============= Running time: {i}th =============")
        logging.info("Creating server and clients ...")
        start = time.time()

        # Generate config.train.model
        init_model = initialize_model(config)

        server = initialize_server(config, i, model_str, init_model)
        server.train()

        time_list.append(time.time() - start)

        del init_model, server
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    logging.info(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")

    # Global average
    average_data(save_path=config.general.save_path,
                 times=config.general.times)

    logging.info("All done!")

    reporter.report()
    logging.info('save config')
    config_save_path = f'{config.general.save_path}/exp_config.yaml'
    OmegaConf.save(config, config_save_path)

def load_config():
    base_parser = argparse.ArgumentParser(
        add_help=False,
        description="配置加载主程序"
    )
    base_parser.add_argument("--config", type=str,
                             help="指定配置文件路径（与命令行参数互斥）")
    args = base_parser.parse_args()
    cfg = OmegaConf.load(args.config)
    return cfg


def clean(save_path, fault_path):
    logging.info(f'try-catch: 正在清理{save_path}')
    error_dir = 'fault'
    if not os.path.exists(error_dir):
        os.makedirs(error_dir)
    log_path = os.path.join(error_dir, f'{fault_path}.log')
    os.system(f"mv {save_path}/exp.log {log_path}")
    os.system(f"rm -rf {save_path}")


if __name__ == "__main__":
    # set_start_method('spawn', force=True)
    total_start = time.time()
    config = load_config()
    save_path, fault_path = mk_exp_dirs(config)
    signal_save_path = save_path
    signal_fault_path = fault_path
    signal.signal(signal.SIGTERM, handle_signal)
    config.general.save_path = save_path
    setup_logger(config.general.save_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = config.general.device_id
    if config.general.device == "cuda" and not torch.cuda.is_available():
        logging.info("\ncuda is not avaiable.\n")
        config.general.device = "cpu"
    logging.info("=" * 50)
    report_details_param(config)
    logging.info("=" * 50)
    try:
        run(config)
        send_serverchan('已执行完毕', f'{config.train.dataset_name} Algo:{config.train.algorithm}')
    except Exception as e:
        logging.error(f"发生错误: {e}", exc_info=True)
        clean(save_path, fault_path)
        send_serverchan('发生错误', f'错误为{e}')
        exit(999)

