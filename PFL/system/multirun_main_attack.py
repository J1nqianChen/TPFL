import logging
import os
import signal
import time
from typing import Optional

import torch
from omegaconf import OmegaConf, DictConfig, ListConfig

from main import load_config, mk_exp_dirs, report_details_param, run, clean, send_serverchan
from utils.result_utils import setup_logger

signal_save_path: Optional[str] = None
signal_fault_path: Optional[str] = None


def handle_signal_multi(signum, frame):
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


def process_algo_hyperparams(cfg):
    """
    处理 OmegaConf 配置中的 algo_hyperparam 字段，返回每个算法的搜索参数长度及其完整路径

    返回格式:
        {
            "算法名": {
                "length": 搜索参数长度 (若无搜索则为 1),
                "path": search 字段的完整路径 (如 "algo_hyperparam.FedProx.mu.search", 若无则为 None)
            }
        }
    """

    def _collect_searches(node, current_path):
        """递归收集当前节点下所有 search 字段的完整路径和值"""
        searches = []
        if OmegaConf.is_config(node):
            if isinstance(node, DictConfig):
                # 检查当前节点是否有 search
                if "search" in node:
                    search_path = f"{current_path}.search" if current_path else "search"
                    searches.append((search_path, node.search))
                # 递归子节点
                for key in node:
                    child_path = f"{current_path}.{key}" if current_path else key
                    searches.extend(_collect_searches(node[key], child_path))
            elif isinstance(node, ListConfig):
                # 处理列表中的字典项
                for idx, item in enumerate(node):
                    child_path = f"{current_path}[{idx}]"
                    searches.extend(_collect_searches(item, child_path))
        return searches

    result = {}

    for algo_name in cfg.algo_hyperparam:
        algo_node = cfg.algo_hyperparam[algo_name]
        # 从 algo_hyperparam 下一级开始构建路径，例如 "algo_hyperparam.FedProx"
        initial_path = f"algo_hyperparam.{algo_name}"
        searches = _collect_searches(algo_node, initial_path)

        # 处理结果
        if len(searches) > 1:
            raise NotImplementedError(f"算法 {algo_name} 下发现多个 search 字段")
        elif len(searches) == 1:
            search_path, search_value = searches[0]
            if isinstance(search_value, (list, ListConfig)):
                result[algo_name] = {
                    "length": len(search_value),
                    "path": search_path
                }
            else:
                raise RuntimeError(f"路径 {search_path} 的 search 字段不是列表类型")
        else:
            result[algo_name] = {
                "length": 1,
                "path": None
            }

    return result


if __name__ == '__main__':
    # set_start_method('spawn', force=True)
    total_start = time.time()
    config = load_config()
    for att_ratio in config.attack.ratio_list:
        config.attack.ratio = att_ratio
        print(f'攻击：{config.attack.name}, Ratio:{config.attack.ratio}, 防御:{config.defense.name}')
        # Set
        logging.info(config)
        save_path, fault_path = mk_exp_dirs(config)
        signal_save_path = save_path
        signal_fault_path = fault_path
        signal.signal(signal.SIGTERM, handle_signal_multi)
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
        except Exception as e:
            logging.error(f"发生错误: {e}", exc_info=True)
            clean(save_path, fault_path)
            send_serverchan('发生错误', f'内容为：{e} DeviceGPU ID:{config.general.device_id}')
            exit(999)
        send_serverchan(f'攻击：{config.attack.name}, Ratio:{config.attack.ratio}, 防御:{config.defense.name}已执行完毕', f'DeviceGPU ID:{config.general.device_id}')

    send_serverchan('已执行完毕', f'{config.train.dataset_name} DeviceGPU ID:{config.general.device_id}')
