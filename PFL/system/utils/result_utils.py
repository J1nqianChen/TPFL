import logging
import sys
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import logging
import os


def average_data(save_path, times):
    test_acc = get_all_results_for_one_algo(save_path, times)

    max_accurancy = []
    for i in range(times):
        max_accurancy.append(test_acc[i].max())

    logging.info("std for best accurancy:" + str(np.std(max_accurancy)))
    logging.info("mean for best accurancy:" + str(np.mean(max_accurancy)))


def get_all_results_for_one_algo(save_path, times):
    test_acc = []
    for i in range(times):
        file_name = os.path.join(save_path, f'Time_{i}', 'record', 'record.h5')
        test_acc.append(np.array(
            read_data_then_delete(file_name, delete=False)))

    return test_acc


def read_data_then_delete(file_path, delete=False):
    with h5py.File(file_path, 'r') as hf:
        rs_test_acc = np.array(hf.get('rs_test_acc'))

    if delete:
        os.remove(file_path)
    logging.info(f"Length:  {len(rs_test_acc)}")

    return rs_test_acc


def setup_logger(log_path: str = "experiments/logs"):
    """配置同时输出到控制台和文件的日志系统"""
    Path(log_path).mkdir(parents=True, exist_ok=True)
    log_file = Path(log_path) / "exp.log"

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 清除已有处理器（避免重复日志）
    if logger.hasHandlers():
        logger.handlers.clear()

    # 文件处理器（带每日轮转）
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter(
        '%(message)s'
    ))

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(message)s'))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # 重定向未捕获的异常
    def handle_exception(exc_type, exc_value, exc_traceback):
        logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = handle_exception

    return logger