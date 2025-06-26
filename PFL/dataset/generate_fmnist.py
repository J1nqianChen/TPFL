import numpy as np
import logging
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

from utils.dataset_utils import check, separate_data, split_data, save_file


random.seed(1)
np.random.seed(1)
num_classes = 10


# Allocate data to users
def generate_fmnist(num_clients, num_classes, niid, balance, partition, dir_param):
    dir_path = f"FMNIST_IID_Client{num_clients}/"
    if partition == 'pat':
        dir_path = f"FMNIST_NonIID_Pat2_Client{num_clients}/"  # default pat=2 client 20
    elif partition == 'dir':
        dir_path = f"FMNIST_NonIID_Dir{dir_param}_Client{num_clients}/"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if check(config_path, train_path, test_path, num_clients, num_classes, niid, balance, partition, dir_param=dir_param):
        return


    # pre-compute normalize params
    pre_transform = transforms.ToTensor()
    pre_trainset = torchvision.datasets.FashionMNIST(
        root=dir_path+"rawdata", train=True, download=True, transform=pre_transform)
    pre_trainloader = torch.utils.data.DataLoader(
        pre_trainset, batch_size=1024, shuffle=False)

    # 动态初始化统计量（根据第一个batch确定通道数）
    total_sum = None
    total_sq_sum = None
    n_pixels = 0
    for data, _ in tqdm(pre_trainloader):
        # 获取通道数（单通道为1，RGB为3）
        # print(data.shape)
        channels = data.shape[1]
        n_pixels += data.shape[0] * data.shape[2] * data.shape[3]
        # 如果是第一次迭代，初始化统计量
        if total_sum is None:
            total_sum = torch.zeros(channels)
            total_sq_sum = torch.zeros(channels)

        # 展平数据：[batch, C, H, W] → [C, batch*H*W]
        data_flat = data.permute(1, 0, 2, 3).flatten(1)
        # print(data_flat.shape)
        # 累加总和和平方和
        total_sum += data_flat.sum(dim=1)
        total_sq_sum += (data_flat ** 2).sum(dim=1)


    # 计算均值和标准差
    mean = total_sum / n_pixels
    std = (total_sq_sum / n_pixels - mean ** 2).sqrt()

    print(f'Mean: {mean.tolist()}')
    print(f'Std:  {std.tolist()}')

    # Get FashionMNIST data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    trainset = torchvision.datasets.FashionMNIST(
        root=dir_path+"rawdata", train=True, download=True, transform=transform)
    testset = torchvision.datasets.FashionMNIST(
        root=dir_path+"rawdata", train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.data), shuffle=False)

    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data

    dataset_image = []
    dataset_label = []

    dataset_image.extend(trainset.data.cpu().detach().numpy())
    dataset_image.extend(testset.data.cpu().detach().numpy())
    dataset_label.extend(trainset.targets.cpu().detach().numpy())
    dataset_label.extend(testset.targets.cpu().detach().numpy())
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    # dataset = []
    # for i in range(num_classes):
    #     idx = dataset_label == i
    #     dataset.append(dataset_image[idx])

    X, y, statistic, batch_size = separate_data((dataset_image, dataset_label), num_clients, num_classes,
                                    niid, balance, partition, dir_param = dir_param)
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
        statistic, niid, balance, partition, dir_param=dir_param, batch_size=batch_size)


if __name__ == "__main__":
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None
    alpha = float(sys.argv[5])
    logging.info(alpha)
    num_clients = int(sys.argv[4])
    logging.info(num_clients)
    generate_fmnist(num_clients, num_classes, niid, balance, partition, dir_param=alpha)