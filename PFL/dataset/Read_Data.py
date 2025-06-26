import os
import sys

sys.path.append(os.path.split(os.path.split(os.path.abspath(os.path.dirname(__file__)))[0])[0])
from torchvision.transforms import transforms

from PFL.system.utils.data_utils import read_data
import numpy as np
import logging
import torch
from torch.utils.data import DataLoader, Dataset


def read_total(dataset, num_client=20):
    imgs_train = []
    labels_train = []
    for i in range(num_client):
        list = read_data(dataset, i, True)
        imgs_train.extend(list['x'])
        labels_train.extend(list['y'])
    X_train = torch.Tensor(np.array(imgs_train)).type(torch.float32)
    y_train = torch.Tensor(np.array(labels_train)).type(torch.int64)
    train_data = [(x, y) for x, y in zip(X_train, y_train)]

    imgs_test = []
    labels_test = []
    for i in range(num_client):
        list = read_data(dataset, i, False)
        # logging.info(i)
        # logging.info(f'num_sample: {len(list["x"])}')
        imgs_test.extend(list['x'])
        labels_test.extend(list['y'])
    X_test = torch.Tensor(np.array(imgs_test)).type(torch.float32)
    y_test = torch.Tensor(np.array(labels_test)).type(torch.int64)
    test_data = [(x, y) for x, y in zip(X_test, y_test)]

    return train_data, test_data


def select_read(dataset, select_list, ignore_val=True):
    imgs_train = []
    labels_train = []
    for i in select_list:
        list = read_data(dataset, i, True)
        imgs_train.extend(list['x'])
        labels_train.extend(list['y'])
    X_train = torch.Tensor(np.array(imgs_train)).type(torch.float32)
    y_train = torch.Tensor(np.array(labels_train)).type(torch.int64)
    train_data = [(x, y) for x, y in zip(X_train, y_train)]

    imgs_test = []
    labels_test = []
    sample_num = 0
    for i in select_list:
        list = read_data(dataset, i, False)
        if ignore_val:
            list_1_x = list[0]['x']
            list_2_x = list[1]['x']
            list_3_x = list[2]['x']
            list_1_y = list[0]['y']
            list_2_y = list[1]['y']
            list_3_y = list[2]['y']

            total_x = np.concatenate([list_1_x, list_2_x, list_3_x], axis=0)
            total_y = np.concatenate([list_1_y, list_2_y, list_3_y], axis=0)
        else:
            raise NotImplementedError
        logging.info(i)
        sample_num += total_x.shape[0]
        logging.info(f'num_sample: {sample_num}')
        imgs_test.extend(total_x)
        labels_test.extend(total_y)
    X_test = torch.Tensor(np.array(imgs_test)).type(torch.float32)
    y_test = torch.Tensor(np.array(labels_test)).type(torch.int64)
    test_data = [(x, y) for x, y in zip(X_test, y_test)]

    return train_data, test_data, sample_num


class CifarC(Dataset):
    def __init__(self, numpy_data, label_data, transform=None):
        self.data = torch.from_numpy(numpy_data).to(torch.float32)
        self.labels = torch.from_numpy(label_data)

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            sample = self.transform(sample)

        return sample, label


def read_cifar_c(dataset, client_id, keyword=None, severity=1):
    dataset_path = os.path.abspath(os.path.dirname(__file__))
    dataset_path = os.path.join(dataset_path, dataset)
    curr_dir_list = os.listdir(dataset_path)
    # for client_id in select_list:
    client_sample_list = []
    client_label_list = []
    for item in curr_dir_list:
        if int(item.split('_')[0]) == client_id:
            curr_path = os.path.join(dataset_path, item)
            item_list = np.load(curr_path, allow_pickle=True)['data'].tolist()
            x = item_list['x']
            y = item_list['y']
            per_severity_num = int(x.shape[0] / 5)
            begin_idx = (severity - 1) * per_severity_num
            end_idx = severity * per_severity_num

            corr_x = x[begin_idx:end_idx, :, :, :].transpose(0, 3, 1, 2)
            corr_y = y[begin_idx:end_idx]
            client_sample_list.append(corr_x)
            client_label_list.append(corr_y)
        else:
            continue

    client_sample_arr = np.concatenate(client_sample_list, axis=0)
    client_label_arr = np.concatenate(client_label_list, axis=0)

    transform = transforms.Compose(
        [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset_corruption = CifarC(client_sample_arr, client_label_arr, transform=transform)
    return dataset_corruption, per_severity_num


if __name__ == '__main__':
    # read_total('Cifar10')
    read_cifar_c('Cifar10_C', client_id=0)
