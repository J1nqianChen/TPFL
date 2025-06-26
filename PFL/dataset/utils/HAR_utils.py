# https://github.com/jindongwang/Deep-learning-activity-recognition/blob/master/pytorch/data_preprocess.py
import numpy as np
import logging
import os
import ujson
import numpy as np
import logging
import gc
from sklearn.model_selection import train_test_split

train_size = 0.75


# This is for parsing the X data, you can ignore it if you do not need preprocessing
def format_data_x(datafile):
    x_data = None
    for item in datafile:
        item_data = np.loadtxt(item, dtype=np.float32)
        if x_data is None:
            x_data = np.zeros((len(item_data), 1))
        x_data = np.hstack((x_data, item_data))
    x_data = x_data[:, 1:]
    # logging.info(x_data.shape)
    X = None
    for i in range(len(x_data)):
        row = np.asarray(x_data[i, :])
        row = row.reshape(9, 128).T
        if X is None:
            X = np.zeros((len(x_data), 128, 9))
        X[i] = row
    # logging.info(X.shape)
    return X


# This is for parsing the Y data, you can ignore it if you do not need preprocessing
def format_data_y(datafile):
    return np.loadtxt(datafile, dtype=np.int32) - 1


def read_ids(datafile):
    return np.loadtxt(datafile, dtype=np.int32)


# =======================================================================================
def split_data(X, y):
    # Split dataset
    train_data, test_data = [], []
    num_samples = {'train': [], 'test': []}

    for i in range(len(y)):
        unique, count = np.unique(y[i], return_counts=True)
        if min(count) > 1:
            X_train, X_test, y_train, y_test = train_test_split(
                X[i], y[i], train_size=train_size, shuffle=True)
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X[i], y[i], train_size=train_size, shuffle=True)

        train_data.append({'x': X_train, 'y': y_train})
        num_samples['train'].append(len(y_train))
        test_data.append({'x': X_test, 'y': y_test})
        num_samples['test'].append(len(y_test))

    logging.info("Total number of samples:", sum(num_samples['train'] + num_samples['test']))
    logging.info("The number of train samples:", num_samples['train'])
    logging.info("The number of test samples:", num_samples['test'])
    del X, y
    # gc.collect()

    return train_data, test_data


def save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, statistic):
    config = {
        'num_clients': num_clients,
        'num_classes': num_classes,
        'Size of samples for labels in clients': statistic,
    }

    # gc.collect()
    logging.info("Saving to disk.\n")

    for idx, train_dict in enumerate(train_data):
        with open(train_path + str(idx) + '.npz', 'wb') as f:
            np.savez_compressed(f, data=train_dict)
    for idx, test_dict in enumerate(test_data):
        test_x = test_dict['x']
        test_y = test_dict['y']

        len_test_dict = test_x.shape[0]
        part_len = len_test_dict // 3
        part_1_x = test_x[:part_len]
        part_2_x = test_x[part_len:2*part_len]
        part_3_x = test_x[2*part_len:]

        part_1_y = test_y[:part_len]
        part_2_y = test_y[part_len:2 * part_len]
        part_3_y = test_y[2 * part_len:]

        part_1 = {'x': part_1_x, 'y': part_1_y}
        part_2 = {'x': part_2_x, 'y': part_2_y}
        part_3 = {'x': part_3_x, 'y': part_3_y}
        with open(test_path + str(idx) + '_1.npz', 'wb') as f:
            np.savez_compressed(f, data=part_1)
        with open(test_path + str(idx) + '_2.npz', 'wb') as f:
            np.savez_compressed(f, data=part_2)
        with open(test_path + str(idx) + '_3.npz', 'wb') as f:
            np.savez_compressed(f, data=part_3)
    with open(config_path, 'w') as f:
        ujson.dump(config, f)

    logging.info("Finish generating dataset.\n")