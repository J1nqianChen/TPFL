import random
import math
import ujson
import numpy as np
import logging
import os
import torch
from PIL import Image
from scipy.stats import entropy, norm
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt


# IMAGE_SIZE = 28
# IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
# NUM_CHANNELS = 1

# IMAGE_SIZE_CIFAR = 32
# NUM_CHANNELS_CIFAR = 3


class TwoCropTransform:
    """Create two crops of the same image"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class Balanced_Data(Dataset):
    def __init__(self, data, labels, filter_threshold=100, key=None, transform=None, num_classes=10):
        self.total_data = data
        self.total_labels = labels
        self.transform = transform
        self.num_classes = num_classes
        self.filter_threshold = filter_threshold

        if len(labels) < 100:
            self.data = self.total_data
            self.labels = self.total_labels
        else:

            self.list_mask, self.list_count, self.down_sample_threshold, self.up_sample_threshold = self.__get_mask__()
            if self.down_sample_threshold == self.up_sample_threshold:
                logging.info('Load All Data')
                self.data = self.total_data
                self.labels = self.total_labels
            else:
                if key:
                    if key == 'down_sample':
                        self.__random_down_sample__()
                    elif key == 'up_sample':
                        self.__random_up_sample__()
                    else:
                        raise NotImplementedError
                else:
                    self.data = self.total_data
                    self.labels = self.total_labels

    def __get_mask__(self):
        list_mask = []
        list_count = []
        for i in range(self.num_classes):
            mask_temp = self.total_labels == i
            list_mask.append(mask_temp)
            list_count.append(mask_temp.sum().item())

        logging.info(list_count)
        down_sample_threshold = min(np.array(list_count)[np.array(list_count) > self.filter_threshold])
        up_sample_threshold = max(list_count)
        return list_mask, list_count, down_sample_threshold, up_sample_threshold

    def __random_down_sample__(self):
        list_temp_data = []
        list_temp_label = []
        for i in range(self.num_classes):
            mask_temp = self.list_mask[i]
            count_temp = self.list_count[i]
            if count_temp > self.down_sample_threshold:
                true_index = np.where(mask_temp)[0]
                random_selected_index = np.random.choice(true_index, self.down_sample_threshold, replace=False)
                list_temp_data.append(self.total_data[random_selected_index])
                list_temp_label.append(self.total_labels[random_selected_index])
            else:
                list_temp_data.append(self.total_data[mask_temp])
                list_temp_label.append(self.total_labels[mask_temp])
        self.data = torch.cat(list_temp_data, dim=0)
        self.labels = torch.cat(list_temp_label, dim=0)

        list_mask_sample = []
        list_count_sample = []
        for i in range(self.num_classes):
            mask_temp = self.labels == i
            list_mask_sample.append(mask_temp)
            list_count_sample.append(mask_temp.sum().item())
        logging.info('Down Sample')
        logging.info(list_count_sample)

    def __random_up_sample__(self):
        list_temp_data = []
        list_temp_label = []
        for i in range(self.num_classes):
            mask_temp = self.list_mask[i]
            count_temp = self.list_count[i]
            if self.down_sample_threshold <= count_temp < self.up_sample_threshold:
                true_index = np.where(mask_temp)[0]
                random_selected_index = np.random.choice(true_index, self.up_sample_threshold, replace=True)
                list_temp_data.append(self.total_data[random_selected_index])
                list_temp_label.append(self.total_labels[random_selected_index])
            else:
                list_temp_data.append(self.total_data[mask_temp])
                list_temp_label.append(self.total_labels[mask_temp])
        self.data = torch.cat(list_temp_data, dim=0)
        self.labels = torch.cat(list_temp_label, dim=0)

        list_mask_sample = []
        list_count_sample = []
        for i in range(self.num_classes):
            mask_temp = self.labels == i
            list_mask_sample.append(mask_temp)
            list_count_sample.append(mask_temp.sum().item())
        logging.info('Up Sample')
        logging.info(list_count_sample)

    def __random_up_sample1__(self):
        list_temp_data = []
        list_temp_label = []
        for i in range(self.num_classes):
            mask_temp = self.list_mask[i]
            count_temp = self.list_count[i]
            if count_temp != 0 and count_temp < self.up_sample_threshold:
                true_index = np.where(mask_temp)[0]
                random_selected_index = np.random.choice(true_index, self.up_sample_threshold, replace=True)
                list_temp_data.append(self.total_data[random_selected_index])
                list_temp_label.append(self.total_labels[random_selected_index])
            else:
                list_temp_data.append(self.total_data[mask_temp])
                list_temp_label.append(self.total_labels[mask_temp])
        self.data = torch.cat(list_temp_data, dim=0)
        self.labels = torch.cat(list_temp_label, dim=0)

        list_mask_sample = []
        list_count_sample = []
        for i in range(self.num_classes):
            mask_temp = self.labels == i
            list_mask_sample.append(mask_temp)
            list_count_sample.append(mask_temp.sum().item())
        logging.info('Up Sample')
        logging.info(list_count_sample)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # sample = {'data': self.data[index], 'label': self.labels[index]}

        data_temp = self.data[index]
        label_temp = self.labels[index]
        if self.transform:
            data_temp = self.transform(data_temp)
        return data_temp, label_temp


def batch_data(data, batch_size):
    '''
    data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
    returns x, y, which are both numpy array of length: batch_size
    '''
    data_x = data['x']
    data_y = data['y']

    # randomly shuffle data
    ran_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(ran_state)
    np.random.shuffle(data_y)

    # loop through mini-batches
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i:i + batch_size]
        batched_y = data_y[i:i + batch_size]
        yield (batched_x, batched_y)


def get_random_batch_sample(data_x, data_y, batch_size):
    num_parts = len(data_x) // batch_size + 1
    if (len(data_x) > batch_size):
        batch_idx = np.random.choice(list(range(num_parts + 1)))
        sample_index = batch_idx * batch_size
        if (sample_index + batch_size > len(data_x)):
            return (data_x[sample_index:], data_y[sample_index:])
        else:
            return (data_x[sample_index: sample_index + batch_size], data_y[sample_index: sample_index + batch_size])
    else:
        return (data_x, data_y)


def get_batch_sample(data, batch_size):
    data_x = data['x']
    data_y = data['y']

    # np.random.seed(100)
    ran_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(ran_state)
    np.random.shuffle(data_y)

    batched_x = data_x[0:batch_size]
    batched_y = data_y[0:batch_size]
    return (batched_x, batched_y)


def read_data(dataset, idx, is_train=True):
    current_dir = os.path.abspath(os.path.dirname(__file__))
    parts = current_dir.split(os.sep)
    target_parts = parts[:-2]
    path_ = os.path.join('/', *target_parts, 'dataset')
    dataset_path = path_
    # dataset_path = '/mnt/edge_computing/Codefmeister/code/MINE_FL/PFL/dataset'

    if is_train:
        train_data_dir = os.path.join(dataset_path, dataset, 'train/')

        train_file = train_data_dir + str(idx) + '.npz'
        with open(train_file, 'rb') as f:
            train_data = np.load(f, allow_pickle=True)['data'].tolist()

        return train_data

    else:
        test_data_dir = os.path.join(dataset_path, dataset, 'test/')

        test_file = test_data_dir + str(idx) + '_1.npz'
        with open(test_file, 'rb') as f:
            test_data_1 = np.load(f, allow_pickle=True)['data'].tolist()

        test_file_2 = test_data_dir + str(idx) + '_2.npz'
        with open(test_file_2, 'rb') as f:
            test_data_2 = np.load(f, allow_pickle=True)['data'].tolist()

        test_file_3 = test_data_dir + str(idx) + '_3.npz'
        with open(test_file_3, 'rb') as f:
            test_data_3 = np.load(f, allow_pickle=True)['data'].tolist()

        return test_data_1, test_data_2, test_data_3


def read_client_data(dataset, idx, is_train=True, transform=None, times=None):
    if "AGNews" in dataset:
        return read_client_data_text(dataset, idx, is_train, times)

    if is_train:
        train_data = read_data(dataset, idx, is_train)
        X_train = torch.Tensor(train_data['x']).type(torch.float32)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        if transform:
            list_1 = []
            list_2 = []
            for i in range(X_train.shape[0]):
                img = X_train[i, :, :, :]
                transformed_img = transform(img)
                list_1.append(transformed_img[0])
                list_2.append(transformed_img[1])
            stack_img1 = torch.stack(list_1, dim=0)
            stack_img2 = torch.stack(list_2, dim=0)
            X_train = torch.stack([stack_img1, stack_img2], dim=1)


        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        return train_data
    else:
        part1, part2, part3 = read_data(dataset, idx, is_train)


        X_part_1 = torch.Tensor(part1['x']).type(torch.float32)
        y_part_1 = torch.Tensor(part1['y']).type(torch.int64)
        X_part_2 = torch.Tensor(part2['x']).type(torch.float32)
        y_part_2 = torch.Tensor(part2['y']).type(torch.int64)
        X_part_3 = torch.Tensor(part3['x']).type(torch.float32)
        y_part_3 = torch.Tensor(part3['y']).type(torch.int64)

        if times == 0:
            X_test_1 = X_part_1
            y_test_1 = y_part_1
            X_test_2 = X_part_2
            y_test_2 = y_part_2
            X_val = X_part_3
            y_val = y_part_3
        elif times == 1:
            X_test_1 = X_part_2
            y_test_1 = y_part_2
            X_test_2 = X_part_3
            y_test_2 = y_part_3
            X_val = X_part_1
            y_val = y_part_1
        elif times == 2:
            X_test_1 = X_part_1
            y_test_1 = y_part_1
            X_test_2 = X_part_3
            y_test_2 = y_part_3
            X_val = X_part_2
            y_val = y_part_2
        else:
            raise NotImplementedError('暂不支持超过三次推理，现在为3-Fold')


        X_test = torch.cat([X_test_1, X_test_2], dim=0)
        y_test = torch.cat([y_test_1, y_test_2], dim=0)
        if transform:
            X_test = torch.stack([transform(x) for x in X_test])
            X_val = torch.stack([transform(x) for x in X_val])

        test_data = [(x, y) for x, y in zip(X_test, y_test)]
        val_data = [(x, y) for x, y in zip(X_val, y_val)]
        return test_data, val_data


# def read_client_test_val_data(dataset, idx, seed, transform):
#     pass
#     # test_data = read_data(dataset, idx, False)
#     # X_test = torch.Tensor(test_data['x']).type(torch.float32)
#     # y_test = torch.Tensor(test_data['y']).type(torch.int64)
#     #
#     # shuffle_idx = torch.randperm(X_test.shape[0])
#     # X_shuffle = X_test[shuffle_idx]
#     # y_shuffle = y_test[shuffle_idx]
#     #
#     # # train:test:val = 7:2:1
#     # val_size
#     # X_val[]
#     #
#     # if transform:
#     #     X_test = torch.stack([transform(x) for x in X_test])
#     #
#     # test_data = [(x, y) for x, y in zip(X_test, y_test)]
#     # return test_data


def read_client_data_drop(dataset, idx, is_train=True, class_id=0, number=0, seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if dataset[:2] == "ag" or dataset[:2] == "SS":
        return read_client_data_text(dataset, idx)

    if is_train:
        train_data = read_data(dataset, idx, is_train)
        X_train = torch.Tensor(train_data['x']).type(torch.float32)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        y_train_mask = y_train != class_id
        X_train_drop_1 = X_train[y_train_mask]
        y_train_drop_1 = y_train[y_train_mask]

        y_train_mask = y_train == class_id
        X_train_drop_2 = X_train[y_train_mask]
        y_train_drop_2 = y_train[y_train_mask]

        select_idx = np.random.choice(np.arange(X_train_drop_2.shape[0]).tolist(), number, replace=False)
        x_train_drop_2_select = X_train_drop_2[select_idx, :]
        y_train_drop_2_select = y_train_drop_2[select_idx]

        x_cat = torch.cat([X_train_drop_1, x_train_drop_2_select], dim=0)
        y_cat = torch.cat([y_train_drop_1, y_train_drop_2_select], dim=0)
        train_data = [(x, y) for x, y in zip(x_cat, y_cat)]
        return train_data


    else:
        test_data = read_data(dataset, idx, is_train)
        X_test = torch.Tensor(test_data['x']).type(torch.float32)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)
        test_data = [(x, y) for x, y in zip(X_test, y_test)]
        return test_data


def read_client_data_text(dataset, idx, is_train=True, times=None):
    if is_train:
        train_data = read_data(dataset, idx, is_train)
        X_train, X_train_lens = list(zip(*train_data['x']))
        y_train = train_data['y']

        X_train = torch.Tensor(X_train).type(torch.int64)
        X_train_lens = torch.Tensor(X_train_lens).type(torch.int64)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        train_data = [((x, lens), y) for x, lens, y in zip(X_train, X_train_lens, y_train)]
        return train_data
    else:

        part1, part2, part3 = read_data(dataset, idx, is_train)

        X_test_part1, X_test_lens_part1 = list(zip(*part1['x']))
        y_test_part1 = part1['y']
        X_test_part2, X_test_lens_part2 = list(zip(*part2['x']))
        y_test_part2 = part2['y']
        X_test_part3, X_test_lens_part3 = list(zip(*part3['x']))
        y_test_part3 = part3['y']


        X_test_part1 = torch.Tensor(X_test_part1).type(torch.int64)
        X_test_lens_part1 = torch.Tensor(X_test_lens_part1).type(torch.int64)
        y_test_part1 = torch.Tensor(y_test_part1).type(torch.int64)

        X_test_part2 = torch.Tensor(X_test_part2).type(torch.int64)
        X_test_lens_part2 = torch.Tensor(X_test_lens_part2).type(torch.int64)
        y_test_part2 = torch.Tensor(y_test_part2).type(torch.int64)
        
        X_test_part3 = torch.Tensor(X_test_part3).type(torch.int64)
        X_test_lens_part3 = torch.Tensor(X_test_lens_part3).type(torch.int64)
        y_test_part3 = torch.Tensor(y_test_part3).type(torch.int64)


        if times == 0:
            X_test = torch.cat([X_test_part1, X_test_part2], dim=0)
            X_test_lens = torch.cat([X_test_lens_part1, X_test_lens_part2], dim=0)
            y_test = torch.cat([y_test_part1, y_test_part2], dim=0)
            X_val = X_test_part3
            X_val_lens = X_test_lens_part3
            y_val = y_test_part3
        elif times == 1:
            X_test = torch.cat([X_test_part3, X_test_part2], dim=0)
            X_test_lens = torch.cat([X_test_lens_part3, X_test_lens_part2], dim=0)
            y_test = torch.cat([y_test_part3, y_test_part2], dim=0)
            X_val = X_test_part1
            X_val_lens = X_test_lens_part1
            y_val = y_test_part1
        elif times == 2:
            X_test = torch.cat([X_test_part1, X_test_part3], dim=0)
            X_test_lens = torch.cat([X_test_lens_part1, X_test_lens_part3], dim=0)
            y_test = torch.cat([y_test_part1, y_test_part3], dim=0)
            X_val = X_test_part2
            X_val_lens = X_test_lens_part2
            y_val = y_test_part2


        test_data = [((x, lens), y) for x, lens, y in zip(X_test, X_test_lens, y_test)]
        val_data = [((x, lens), y) for x, lens, y in zip(X_val, X_val_lens, y_val)]
        return test_data, val_data


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


def select_test_read(dataset, select_list):
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
        logging.info(i)
        sample_num += len(list["x"])
        logging.info(f'num_sample: {sample_num}')
        imgs_test.extend(list['x'])
        labels_test.extend(list['y'])
    X_test = torch.Tensor(np.array(imgs_test)).type(torch.float32)
    y_test = torch.Tensor(np.array(labels_test)).type(torch.int64)
    test_data = [(x, y) for x, y in zip(X_test, y_test)]

    return train_data, test_data, sample_num


def cal_entropy(label_count):
    label_dis = label_count / np.sum(label_count, keepdims=True)
    entropy1 = entropy(label_dis)
    return entropy1


def transform_to_interval(observation, expectation, variance):
    if variance == 0:
        return 1
    centered_value = observation - expectation
    normalized_value = centered_value / (math.sqrt(variance))
    transformed_value = 1 / (1 + math.exp(-normalized_value))
    return transformed_value


def add_backdoor_pattern_tensor(data, label, insert_num, target_label=2):
    batch_size, C, H, W = data.shape
    assert insert_num <= batch_size

    cross_size = 5
    start_x = 32 - cross_size
    start_y = 32 - cross_size

    # Create a mask for the red cross pattern
    pattern_mask = torch.zeros((C, H, W), dtype=data.dtype)
    pattern_mask[:, start_y:start_y + cross_size, start_x:start_x + cross_size] = torch.tensor(
        [1, -1, -1]).view(C, 1, 1)

    # Creating the cross pattern in the mask
    for i in range(cross_size):
        pattern_mask[0, start_y + i, start_x:start_x + cross_size] = 255  # R
        pattern_mask[:, start_y:start_y + cross_size, start_x + i] = torch.tensor([255, 0, 0]).unsqueeze(
            -1)  # R for all, G&B stay 0

    # Randomly select images to insert the pattern
    insert_index_list = np.random.choice(range(batch_size), insert_num, replace=False)

    # Add the pattern to the selected images
    data[insert_index_list] += pattern_mask
    label[insert_index_list] = target_label

    # Ensure the values do not exceed the maximum for the datatype (e.g., 255 for uint8)
    data[insert_index_list] = torch.clamp(data[insert_index_list], -1, 1)

    return data, label


def visualize_tensor_images(tensor):
    """
    Visualizes each image in a tensor of shape (B, C, H, W) on CUDA,
    assuming the images are normalized to [-1, 1].
    """
    # Ensure tensor is on CPU and convert to numpy for visualization
    tensor = tensor.cpu()

    # Rescale the images from [-1, 1] to [0, 1] for visualization
    np_images = ((tensor.numpy() + 1) * 0.5)

    # Check if the tensor is in CHW format and convert to HWC for visualization
    if tensor.shape[1] == 3:  # Assuming 3 channels for RGB
        np_images = np.transpose(np_images, (0, 2, 3, 1))

    # Visualize each image
    batch_size = np_images.shape[0]
    for i in range(batch_size):
        plt.figure(figsize=(5, 5))
        plt.imshow(np.clip(np_images[i], 0, 1))  # Ensure the image is within [0, 1] after any rounding errors
        plt.title(f"Image {i + 1}")
        plt.axis('off')
        plt.show()


import hashlib
import os
import pickle
import random
from collections import OrderedDict

import numpy as np
import logging
import torch
from sklearn.utils import shuffle as shuffle_func


def mf(file_path):
    _dir = os.path.dirname(file_path)
    _dir and os.makedirs(_dir, exist_ok=True)
    return file_path


def md(dirname):
    dirname and os.makedirs(dirname, exist_ok=True)
    return dirname


def set_seed(seed, to_numpy=True, to_torch=True, to_torch_cudnn=True):
    if seed is None:
        return
    random.seed(seed)
    if to_numpy:
        np.random.seed(seed)
    if to_torch:
        torch.manual_seed(seed)
        if torch.cuda.is_available() and to_torch_cudnn:
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True


def consistent_hash(*objs, code_len=6):
    assert os.environ.get('PYTHONHASHSEED') == '0', "env variable : PYTHONHASHSEED==0 should be specified"
    return hashlib.md5(pickle.dumps(objs)).hexdigest()[:code_len]


def formula(func, *params_args):
    res = OrderedDict()
    for name in params_args[0].keys():
        weight = func(*[params[name] for params in params_args])
        res[name] = weight.detach().clone()
    return res


def model_size(model):
    if isinstance(model, torch.nn.Module):
        params_iter = model.named_parameters()
    elif isinstance(model, dict):
        params_iter = model.items()
    else:
        raise Exception(f"unknow type: {type(model)}, expected is torch.nn.Module or dict")
    res = 0.0
    for _, weight in params_iter:
        res += (weight.element_size() * weight.nelement())
    return res


def save_pkl(obj, file_path):
    with open(mf(file_path), "wb") as _f:
        pickle.dump(obj, _f)


def load_pkl(file_path):
    with open(file_path, "rb") as _f:
        return pickle.load(_f)


def batch_iter(*iters, batch_size=256, shuffle=False, random_state=None):
    _iter_num = len(iters)
    if shuffle:
        iters = shuffle_func(*iters, random_state=random_state)
        if _iter_num == 1:
            iters = (iters,)
    _current_size = 0
    _batch = [[] for _ in range(_iter_num)]
    for _item_tuples in zip(*iters):
        for i in range(_iter_num):
            _batch[i].append(_item_tuples[i])
        _current_size += 1
        if _current_size >= batch_size:
            yield _batch if _iter_num > 1 else _batch[0]
            _current_size = 0
            _batch = [[] for _ in range(_iter_num)]
    if _current_size:
        yield _batch if _iter_num > 1 else _batch[0]


def grad_False(model, select_frozen_layers=None):
    if select_frozen_layers == None:
        for name, param in model.named_parameters():
            param.requires_grad = False
    else:
        i = 0
        for name, param in model.named_parameters():
            if select_frozen_layers in model.named_parameter_layers[i]:
                param.requires_grad = False
            i += 1


def grad_True(model):
    for name, param in model.named_parameters():
        param.requires_grad = True


def find_z_max(n, m):
    """
    Find the maximum z such that phi(z) < (n - m - s) / (n - m),
    where s = ceil(n / 2) + 1 - m.

    Args:
        n (int): Value for n.
        m (int): Value for m.

    Returns:
        float: The maximum z.
    """
    # Compute s
    s = int(n / 2 + 1) - m

    # Calculate the target value
    target_value = (n - m - s) / (n - m)
    logging.info(f'target_value is {target_value}')

    def normal_cdf(z):
        return norm.cdf(z)

    # Define the function to find max z
    def objective(z):
        return normal_cdf(z) - target_value

    current_z = 0
    for z in np.arange(-20, 20, 0.1):
        if objective(z) > 0:
            current_z = z - 0.1
            break

    return current_z


def compute_mean_var_of_benign_updates(benign_models):
    """
    Compute the mean and variance of benign model updates.

    Args:
        benign_models (list of nn.Module): List of benign model updates.

    Returns:
        tuple: mean and variance of the model parameters.
    """
    # Extract parameters from each model and flatten them
    benign_updates = []
    for model in benign_models:
        params = flatten_params(model)
        benign_updates.append(params)

    # Stack all updates into a single tensor
    benign_updates_tensor = torch.stack(benign_updates)

    # Compute mean and variance
    mean_benign = torch.mean(benign_updates_tensor, dim=0)
    var_benign = torch.var(benign_updates_tensor, dim=0)

    return mean_benign, var_benign


def get_malicious_updates_fang_trmean(all_updates, deviation, n_attackers):
    b = 2
    max_vector = torch.max(all_updates, 0)[0]
    min_vector = torch.min(all_updates, 0)[0]

    max_ = (max_vector > 0).type(torch.FloatTensor).cuda()
    min_ = (min_vector < 0).type(torch.FloatTensor).cuda()

    max_[max_ == 1] = b
    max_[max_ == 0] = 1 / b
    min_[min_ == 1] = b
    min_[min_ == 0] = 1 / b

    max_range = torch.cat((max_vector[:, None], (max_vector * max_)[:, None]), dim=1)
    min_range = torch.cat(((min_vector * min_)[:, None], min_vector[:, None]), dim=1)

    rand = torch.from_numpy(np.random.uniform(0, 1, [len(deviation), n_attackers])).type(torch.FloatTensor).cuda()

    max_rand = torch.stack([max_range[:, 0]] * rand.shape[1]).T + rand * torch.stack(
        [max_range[:, 1] - max_range[:, 0]] * rand.shape[1]).T
    min_rand = torch.stack([min_range[:, 0]] * rand.shape[1]).T + rand * torch.stack(
        [min_range[:, 1] - min_range[:, 0]] * rand.shape[1]).T

    mal_vec = (torch.stack(
        [(deviation > 0).type(torch.FloatTensor)] * max_rand.shape[1]).T.cuda() * max_rand + torch.stack(
        [(deviation > 0).type(torch.FloatTensor)] * min_rand.shape[1]).T.cuda() * min_rand).T

    return mal_vec


def get_model_size_in_mb(model):
    """
    Calculate the size of a PyTorch model in megabytes (MB).

    Args:
        model (torch.nn.Module): The PyTorch model.

    Returns:
        float: The size of the model in megabytes (MB).
    """
    # Calculate the total size of the model's parameters in bytes
    model_size_in_bytes = sum(param.numel() * param.element_size() for param in model.parameters())

    # Convert bytes to megabytes
    model_size_in_mb = model_size_in_bytes / (1024 ** 2)  # 1 MB = 1024^2 bytes

    return model_size_in_mb


def our_attack_dist(all_updates, model_re, n_attackers, dev_type='std'):
    if dev_type == 'unit_vec':
        deviation = model_re / torch.norm(model_re)  # unit vector, dir opp to good dir
    elif dev_type == 'sign':
        deviation = torch.sign(model_re)
    elif dev_type == 'std':
        deviation = torch.std(all_updates, 0)

    lamda = torch.Tensor([10.0]).float().cuda()
    # logging.info(lamda)
    threshold_diff = 1e-5
    lamda_fail = lamda
    lamda_succ = 0

    distances = []
    for update in all_updates:
        distance = torch.norm((all_updates - update), dim=1) ** 2
        distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)

    max_distance = torch.max(distances)
    del distances

    while torch.abs(lamda_succ - lamda) > threshold_diff:
        mal_update = (model_re - lamda * deviation)
        distance = torch.norm((all_updates - mal_update), dim=1) ** 2
        max_d = torch.max(distance)

        if max_d <= max_distance:
            # logging.info('successful lamda is ', lamda)
            lamda_succ = lamda
            lamda = lamda + lamda_fail / 2
        else:
            lamda = lamda - lamda_fail / 2

        lamda_fail = lamda_fail / 2

    mal_update = (model_re - lamda_succ * deviation)

    return mal_update


def flatten_params(model):
    """
    Flatten the parameters of a model into a single tensor.

    Args:
        model (nn.Module): The model whose parameters will be flattened.

    Returns:
        torch.Tensor: A 1D tensor containing all the parameters.
    """
    # for param in model.parameters():
    #     if torch.isnan(param).any():
    #         logging.info('Existed NaN in model parameters')
    return torch.cat([param.data.view(-1) for param in model.parameters()])


def reload_params(model, flat_params):
    """
    Reload the parameters from a flattened tensor back into the model.

    Args:
        model (nn.Module): The model to reload the parameters into.
        flat_params (torch.Tensor): A 1D tensor containing the parameters.
    """
    idx = 0
    for param in model.parameters():
        param_length = param.numel()
        param.data.copy_(flat_params[idx:idx + param_length].view(param.size()))
        idx += param_length


