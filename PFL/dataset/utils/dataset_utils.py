import hashlib
import os

import numpy as np
import logging
import torch
import torchvision
import torchvision.transforms
import ujson
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder, ImageFolder
from torchvision.transforms import transforms
from tqdm import tqdm

train_size = 0.7  # merge original training set and test set, then split it manually.
# train : val : test = 7 : 1 : 2
# least_samples = -1
# alpha = 0.5  # for Dirichlet distribution
# alpha = 1.0  # for Dirichlet distribution
np.random.seed(1234)


class ImageFolder_custom(DatasetFolder):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        imagefolder_obj = ImageFolder(self.root, self.transform, self.target_transform)
        self.loader = imagefolder_obj.loader
        if self.dataidxs is not None:
            self.samples = np.array(imagefolder_obj.samples)[self.dataidxs]
        else:
            self.samples = np.array(imagefolder_obj.samples)

    def __getitem__(self, index):
        path = self.samples[index][0]
        target = self.samples[index][1]
        target = int(target)
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        if self.dataidxs is None:
            return len(self.samples)
        else:
            return len(self.dataidxs)


def check(config_path, train_path, test_path, num_clients, num_classes, niid=False,
          balance=True, partition=None, dir_param=0.1):
    logging.info(f'check client {num_clients} dir {dir_param}')
    # check existing dataset
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = ujson.load(f)
        if config['num_clients'] == num_clients and \
                config['num_classes'] == num_classes and \
                config['non_iid'] == niid and \
                config['balance'] == balance and \
                config['partition'] == partition and \
                config['alpha'] == dir_param:
            logging.info("\nDataset already generated.\n")
            return True

    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    return False


def separate_data(data, num_clients, num_classes, niid=False, balance=False, partition=None, dataset_name='Cifar10',
                  class_per_client=2, dir_param=0.1):
    X = [[] for _ in range(num_clients)]
    y = [[] for _ in range(num_clients)]
    statistic = [[] for _ in range(num_clients)]

    dataset_content, dataset_label = data

    dataidx_map = {}

    if dataset_name == 'Cifar10' or dataset_name == 'Cifar100':
        if num_clients == 20 or num_clients == 50:
            batch_size = 128
        elif num_clients == 100:
            batch_size = 50
        elif num_clients == 500 or num_clients == 1000:
            batch_size = 10
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    least_samples = batch_size / train_size  # least samples for each client

    if not niid:
        partition = 'pat'
        class_per_client = num_classes

    if partition == 'pat':
        logging.info('Pat NonIID')
        if dataset_name == 'Cifar10':
            class_per_client = 2
        elif dataset_name == 'Cifar100':
            class_per_client = 20
        logging.info(class_per_client)
        idxs = np.array(range(len(dataset_label)))
        idx_for_each_class = []
        for i in range(num_classes):
            idx_for_each_class.append(idxs[dataset_label == i])

        class_num_per_client = [class_per_client for _ in range(num_clients)]
        for i in range(num_classes):
            selected_clients = []
            for client in range(num_clients):
                if class_num_per_client[client] > 0:
                    selected_clients.append(client)
            if len(selected_clients) == 0:
                break
            selected_clients = selected_clients[:int(np.ceil((num_clients / num_classes) * class_per_client))]

            num_all_samples = len(idx_for_each_class[i])
            num_selected_clients = len(selected_clients)
            num_per = num_all_samples / num_selected_clients
            if balance:
                num_samples = [int(num_per) for _ in range(num_selected_clients - 1)]
            else:
                num_samples = np.random.randint(max(num_per / 10, least_samples / num_classes), num_per,
                                                num_selected_clients - 1).tolist()
            num_samples.append(num_all_samples - sum(num_samples))

            idx = 0
            for client, num_sample in zip(selected_clients, num_samples):
                if client not in dataidx_map.keys():
                    dataidx_map[client] = idx_for_each_class[i][idx:idx + num_sample]
                else:
                    dataidx_map[client] = np.append(dataidx_map[client], idx_for_each_class[i][idx:idx + num_sample],
                                                    axis=0)
                idx += num_sample
                class_num_per_client[client] -= 1

    elif partition == "dir":
        # https://github.com/IBM/probabilistic-federated-neural-matching/blob/master/experiment.py
        min_size = 0
        K = num_classes
        N = len(dataset_label)

        try_cnt = 1
        while min_size < least_samples:
            if try_cnt > 1:
                logging.info(
                    f'Client data size does not meet the minimum requirement {least_samples}. Try allocating again for the {try_cnt}-th time.')

            idx_batch = [[] for _ in range(num_clients)]
            for k in range(K):
                idx_k = np.where(dataset_label == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(dir_param, num_clients))
                proportions = np.array([p * (len(idx_j) < N / num_clients) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
            try_cnt += 1

        for j in range(num_clients):
            dataidx_map[j] = idx_batch[j]

    else:
        raise NotImplementedError

    # assign data
    for client in range(num_clients):
        idxs = dataidx_map[client]
        X[client] = dataset_content[idxs]
        y[client] = dataset_label[idxs]

        for i in np.unique(y[client]):
            statistic[client].append((int(i), int(sum(y[client] == i))))

    del data
    # gc.collect()

    for client in range(num_clients):
        logging.info(f"Client {client}\t Size of data: {len(X[client])}\t Labels: ", np.unique(y[client]))
        logging.info(f"\t\t Samples of labels: ", [i for i in statistic[client]])
        logging.info("-" * 50)

    return X, y, statistic, batch_size


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
    # logging.info()
    del X, y
    # gc.collect()

    return train_data, test_data


def save_file(config_path, train_path, test_path, train_data, test_data, num_clients,
              num_classes, statistic, niid=False, balance=True, partition=None, dir_param=0.1, batch_size=None):
    logging.info(f'save client {num_clients} dir {dir_param}')
    config = {
        'num_clients': num_clients,
        'num_classes': num_classes,
        'non_iid': niid,
        'balance': balance,
        'partition': partition,
        'Size of samples for labels in clients': statistic,
        'alpha': dir_param,
        'batch_size': batch_size,
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


# def visualize(statistic, save_path, alpha):
#     colors = ['greenyellow', 'yellow', 'turquoise', 'deepskyblue', 'orange', 'darkcyan', 'palegreen', 'forestgreen',
#               'gold',
#               'olivedrab', 'lime', 'teal', 'salmon', 'sandbrown']
#     plt.figure()
#     for i in range(len(statistic)):
#         temp = statistic[i]
#         logging.info(len(temp))
#         sum = 0
#         for tuple in temp:
#             plt.bar([f'{i}'], tuple[1], bottom=sum, color=colors[tuple[0]], width=0.5, label=f"class {tuple[0]}")
#             sum += tuple[1]
#     path = os.path.join(save_path, f'result_{alpha}.png')
#     plt.savefig(path)


def generate_one_set(dataset, num_class):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    if 'Cifar10_' in dataset:
        dir_path = 'Cifar10'
        trainset = torchvision.datasets.CIFAR10(
            root=dir_path + "rawdata", train=True, download=True, transform=transform)
        trainloader = DataLoader(trainset, batch_size=1, shuffle=False)

    elif 'Cifar100_' in dataset:
        dir_path = 'Cifar100'
        trainset = torchvision.datasets.CIFAR100(
            root=dir_path + "rawdata", train=True, download=True, transform=transform)
        trainloader = DataLoader(trainset, batch_size=1, shuffle=False)
    elif 'Tiny-imagenet' in dataset:
        dir_path = "Tiny-imagenet/"
        trainset = ImageFolder_custom(root=dir_path + 'rawdata/tiny-imagenet-200/train/', transform=transform)
        trainloader = DataLoader(trainset, batch_size=1, shuffle=False)
    elif 'mnist' in dataset:
        dir_path = 'mnist'
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

        trainset = torchvision.datasets.MNIST(
            root=dir_path + "rawdata", train=True, download=True, transform=transform)
        trainloader = DataLoader(trainset, batch_size=1, shuffle=False)
    else:
        exit(-1)
    target = 0
    list_share = []
    for x, y in tqdm(trainloader):
        if y[0] == target:
            list_share.append(x)
            target += 1
        if target == num_class:
            break

    one_tensor = torch.cat(list_share, dim=0).cuda()
    return one_tensor


def extract_trustworthy_holdout_set(dataset='SVHN'):
    import torch
    from torchvision import datasets, transforms
    import numpy as np
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Define transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load SVHN dataset
    svhn_dataset = datasets.SVHN(root='./data', split='train', download=True, transform=transform)
    # svhn_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Set the size of the mini-set
    mini_set_size = 100

    # Create a random sampler to sample mini-set
    indices = np.random.choice(len(svhn_dataset), mini_set_size, replace=False)
    mini_set_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices)

    # Create data loader for mini-set
    mini_set_loader = torch.utils.data.DataLoader(svhn_dataset, batch_size=mini_set_size, sampler=mini_set_sampler)

    # Iterate over mini-set loader
    for images, labels in mini_set_loader:
        # Save mini-set to disk
        torch.save({'images': images, 'labels': labels}, '../../system/trust_hold_cifar10.pth')
        logging.info("Mini-set saved successfully!")
        break  # Stop iteration after first batch

    # # Reload mini-set from disk
    # loaded_data = torch.load('./mini_set.pth')
    # reloaded_images = loaded_data['images']
    # reloaded_labels = loaded_data['labels']
    #
    # # Now you can use reloaded_images and reloaded_labels as your mini-set
    # logging.info("Images shape after reloading:", reloaded_images.shape)
    # logging.info("Labels after reloading:", reloaded_labels)


def extract_trustworthy_holdout_set_balanced(dataset='SVHN', samples_per_class=10):
    import torch
    from torchvision import datasets, transforms
    import numpy as np
    from collections import defaultdict
    import os

    # 设置随机种子确保可复现
    torch.manual_seed(42)
    np.random.seed(42)

    # 图像预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) if dataset == 'FashionMNIST' else transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 加载数据集
    if dataset == 'SVHN':
        full_dataset = datasets.SVHN(root='./data', split='train', download=True, transform=transform)
        num_classes = 10
    elif dataset == 'CIFAR10':
        full_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        num_classes = 10
    elif dataset == 'CIFAR100':
        full_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        num_classes = 100
    elif dataset == 'FashionMNIST':
        full_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        num_classes = 10
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    # 创建类别到样本索引的映射
    label_to_indices = defaultdict(list)
    for idx, (_, label) in enumerate(full_dataset):
        if isinstance(label, torch.Tensor):  # SVHN labels are tensors
            label = label.item()
        label_to_indices[label].append(idx)

    # 均匀采样每类 samples_per_class 个样本
    selected_indices = []
    for cls in range(num_classes):
        cls_indices = label_to_indices[cls]
        if len(cls_indices) < samples_per_class:
            raise ValueError(f"Not enough samples in class {cls} to extract {samples_per_class} instances.")
        np.random.shuffle(cls_indices)
        selected_indices.extend(cls_indices[:samples_per_class])

    # 构建 mini-set DataLoader
    sampler = torch.utils.data.SubsetRandomSampler(selected_indices)
    loader = torch.utils.data.DataLoader(full_dataset, batch_size=len(selected_indices), sampler=sampler)

    # 仅取一批数据并保存
    for images, labels in loader:
        save_path = f'../../system/trust_hold_{dataset.lower()}_{samples_per_class}.pth'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({'images': images, 'labels': labels}, save_path)
        print(f"[INFO] Balanced mini-set saved to: {save_path}")
        break




def compute_dataset_checksum(path, total_number):
    """
    Computes a SHA-256 checksum for a dataset stored in numbered .npz files.

    Args:
        path (str): Directory path where the .npz files are stored.
        total_number (int): Total number of .npz files in the dataset.

    Returns:
        str: Hexadecimal SHA-256 checksum of the dataset.
    """
    np.random.seed(123)
    overall_hash = hashlib.sha256()

    for i in range(total_number):
        file_path = os.path.join(path, f"{i}.npz")
        arr = np.load(file_path, allow_pickle=True)['data'].tolist()['x']
        # assert arr.flags.c_contiguous, f"文件 {i} 内存布局非C顺序"
        arr_bytes = arr.tobytes()
        arr_hash = hashlib.sha256(arr_bytes).digest()
        overall_hash.update(arr_hash)

    return overall_hash.hexdigest()


if __name__ == '__main__':
    # list = generate_one_set()
    # logging.info(len(list))
    # extract_trustworthy_holdout_set()
    # str1 = compute_dataset_checksum('../Cifar10_NonIID_Dir0.1_Client20/train', 20)
    # str2 = compute_dataset_checksum('../Cifar10_NonIID_Dir0.1_Client20/train', 20)
    # logging.info(str1)
    # logging.info(str2)

    extract_trustworthy_holdout_set_balanced(samples_per_class=100)