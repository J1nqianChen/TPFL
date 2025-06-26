import copy
import os
import sys

import numpy as np
import logging
import torch
import torch.nn as nn
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
from tqdm import tqdm

current_dir = os.path.abspath(os.path.dirname(__file__))

parts = current_dir.split(os.sep)
target_parts = parts[:-4]
path_ = os.path.join('/', *target_parts)
sys.path.append(path_)

from PFL.dataset.Read_Data import read_cifar_c
from utils.data_utils import read_client_data, read_client_data_drop, add_backdoor_pattern_tensor, find_z_max, \
    reload_params




class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, config, id, train_samples, test_samples, init_model):

        self.model = copy.deepcopy(init_model)
        self.dataset = config.train.dataset_name
        self.device = config.general.device
        self.id = id  # integer
        self.save_path = config.general.save_path
        self.times = None
        self.num_classes = config.train.num_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = config.train.batch_size
        self.learning_rate = config.general.local_learning_rate
        self.local_steps = config.train.local_epochs
        self.learning_rate_decay = config.general.learning_rate_decay

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9,
                                         weight_decay=1e-5)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer,
            gamma=config.general.learning_rate_decay_gamma
        )
        self.learning_rate_decay = config.general.learning_rate_decay

        # check BatchNorm
        self.has_BatchNorm = False
        for layer in self.model.children():
            if isinstance(layer, nn.BatchNorm2d):
                self.has_BatchNorm = True
                break

        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

        self.sample_rate = self.batch_size / self.train_samples
        self.save_local_model = config.general.save_local_model
        self.curr_round = 0
        self.last_loss = 0

        self.check_step = config.general.check_step
        self.algorithm = config.train.algorithm
        self.join_ratio = config.train.participation_ratio
        self.flipped_data = None
        self.is_malicious = False
        self.attack_strategy = None
        self.attack_round = None

        self.num_clients = config.train.num_clients

        self.attack_ratio = config.attack.ratio
        self.corrupted_num = int(self.num_clients * self.attack_ratio)
        # LIE
        self.z_max = None
        self.benign_mean = None
        self.benign_var = None


    def set_benign_mean_var(self, benign_mean, benign_var):
        self.benign_mean = benign_mean
        self.benign_var = benign_var

    def set_malicious(self, attack_st, attack_round=None):
        self.is_malicious = True
        self.attack_strategy = attack_st
        self.attack_round = attack_round

    def set_curr_round(self, round):
        self.curr_round = round

    def load_train_data(self, batch_size=None, origin=True, drop_last=True, transform=None, num_worker=None):
        if batch_size == None:
            batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.id, is_train=True, transform=transform)
        if not origin:
            if self.is_malicious:
                if self.attack_strategy == 'flip':
                    if self.flipped_data is None:
                        raise RuntimeError
                    train_data = self.flipped_data
                    logging.info('Load Flipped Data Successfully')
        if num_worker is not None:
            return DataLoader(train_data, batch_size, drop_last=drop_last, shuffle=True, pin_memory=True,
                              num_workers=num_worker)
        else:
            return DataLoader(train_data, batch_size, drop_last=drop_last, shuffle=True, pin_memory=False)

    def set_times(self, times):
        self.times = times
        logging.info(f'Client {self.id} 成功设置Times：{self.times}')

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        test_data, _ = read_client_data(self.dataset, self.id, is_train=False, times=self.times)
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=True)

    def load_val_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        _, val_data = read_client_data(self.dataset, self.id, is_train=False, times=self.times)
        return DataLoader(val_data, batch_size, drop_last=False, shuffle=True)

    def set_parameters_1(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

    def set_parameters(self, model):
        # state_dict = model.state_dict()
        # self.model.load_state_dict(state_dict)
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

    def clone_model(self, model, target):
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()
            # target_param.grad = param.grad.clone()

    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()

    def test_metrics(self, backdoor_evaluate=False, use_val=False):
        if not use_val:
            testloaderfull = self.load_test_data()
        else:
            testloaderfull = self.load_val_data()
        self.model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []

        with torch.no_grad():
            for x, y in testloaderfull:
                if backdoor_evaluate:
                    x, y = add_backdoor_pattern_tensor(x, y, insert_num=x.shape[0])

                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                output = self.model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()

                test_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())
                y_true.append(label_binarize(y.detach().cpu().numpy(), classes=np.arange(self.num_classes)))

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        # auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        return test_acc, test_num, 0


    def train_metrics(self):
        trainloader = self.load_train_data(origin=True)
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        train_num = 0
        loss = 0
        for x, y in trainloader:
            if type(x) == type([]):
                x[0] = x[0].to(self.device)
            else:
                x = x.to(self.device)
            y = y.to(self.device)
            output = self.model(x)
            train_num += y.shape[0]
            # logging.info(output.shape)
            # logging.info(y.shape)
            loss += self.loss(output, y).item() * y.shape[0]

        self.last_loss = loss

        return loss, train_num

    # def get_next_train_batch(self):
    #     try:
    #         # Samples a new batch for persionalizing
    #         (x, y) = next(self.iter_trainloader)
    #     except StopIteration:
    #         # restart the generator if the previous generator is exhausted.
    #         self.iter_trainloader = iter(self.trainloader)
    #         (x, y) = next(self.iter_trainloader)

    #     if type(x) == type([]):
    #         x = x[0]
    #     x = x.to(self.device)
    #     y = y.to(self.device)

    #     return x, y

    def save_item(self, item, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_path
        if not os.path.exists(item_path):
            os.makedirs(item_path)
        torch.save(item, os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    def load_item(self, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_path
        return torch.load(os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    # @staticmethod
    # def model_exists():
    #     return os.path.exists(os.path.join("models", "server" + ".pt"))

    def get_label_distribution(self, origin=True):
        trainloader = self.load_train_data(origin=origin)
        count = np.zeros([self.num_classes])
        for x, y in tqdm(trainloader):
            for class_idx in range(self.num_classes):
                mask = y == class_idx
                masked_y = y[mask].tolist()
                count[class_idx] += len(masked_y)
        return count

    def flip_dataset(self, original_y=5, target_y=3):

        # Targeted Flip Default: Cat to Dog
        trainloader = self.load_train_data(origin=True)
        x_data = []
        y_data = []
        for x, y in tqdm(trainloader):
            mask = y == original_y
            if mask.any():
                y[mask] = target_y
            # y[mask] = np.random.randint(0, self.num_classes)
            x_data.append(x)
            y_data.append(y)
        x_data = torch.cat(x_data, dim=0)
        y_data = torch.cat(y_data, dim=0)
        flipped_data = [(x, y) for x, y in zip(x_data, y_data)]
        self.flipped_data = flipped_data
        logging.info('flipping')

    # Targeted Attack Utils

    def attack_mp(self):
        if not self.is_malicious:
            raise RuntimeError
        else:
            if self.attack_strategy == 'LIE':

                if self.z_max is None:
                    self.z_max = find_z_max(n=self.num_clients, m=self.corrupted_num)
                    logging.info(f'z_max: {self.z_max}')
                if self.benign_mean is None:
                    logging.info('Initialize')
                    for param in self.model.parameters():
                        param.data = torch.randn_like(param)
                else:
                    logging.info('Attack LIE')
                    # logging.info(self.benign_mean)
                    # logging.info(self.benign_var)
                    sample_param = (self.benign_mean - self.z_max * torch.sqrt(self.benign_var))
                    noise = torch.randn_like(self.benign_mean) * (0.05 * torch.sqrt(self.benign_var))
                    sample_param = sample_param + noise
                    reload_params(self.model, sample_param)

            elif self.attack_strategy == 'Gaussian':
                if self.benign_mean is None:
                    logging.info('Initialize')
                    for param in self.model.parameters():
                        param.data = torch.randn_like(param)
                else:
                    logging.info('Gaussian Attack')
                    sample_param = torch.normal(self.benign_mean, torch.sqrt(self.benign_var))
                    reload_params(self.model, sample_param)
            elif self.attack_strategy == 'Random':
                logging.info('Random Attack')
                for param in self.model.parameters():
                    param.data = torch.randn_like(param.data)

            else:
                raise NotImplementedError

    def evaluate_corruption(self):

        with torch.no_grad():
            list_acc = []
            for severity_i in range(5):
                severity_ = severity_i + 1
                corr_dataset, sample_number = read_cifar_c(self.dataset + '_C', client_id=self.id, severity=severity_)
                corr_dataloader = DataLoader(corr_dataset, batch_size=self.batch_size, shuffle=True)
                test_acc = 0
                test_num = 0

                for x, y in iter(corr_dataloader):

                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)

                    # print(self.model.device)
                    output = self.model(x)

                    test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()

                    test_num += y.shape[0]

                acc_severity_i = test_acc / test_num
                list_acc.append(acc_severity_i)
            logging.info(
                'Please Double Check the Code. This is the result from basic clientbase, which may be not suitable for '
                'pFL methods.')
            # logging.info(list_acc)
            return np.array(list_acc), test_num

    def class_wise_evaluate(self, model):
        model.eval()
        model = model.to(self.device)

        # Initialize counters for each class
        correct_per_class = np.zeros(self.num_classes)
        total_per_class = np.zeros(self.num_classes)

        dataloader = self.load_test_data()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Calculating class-wise accuracy"):
                inputs, labels = batch
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # Get model predictions
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)

                # Update counters for each class
                for label, pred in zip(labels, predicted):
                    if label == pred:
                        correct_per_class[label.item()] += 1
                    total_per_class[label.item()] += 1

        # Calculate accuracy for each class
        # Handle division by zero for classes with no samples
        classwise_accuracy = np.zeros(self.num_classes)
        for i in range(self.num_classes):
            if total_per_class[i] > 0:
                classwise_accuracy[i] = correct_per_class[i] / total_per_class[i]

        return classwise_accuracy
