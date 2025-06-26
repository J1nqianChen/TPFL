import os
import sys

import torch
import torch.nn as nn
import numpy as np
import logging
import time

from torch.utils.data import DataLoader

current_dir = os.path.abspath(os.path.dirname(__file__))

parts = current_dir.split(os.sep)
target_parts = parts[:-3]
path_ = os.path.join('/', *target_parts)
sys.path.append(path_)

from dataset.Read_Data import read_cifar_c
from flcore.clients.clientbase import Client
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from torch.autograd import Variable


class clientDBE(Client):
    def __init__(self, config, id, train_samples, test_samples, init_model):
        super().__init__(config, id, train_samples, test_samples, init_model)

        # self.klw = args.kl_weight
        # self.momentum = args.momentum

        if 'cnn' in config.train.model_str:
            self.klw = config.algo_hyperparam.DBE.kappa.cnn
            self.momentum = config.algo_hyperparam.DBE.mu.cnn
        elif 'resnet' in config.train.model_str:
            self.klw = config.algo_hyperparam.DBE.kappa.resnet
            self.momentum = config.algo_hyperparam.DBE.mu.resnet
        elif 'vit' in config.train.model_str:
            self.klw = config.algo_hyperparam.DBE.kappa.vit
            self.momentum = config.algo_hyperparam.DBE.mu.vit
        else:
            raise NotImplementedError

        self.global_mean = None

        trainloader = self.load_train_data()
        for x, y in trainloader:
            if type(x) == type([]):
                x[0] = x[0].to(self.device)
            else:
                x = x.to(self.device)
            y = y.to(self.device)
            with torch.no_grad():
                rep = self.model.base(x).detach()
            break
        self.running_mean = torch.zeros_like(rep[0])
        self.num_batches_tracked = torch.tensor(0, dtype=torch.long, device=self.device)

        self.client_mean = nn.Parameter(Variable(torch.zeros_like(rep[0])))
        self.opt_client_mean = torch.optim.SGD([self.client_mean], lr=self.learning_rate)

    def train(self):
        trainloader = self.load_train_data()
        # self.model.to(self.device)
        self.model.train()

        start_time = time.time()

        max_local_epochs = self.local_steps
        self.reset_running_stats()
        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                # ====== begin
                rep = self.model.base(x)
                running_mean = torch.mean(rep, dim=0)

                if self.num_batches_tracked is not None:
                    self.num_batches_tracked.add_(1)

                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * running_mean

                if self.global_mean is not None:
                    reg_loss = torch.mean(0.5 * (self.running_mean - self.global_mean) ** 2)
                    output = self.model.head(rep + self.client_mean)
                    loss = self.loss(output, y)
                    loss = loss + reg_loss * self.klw
                else:
                    output = self.model.head(rep)
                    loss = self.loss(output, y)
                # ====== end

                self.opt_client_mean.zero_grad()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.opt_client_mean.step()
                self.detach_running()

        # self.model.cpu()
        #
        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def reset_running_stats(self):
        self.running_mean.zero_()
        self.num_batches_tracked.zero_()

    def detach_running(self):
        self.running_mean.detach_()

    def train_metrics(self):
        trainloader = self.load_train_data()
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                rep = self.model.base(x)
                output = self.model.head(rep + self.client_mean)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num

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
        reps = []

        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                rep = self.model.base(x)
                output = self.model.head(rep + self.client_mean)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)
                reps.extend(rep.detach())

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')

        return test_acc, test_num, auc


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
                    rep = self.model.base(x)
                    output = self.model.head(rep + self.client_mean)

                    test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()

                    test_num += y.shape[0]

                acc_severity_i = test_acc / test_num
                list_acc.append(acc_severity_i)
            logging.info(
                'DBE Corruption Evaluation')
            # logging.info(list_acc)
            return np.array(list_acc), test_num