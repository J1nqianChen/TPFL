import copy
import sys
import time

import numpy as np
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append('/home/chenjinqian/code/MINE_FL/')
from PFL.dataset.Read_Data import read_cifar_c
from flcore.optimizers.fedoptimizer import PerturbedGradientDescent
from flcore.clients.clientbase import Client
import torch.nn.functional as F
from sklearn.preprocessing import label_binarize
from sklearn import metrics


class clientDitto(Client):
    def __init__(self, config, id, train_samples, test_samples, init_model):
        super().__init__(config, id, train_samples, test_samples, init_model)

        self.lamda = config.algo_hyperparam.Ditto.lamda.value
        self.plocal_steps = int(config.algo_hyperparam.Ditto.plocal_steps.value)

        self.pmodel = copy.deepcopy(self.model)
        self.poptimizer = PerturbedGradientDescent(
            self.pmodel.parameters(), lr=self.learning_rate, mu=self.lamda)

    def train(self):
        trainloader = self.load_train_data()
        if self.is_malicious and self.attack_strategy == 'flip':
            trainloader = self.load_train_data(origin=False)
        start_time = time.time()
        self.model.train()

        max_local_steps = self.local_steps
        for step in range(max_local_steps):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step()
        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()
        # self.model.cpu()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
        
    def ptrain(self):
        trainloader = self.load_train_data()
        if self.is_malicious and self.attack_strategy == 'flip':
            trainloader = self.load_train_data(origin=False)
        start_time = time.time()

        # self.model.to(self.device)
        self.pmodel.train()

        max_local_steps = self.plocal_steps
        for step in range(max_local_steps):
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                self.poptimizer.zero_grad()
                output = self.pmodel(x)
                loss = self.loss(output, y)
                loss.backward()
                self.poptimizer.step(self.model.parameters(), self.device)

        # self.model.cpu()

        self.train_time_cost['total_cost'] += time.time() - start_time

    def test_metrics(self, backdoor_evaluate=False, use_val=False):
        if not use_val:
            testloaderfull = self.load_test_data()
        else:
            testloaderfull = self.load_val_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.pmodel.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        
        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.pmodel(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(F.softmax(output).detach().cpu().numpy())
                y_true.append(label_binarize(y.detach().cpu().numpy(), classes=np.arange(self.num_classes)))

        # self.model.cpu()

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        
        return test_acc, test_num, auc


    def evaluate_corruption(self):

        with torch.no_grad():
            list_acc = []
            for severity_i in range(5):
                severity_ = severity_i + 1
                corr_dataset, sample_number = read_cifar_c(self.dataset+'_C', client_id=self.id, severity=severity_)
                corr_dataloader = DataLoader(corr_dataset, batch_size=self.batch_size, shuffle=True)
                test_acc = 0
                test_num = 0

                for x, y in iter(corr_dataloader):

                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)

                    output = self.pmodel(x)

                    test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()

                    test_num += y.shape[0]

                acc_severity_i = test_acc / test_num
                list_acc.append(acc_severity_i)
            logging.info('Please Double Check the Code. This is the result from basic clientbase, which may be not suitable for '
                  'pFL methods.')
            # logging.info(list_acc)
            return np.array(list_acc), test_num