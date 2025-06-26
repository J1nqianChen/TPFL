import copy
import sys
import time

import numpy as np
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append('/home/chenjinqian/code/MINE_FL/')
from PFL.dataset.Read_Data import read_cifar_c
from flcore.clients.clientbase import Client
from sklearn.preprocessing import label_binarize


class clientROD(Client):
    def __init__(self, config, id, train_samples, test_samples, init_model):
        super().__init__(config, id, train_samples, test_samples, init_model)

        self.head = copy.deepcopy(self.model.head)
        self.opt_head = torch.optim.SGD(self.head.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler_head = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.opt_head,
            gamma=config.general.learning_rate_decay_gamma
        )

        self.sample_per_class = torch.zeros(self.num_classes)
        trainloader = self.load_train_data()
        for x, y in trainloader:
            for yy in y:
                self.sample_per_class[yy.item()] += 1
        self.model_poison = config.attack.model_poison

    def train(self):
        if not self.model_poison:
            trainloader = self.load_train_data(origin=True)
            if self.is_malicious and self.attack_strategy == 'flip':
                trainloader = self.load_train_data(origin=False)

            start_time = time.time()

            # self.model.to(self.device)
            self.model.train()

            max_local_epochs = self.local_steps

            for step in range(max_local_epochs):
                for i, (x, y) in enumerate(trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    rep = self.model.base(x)
                    out_g = self.model.head(rep)
                    loss_bsm = balanced_softmax_loss(y, out_g, self.sample_per_class)
                    self.optimizer.zero_grad()
                    loss_bsm.backward()
                    self.optimizer.step()

                    out_p = self.head(rep.detach())
                    loss = self.loss(out_g.detach() + out_p, y)
                    self.opt_head.zero_grad()
                    loss.backward()
                    self.opt_head.step()

            # self.model.cpu()

            if self.learning_rate_decay:
                self.learning_rate_scheduler.step()
                self.learning_rate_scheduler_head.step()

            self.train_time_cost['num_rounds'] += 1
            self.train_time_cost['total_cost'] += time.time() - start_time
        else:
            self.attack_mp()

    def test_metrics(self, backdoor_evaluate=False, use_val=False, model=None):
        if not use_val:
            testloader = self.load_test_data()
        else:
            testloader = self.load_val_data()
        if model == None:
            model = self.model
        model.eval()
        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []

        with torch.no_grad():
            for x, y in testloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                rep = self.model.base(x)
                out_g = self.model.head(rep)
                out_p = self.head(rep.detach())
                output = out_g.detach() + out_p

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(F.softmax(output).detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        # auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        auc = 0
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
                    out_g = self.model.head(rep)
                    out_p = self.head(rep.detach())
                    output = out_g.detach() + out_p

                    test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()

                    test_num += y.shape[0]

                acc_severity_i = test_acc / test_num
                list_acc.append(acc_severity_i)
            logging.info('Rod Personalized Inference')
            # logging.info(list_acc)
            return np.array(list_acc), test_num

    def train_metrics(self):
        trainloader = self.load_train_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
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
                out_g = self.model.head(rep)
                out_p = self.head(rep.detach())
                output = out_g.detach() + out_p
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        return losses, train_num


# https://github.com/jiawei-ren/BalancedMetaSoftmax-Classification
def balanced_softmax_loss(labels, logits, sample_per_class, reduction="mean"):
    """Compute the Balanced Softmax Loss between `logits` and the ground truth `labels`.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      sample_per_class: A int tensor of size [no of classes].
      reduction: string. One of "none", "mean", "sum"
    Returns:
      loss: A float tensor. Balanced Softmax Loss.
    """
    spc = sample_per_class.type_as(logits)
    spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
    logits = logits + spc.log()
    loss = F.cross_entropy(input=logits, target=labels, reduction=reduction)
    return loss
