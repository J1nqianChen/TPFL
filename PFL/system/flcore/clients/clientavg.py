import copy
import logging
import time

import torch
import torch.nn as nn

from flcore.clients.clientbase import Client
from utils.data_utils import add_backdoor_pattern_tensor


class clientAVG(Client):
    def __init__(self, config, id, train_samples, test_samples, init_model):
        super().__init__(config, id, train_samples, test_samples, init_model)

    def train(self):
        start_time = time.time()
        if not self.is_malicious or self.attack_strategy == 'flip':
            trainloader = self.load_train_data(origin=True)
            attack_flag = False
            if self.attack_strategy == 'flip':
                attack_flag = False
                if self.curr_round in self.attack_round or -1 in self.attack_round:
                    attack_flag = True

                if attack_flag:
                    trainloader = self.load_train_data(origin=False)
                    global_model = copy.deepcopy(self.model)


            # self.model.to(self.device)
            self.model.train()

            max_local_steps = self.local_steps

            for step in range(max_local_steps):
                for i, (x, y) in enumerate(trainloader):
                    # visualize_tensor_images(x)
                    if self.is_malicious and self.attack_strategy == 'backdoor_pattern' and (
                            self.curr_round in self.attack_round or -1 in self.attack_round):
                        x, y = add_backdoor_pattern_tensor(x, y, insert_num=40)

                    # visualize_tensor_images(x)
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

            if attack_flag:     # Combine Flip with Model Replacement
                logging.info('Model Replacement')
                for param_client, param_global in zip(self.model.parameters(), global_model.parameters()):
                    param_client.data = self.num_clients * param_client.data - (
                            self.num_clients - 1) * param_global.data


        elif self.is_malicious and self.attack_strategy in ['LIE', 'Random']:
            self.attack_mp()

        else:
            raise NotImplementedError
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
