import time

import torch
from torch import nn

from flcore.clients.clientbase import Client
from flcore.optimizers.fedoptimizer import SCAFFOLDOptimizer


class clientSCAFFOLD(Client):
    def __init__(self, config, id, train_samples, test_samples, init_model):
        super().__init__(config, id, train_samples, test_samples, init_model)

        self.optimizer = SCAFFOLDOptimizer(self.model.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer,
            gamma=0.99
        )

        self.client_c = []
        for param in self.model.parameters():
            self.client_c.append(torch.zeros_like(param))
        self.global_c = None
        self.global_model = None

    def train(self):
        trainloader = self.load_train_data()
        # self.model.to(self.device)
        self.model.train()

        start_time = time.time()

        max_local_epochs = self.local_steps

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step(self.global_c, self.client_c)

        # self.model.cpu()
        self.num_batches = len(trainloader)
        self.update_yc(max_local_epochs)
        # self.delta_c, self.delta_y = self.delta_yc(max_local_epochs)

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def set_parameters(self, model, global_c):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

        self.global_c = global_c
        self.global_model = model

    def update_yc(self, max_local_epochs=None):
        if max_local_epochs is None:
            max_local_epochs = self.local_steps
        for ci, c, x, yi in zip(self.client_c, self.global_c, self.global_model.parameters(), self.model.parameters()):
            ci.data = ci - c + 1 / self.num_batches / max_local_epochs / self.learning_rate * (x - yi)

    def delta_yc(self, max_local_epochs=None):
        if max_local_epochs is None:
            max_local_epochs = self.local_steps
        delta_y = []
        delta_c = []
        for c, x, yi in zip(self.global_c, self.global_model.parameters(), self.model.parameters()):
            delta_y.append(yi - x)
            delta_c.append(- c + 1 / self.num_batches / max_local_epochs / self.learning_rate * (x - yi))

        return delta_y, delta_c
