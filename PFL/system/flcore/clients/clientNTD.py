import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from flcore.clients.clientbase import Client


class clientNTD(Client):
    def __init__(self, config, id, train_samples, test_samples, init_model):
        super().__init__(config, id, train_samples, test_samples, init_model)

        self.beta = config.algo_hyperparam.FedNTD.beta.value
        self.tau = config.algo_hyperparam.FedNTD.tau.value

        self.global_model = None
        self.KLDiv = nn.KLDivLoss(reduction="batchmean")

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
                output_g = self.global_model(x)
                loss = self.loss(output, y)
                loss += self._ntd_loss(output, output_g, y) * self.beta
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

        self.global_model = model.eval().requires_grad_(False)

    # https://github.com/Lee-Gihun/FedNTD/blob/master/algorithms/fedntd/criterion.py#L30
    def _ntd_loss(self, logits, dg_logits, targets):
        """Not-tue Distillation Loss"""

        # Get smoothed local model prediction
        logits = refine_as_not_true(logits, targets, self.num_classes)
        pred_probs = F.log_softmax(logits / self.tau, dim=1)

        # Get smoothed global model prediction
        with torch.no_grad():
            dg_logits = refine_as_not_true(dg_logits, targets, self.num_classes)
            dg_probs = torch.softmax(dg_logits / self.tau, dim=1)

        loss = (self.tau ** 2) * self.KLDiv(pred_probs, dg_probs)

        return loss


# https://github.com/Lee-Gihun/FedNTD/blob/master/algorithms/fedntd/utils.py#L6
def refine_as_not_true(logits, targets, num_classes):
    nt_positions = torch.arange(0, num_classes).to(logits.device)
    nt_positions = nt_positions.repeat(logits.size(0), 1)
    nt_positions = nt_positions[nt_positions[:, :] != targets.view(-1, 1)]
    nt_positions = nt_positions.view(-1, num_classes - 1)

    logits = torch.gather(logits, 1, nt_positions)

    return logits