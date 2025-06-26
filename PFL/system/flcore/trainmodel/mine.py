import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import numpy as np
import logging


class Mine(nn.Module):
    def __init__(self, input_size=2, hidden_size=100):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # self.fc3 = nn.Linear(hidden_size, hidden_size)
        # self.fc4 = nn.Linear(hidden_size, hidden_size)
        # self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.normal_(self.fc2.weight, std=0.02)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.normal_(self.fc3.weight, std=0.02)
        nn.init.constant_(self.fc3.bias, 0)
        # nn.init.normal_(self.fc4.weight, std=0.02)
        # nn.init.constant_(self.fc4.bias, 0)
        # nn.init.normal_(self.fc5.weight, std=0.02)
        # nn.init.constant_(self.fc5.bias, 0)
        # nn.init.normal_(self.fc6.weight, std=0.02)
        # nn.init.constant_(self.fc6.bias, 0)

    def forward(self, input):
        output = F.relu(self.fc1(input))
        output = F.relu(self.fc2(output))
        # output = F.relu(self.fc3(output))
        # output = F.relu(self.fc4(output))
        # output = F.relu(self.fc5(output))
        output = self.fc3(output)
        return output

    def reset(self):
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.normal_(self.fc2.weight, std=0.02)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.normal_(self.fc3.weight, std=0.02)
        nn.init.constant_(self.fc3.bias, 0)


def mutual_information(joint, marginal, mine_net):
    # logging.info(joint.shape)
    t = mine_net(joint)
    et = torch.exp(mine_net(marginal))
    mi_lb = torch.mean(t) - torch.log(torch.mean(et))
    return mi_lb, t, et


def learn_mine(batch, mine_net, mine_net_optim, ma_et, ma_rate=0.01):
    # batch is a tuple of (joint, marginal)
    joint, marginal = batch
    # logging.info(type(joint))
    mine_net.train()

    mi_lb, t, et = mutual_information(joint, marginal, mine_net)
    mi_lb = mi_lb
    ma_et = (1 - ma_rate) * ma_et + ma_rate * torch.mean(et)

    # unbiasing use moving average
    loss = -(torch.mean(t) - (1 / ma_et.mean()).detach() * torch.mean(et))
    # use biased estimator
    #     loss = - mi_lb

    mine_net_optim.zero_grad()
    loss.backward()
    mine_net_optim.step()
    return mi_lb, ma_et




def sample_batch(data, batch_size=100, sample_mode='joint'):
    if sample_mode == 'joint':
        index = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
        batch = data[index, :, :]
    else:
        joint_index = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
        marginal_index = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
        batch = torch.stack([data[joint_index, :, 0],
                          data[marginal_index, :, 1]],
                         dim=2)
    # logging.info(batch.shape)
    batch = batch.reshape(batch.shape[0], -1)
    return batch


def train(data, mine_net, mine_net_optim, batch_size=100, iter_num=int(5e+2), log_freq=int(1e+3)):
    # data is x or y

    result = list()
    ma_et = 1.
    for i in range(iter_num):
        batch = sample_batch(data, batch_size=batch_size) \
            , sample_batch(data, batch_size=batch_size, sample_mode='marginal')
        mi_lb, ma_et = learn_mine(batch, mine_net, mine_net_optim, ma_et)
        result.append(mi_lb)
        if (i + 1) % (log_freq) == 0:
            logging.info(result[-1])
    return result


def test(data, mine_net, batch_size=100):
    # data is x or y
    result = list()
    ma_et = 1.
    batch = sample_batch(data, batch_size=batch_size) \
        , sample_batch(data, batch_size=batch_size, sample_mode='marginal')

    joint, marginal = batch
    mine_net.eval()
    mi_lb, t, et = mutual_information(joint, marginal, mine_net)

    return mi_lb

