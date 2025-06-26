# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

from flcore.clients.clientgpfl import *
from flcore.servers.serverbase import Server
import logging

from utils.data_utils import read_client_data


class GPFL(Server):
    def __init__(self, config, times, init_model):
        super().__init__(config, times, init_model)
        self.feature_dim = list(init_model.head.parameters())[0].shape[1]
        server_GCE = GCE(in_features=self.feature_dim,
                                              num_classes=config.train.num_classes,
                                              dev=config.general.device).to(config.general.device)
        server_CoV = CoV(self.feature_dim).to(config.general.device)
        self.set_clients_gpfl(config, server_GCE, server_CoV)

        logging.info(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        logging.info("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

    def set_clients_gpfl(self, config, server_GCE, server_CoV):
        for i in range(self.num_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False, times=self.times)
            client = clientGPFL(config,
                                id=i,
                                train_samples=len(train_data),
                                test_samples=len(test_data),
                                init_model=copy.deepcopy(self.global_model), server_GCE=server_GCE,
                                server_CoV=server_CoV)
            self.clients.append(client)
            drop_len = len(train_data) % self.batch_size
            logging.info(f'client {i} drop {drop_len} samples')
        if self.attack_flag:
            logging.info(f'attack strategy: {self.attack_strategy}')
            for m_id in self.malicious_id_list:
                logging.info(f'malicious_client: {m_id}')
                client_malicious = self.clients[m_id]
                client_malicious.set_malicious(self.attack_strategy, self.attack_round)
                logging.info(client_malicious.attack_round)
                if self.attack_strategy == 'flip':
                    class_dis = client_malicious.get_label_distribution(origin=True)
                    min_indices = np.argsort(class_dis)[:3]
                    max_indices = np.argmax(class_dis)
                    original_y = max_indices
                    target_y = np.random.choice(min_indices)
                    if original_y == target_y:
                        raise RuntimeError
                    if class_dis[original_y] == 0:
                        raise RuntimeError
                    client_malicious.flip_dataset(original_y=original_y, target_y=target_y)
                    logging.info(f'flip {original_y} to {target_y}')

        for c in self.clients:
            c.set_times(self.times)
    def train(self):
        for i in range(self.global_rounds + 1):
            self.curr_round = i
            s_t = time.time()
            self.selected_clients = self.select_clients()

            if len(self.selected_clients) == 0:
                continue

            if i % self.eval_gap == 0:
                logging.info(f"\n-------------Round number: {i}-------------")
                logging.info("\nEvaluate performance")
                self.evaluate()
                if self.check_early_stop():
                    logging.info('五轮精度波动小于0.0001，退出训练')
                    break
            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            self.aggregate_parameters()
            self.send_models()

            self.global_GCE()
            self.global_CoV()

            self.Budget.append(time.time() - s_t)
            logging.info('-' * 25 + 'time cost' + '-' * 25 + str(self.Budget[-1]))

            # if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
            #     break

        logging.info("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        logging.info(max(self.rs_test_acc))
        logging.info("\nAverage time cost per round.")
        logging.info(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_train_samples = 0
        for client in self.selected_clients:
            active_train_samples += client.train_samples

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []
        for client in self.selected_clients:
            self.uploaded_weights.append(client.train_samples / active_train_samples)
            self.uploaded_ids.append(client.id)
            self.uploaded_models.append(client.model.base)

    def global_GCE(self):
        active_train_samples = 0
        for client in self.selected_clients:
            active_train_samples += client.train_samples

        self.uploaded_weights = []
        self.uploaded_model_gs = []
        for client in self.selected_clients:
            self.uploaded_weights.append(client.train_samples / active_train_samples)
            self.uploaded_model_gs.append(client.GCE)

        self.GCE = copy.deepcopy(self.uploaded_model_gs[0])
        for param in self.GCE.parameters():
            param.data = torch.zeros_like(param.data)

        for w, client_model in zip(self.uploaded_weights, self.uploaded_model_gs):
            self.add_GCE(w, client_model)

        for client in self.clients:
            client.set_GCE(self.GCE)

    def add_GCE(self, w, GCE):
        for server_param, client_param in zip(self.GCE.parameters(), GCE.parameters()):
            server_param.data += client_param.data.clone() * w

    def global_CoV(self):
        active_train_samples = 0
        for client in self.selected_clients:
            active_train_samples += client.train_samples

        self.uploaded_weights = []
        self.uploaded_model_gs = []
        for client in self.selected_clients:
            self.uploaded_weights.append(client.train_samples / active_train_samples)
            self.uploaded_model_gs.append(client.CoV)

        self.CoV = copy.deepcopy(self.uploaded_model_gs[0])
        for param in self.CoV.parameters():
            param.data = torch.zeros_like(param.data)

        for w, client_model in zip(self.uploaded_weights, self.uploaded_model_gs):
            self.add_CoV(w, client_model)

        for client in self.clients:
            client.set_CoV(self.CoV)

    def add_CoV(self, w, CoV):
        for server_param, client_param in zip(self.CoV.parameters(), CoV.parameters()):
            server_param.data += client_param.data.clone() * w


class GCE(nn.Module):
    def __init__(self, in_features, num_classes, dev='cpu'):
        super(GCE, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.embedding = nn.Embedding(num_classes, in_features)
        self.dev = dev

    def forward(self, x, label):
        embeddings = self.embedding(torch.tensor(range(self.num_classes), device=self.dev))
        cosine = F.linear(F.normalize(x), F.normalize(embeddings))
        one_hot = torch.zeros(cosine.size(), device=self.dev)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        softmax_value = F.log_softmax(cosine, dim=1)
        softmax_loss = one_hot * softmax_value
        softmax_loss = - torch.mean(torch.sum(softmax_loss, dim=1))

        return softmax_loss


class CoV(nn.Module):
    def __init__(self, in_dim):
        super(CoV, self).__init__()

        self.Conditional_gamma = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.LayerNorm([in_dim]),
        )
        self.Conditional_beta = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.LayerNorm([in_dim]),
        )
        self.act = nn.ReLU()

    def forward(self, x, context):
        gamma = self.Conditional_gamma(context)
        beta = self.Conditional_beta(context)

        out = torch.multiply(x, gamma + 1)
        out = torch.add(out, beta)
        out = self.act(out)
        return out




