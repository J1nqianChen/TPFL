import copy
import time

import numpy as np
import logging
import torch
import torch.nn.functional as F
from tqdm import tqdm

from flcore.clients.clientdyn import clientDyn
from flcore.servers.serverbase import Server
import logging


class FedDyn(Server):
    def __init__(self, config, times, init_model):
        super().__init__(config, times, init_model)

        # select slow clients
        self.set_clients(config, clientDyn)

        logging.info(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        logging.info("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

        self.alpha = config.algo_hyperparam.FedDyn.alpha.value

        self.server_state = copy.deepcopy(init_model)
        for param in self.server_state.parameters():
            param.data = torch.zeros_like(param.data)



    def train(self):
        for i in range(self.global_rounds + 1):
            self.curr_round = i
            s_t = time.time()
            self.selected_clients = self.select_clients()
            if len(self.selected_clients) == 0:
                continue
            self.send_models()

            if i % self.eval_gap == 0:
                logging.info(f"\n-------------Round number: {i}-------------")
                logging.info("\nEvaluate global model")
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
            self.update_server_state()
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            logging.info('-' * 50 + str(self.Budget[-1]))

            # if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
            #     break

            if (i+1) % self.check_step == 0:
                self.save_check_model(i)


        logging.info("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        logging.info(max(self.rs_test_acc))
        logging.info("\nBest local accuracy.")
        logging.info("\nAveraged time per iteration.")
        logging.info(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        # if self.num_new_clients > 0:
        #     self.eval_new_clients = True
        #     self.set_new_clients(clientDyn)
        #     logging.info(f"\n-------------Fine tuning round-------------")
        #     logging.info("\nEvaluate new clients")
        #     self.evaluate()

    def add_parameters(self, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() / self.num_join_clients

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data = torch.zeros_like(param.data)

        for client_model in self.uploaded_models:
            self.add_parameters(client_model)

        for server_param, state_param in zip(self.global_model.parameters(), self.server_state.parameters()):
            server_param.data -= (1 / self.alpha) * state_param

    def update_server_state(self):
        assert (len(self.uploaded_models) > 0)

        model_delta = copy.deepcopy(self.uploaded_models[0])
        for param in model_delta.parameters():
            param.data = torch.zeros_like(param.data)

        for client_model in self.uploaded_models:
            for server_param, client_param, delta_param in zip(self.global_model.parameters(),
                                                               client_model.parameters(), model_delta.parameters()):
                delta_param.data += (client_param - server_param) / self.num_clients

        for state_param, delta_param in zip(self.server_state.parameters(), model_delta.parameters()):
            state_param.data -= self.alpha * delta_param

    # def evaluate_t(self, acc=None):
    #     count_arr = np.zeros([self.num_classes])
    #     corr_arr = np.zeros([self.num_classes])
    #     self.global_model.eval()
    #     for x, y in tqdm(self.total_test_loader):
    #         x = x.cuda()
    #         y = y.cuda()
    #         output = self.global_model(x)
    #         predictions = F.softmax(output, dim=1)
    #
    #         _, pred_labels = torch.max(predictions, 1)
    #         for j in range(self.num_classes):
    #             mask = y == j
    #             specify_y = y[mask]
    #             pred_labels_mask = pred_labels[mask]
    #             count_arr[j] += len(specify_y.tolist())
    #             corr_arr[j] += torch.sum(torch.eq(pred_labels_mask, specify_y)).item()
    #
    #     test_acc = corr_arr.sum() / count_arr.sum()
    #     if acc is None:
    #         self.rs_test_acc.append(test_acc)
    #     else:
    #         acc.append(test_acc)
    #
    #     logging.info(f'Global Acc: {test_acc}')