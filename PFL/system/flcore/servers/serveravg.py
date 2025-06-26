import copy
import logging
import time
import torch
import numpy as np
import logging

from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
import logging

from utils.data_utils import get_model_size_in_mb, flatten_params, get_malicious_updates_fang_trmean, reload_params, \
    compute_mean_var_of_benign_updates


class FedAvg(Server):
    def __init__(self, args, times, init_model):
        super().__init__(args, times, init_model)

        # select slow clients
        self.set_clients(args, clientAVG)

        logging.info(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")

        logging.info("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

    def train(self):
        avg_list = []
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
                if i == self.global_rounds:
                    self.evaluate_corruption()

            t_1 = time.time()
            for client in self.selected_clients:
                client.set_curr_round(i)
                client.train()
            t_2 = time.time()
            avg_time = (t_2 - t_1) / self.num_join_clients
            avg_list.append(avg_time)
            logging.info(f'Client Avg Time: {avg_time * 1000} ')
            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            logging.info('-' * 25 + 'time cost' + '-' * 25 + str(self.Budget[-1]))

            # modification
            if (i + 1) % self.check_step == 0:
                self.save_check_model(i)

        logging.info(f'mean time:{np.mean(np.array(avg_list))}')
        logging.info(f'var time:{np.mean(np.std(avg_list))}')
        logging.info("\nBest global accuracy.")
        # self.logging.info_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        logging.info(max(self.rs_test_acc))
        logging.info("\nAverage time cost per round.")
        logging.info(sum(self.Budget[1:]) / len(self.Budget[1:]))


        self.save_results()
        self.save_best_model()


    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        self.uploaded_weights = []
        tot_samples = 0
        self.uploaded_ids = []
        self.uploaded_models = []

        benign_update = []
        for client in self.selected_clients:
            self.uploaded_weights.append(client.train_samples)
            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_models.append(client.model)
            if not client.is_malicious:
                benign_update.append(client.model)
            self.communication_cost += get_model_size_in_mb(client.model)

        if self.attack_flag and self.attack_strategy == 'STAT_OPT':

            attack_num = int(self.fake_frac * self.num_clients)
            logging.info('STAT_OPT Attack')
            benign_gradients = []
            old_param = flatten_params(self.global_model)
            for model in self.uploaded_models:
                new_param = flatten_params(model)
                benign_gradients.append(new_param - old_param)
            user_grads = torch.stack(benign_gradients, dim=0)
            agg_grads = torch.mean(user_grads, 0)
            deviation = torch.sign(agg_grads)
            malicious_grads = get_malicious_updates_fang_trmean(user_grads, deviation, attack_num)
            bad_params = old_param + malicious_grads

            for k in range(attack_num):
                bad_model = copy.deepcopy(self.uploaded_models[-1])
                bad_param = bad_params[k, :]
                reload_params(bad_model, bad_param)
                self.uploaded_models.append(bad_model)
                self.uploaded_ids.append(k + self.num_clients)
                self.uploaded_weights.append(self.uploaded_weights[-1])

        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

        if self.attack_flag:
            if self.attack_strategy in ['LIE', 'Gaussian']:
                logging.info('Computing Benign Params')
                with torch.no_grad():
                    mean_para, var_para = compute_mean_var_of_benign_updates(benign_update)
                self.mean_para_benign = mean_para
                self.var_para_benign = var_para
                for client in self.selected_clients:
                    client.set_benign_mean_var(self.mean_para_benign, self.var_para_benign)

