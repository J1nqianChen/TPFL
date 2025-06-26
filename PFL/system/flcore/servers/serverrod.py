import copy
import time

import numpy as np
import logging

from flcore.aggregator.krum import krum_aggregate, multikrum_aggregate, median_aggregate, trimmed_mean_aggregate, \
    norm_clipping_aggregate
from flcore.clients.clientrod import clientROD
from flcore.servers.serverbase import Server
import logging


class FedROD(Server):
    def __init__(self, config, times, init_model):
        super().__init__(config, times, init_model)

        self.set_clients(config, clientROD)

        logging.info(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        logging.info("Finished creating server and clients.")

        # self.load_model()


    def train(self):
        avg_list = []
        for i in range(self.global_rounds+1):
            self.curr_round = i
            self.selected_clients = self.select_clients()
            if len(self.selected_clients) == 0:
                continue
            self.send_models()

            if i%self.eval_gap == 0:
                logging.info(f"\n-------------Round number: {i}-------------")
                logging.info("\nEvaluate global model")
                self.evaluate()
                if self.check_early_stop():
                    logging.info('五轮精度波动小于0.0001，退出训练')
                    break
            t_1 = time.time()
            for client in self.selected_clients:
                client.set_curr_round(i)
                client.train()
            t_2 = time.time()
            avg_time = (t_2 - t_1) / self.num_join_clients
            avg_list.append(avg_time * 1000)
            logging.info(f'Client Avg Time: {avg_time * 1000} ')


            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            self.aggregate_parameters()

        logging.info(f'mean time:{np.mean(np.array(avg_list))}')
        logging.info(f'var time:{np.mean(np.std(avg_list))}')
        logging.info("\nBest global accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        logging.info(max(self.rs_test_acc))

        self.evaluate_corruption()
        self.save_results()
        self.save_best_model()
        # self.save_personalized_best()

    def aggregate_parameters(self):
        if not self.defense_flag:
            assert (len(self.uploaded_models) > 0)

            self.global_model = copy.deepcopy(self.uploaded_models[0])
            for param in self.global_model.parameters():
                param.data.zero_()

            for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
                self.add_parameters(w, client_model)
        else:
            malicious_num = len(self.malicious_id_list.tolist())
            if self.defense_name == 'krum':
                logging.info('Robust Aggregate via Krum')
                agg_model = krum_aggregate(uploaded_models=self.uploaded_models, m=malicious_num)
                self.global_model = copy.deepcopy(agg_model)
            elif self.defense_name == 'multikrum':
                logging.info('Robust Aggregate via MultiKrum')
                agg_model = multikrum_aggregate(uploaded_models=self.uploaded_models, uploaded_weight=self.uploaded_weights, m=len(self.malicious_id_list.tolist()))
                self.global_model = copy.deepcopy(agg_model)
            elif self.defense_name == 'median':
                logging.info('Robust Aggregate via Median')
                agg_model = median_aggregate(uploaded_models=self.uploaded_models, m=malicious_num)
                self.global_model = copy.deepcopy(agg_model)
            elif self.defense_name == 'trimmed_mean':
                logging.info('Robust Aggregate via TrimmedMean')
                agg_model = trimmed_mean_aggregate(uploaded_models=self.uploaded_models, uploaded_id=self.uploaded_ids,
                                                   m=malicious_num)
                self.global_model = copy.deepcopy(agg_model)
            elif self.defense_name == 'norm_clipping':
                logging.info('Robust Aggregate via Norm Clipping')
                agg_model = norm_clipping_aggregate(uploaded_models=self.uploaded_models, uploaded_weights=self.uploaded_weights)
                self.global_model = copy.deepcopy(agg_model)
