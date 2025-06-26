import logging
import time
import copy
import numpy as np
import logging
from flcore.clients.clientas import clientAS
from flcore.servers.serverbase import Server
import logging


class FedAS(Server):
    def __init__(self, config, times, init_model):
        super().__init__(config, times, init_model)

        # select slow clients
        self.set_clients(config, clientAS)

        logging.info(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        logging.info("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

    def all_clients(self):
        return self.clients

    def send_selected_models(self, selected_ids, epoch):
        assert (len(self.clients) > 0)

        # for client in self.clients:
        for client in [client for client in self.clients if (client.id in selected_ids)]:
            start_time = time.time()

            progress = epoch / self.global_rounds

            # client.set_parameters(self.global_model, progress)
            client.set_parameters(self.global_model)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def aggregate_wrt_fisher(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()

        # calculate the aggregrate weight with respect to the FIM value of model
        FIM_weight_list = []
        for id in self.uploaded_ids:
            FIM_weight_list.append(self.clients[id].fim_trace_history[-1])
        # normalization to obtain weight
        FIM_weight_list = [FIM_value / sum(FIM_weight_list) for FIM_value in FIM_weight_list]

        for w, client_model in zip(FIM_weight_list, self.uploaded_models):
            self.add_parameters(w, client_model)

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.curr_round = i
            self.selected_clients = self.select_clients()
            if len(self.selected_clients) == 0:
                continue
            self.alled_clients = self.all_clients()

            selected_ids = [client.id for client in self.selected_clients]

            # self.send_models()

            # evaluate personalized models, ie FedAvg-C
            if i % self.eval_gap == 0:
                logging.info(f"\n-------------Round number: {i}-------------")
                logging.info("\nEvaluate global model")
                self.evaluate()
                if self.check_early_stop():
                    logging.info('五轮精度波动小于0.0001，退出训练')
                    break
                if i == self.global_rounds:
                    self.evaluate_corruption()

            # self.send_models()
            self.send_selected_models(selected_ids, i)

            # logging.info(f'send selected models done')

            # for client in self.selected_clients:
            #     client.train()

            for client in self.alled_clients:
                # logging.info("===============")
                client.train(client.id in selected_ids)
            # assert 1==0

            self.print_fim_histories()

            self.receive_models()

            self.aggregate_wrt_fisher()

            self.Budget.append(time.time() - s_t)
            logging.info('-' * 25 + 'time cost' + '-' * 25 + str(self.Budget[-1]))

        logging.info("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        logging.info(max(self.rs_test_acc))
        logging.info("\nAverage time cost per round.")
        logging.info(sum(self.Budget[1:]) / len(self.Budget[1:]))

        # logging.info(f'+++++++++++++++++++++++++++++++++++++++++')
        # gen_acc = self.avg_generalization_metrics()
        # logging.info(f'Generalization Acc: {gen_acc}')
        # logging.info(f'+++++++++++++++++++++++++++++++++++++++++')

        self.save_results()
        self.save_global_model()

        # if self.num_new_clients > 0:
        #     self.eval_new_clients = True
        #     self.set_new_clients(clientAS)
        #     logging.info(f"\n-------------Fine tuning round-------------")
        #     logging.info("\nEvaluate new clients")
        #     self.evaluate()

    def print_fim_histories(self):
        avg_fim_histories = []

        # Print FIM trace history for each client
        # for client in self.selected_clients:
        for client in self.alled_clients:
            formatted_history = [f"{value:.1f}" for value in client.fim_trace_history]
            logging.info(f"Client{client.id} : {formatted_history}")
            avg_fim_histories.append(client.fim_trace_history)

        # Calculate and print average FIM trace history across clients
        avg_fim_histories = np.mean(avg_fim_histories, axis=0)
        formatted_avg = [f"{value:.1f}" for value in avg_fim_histories]
        logging.info(f"Avg Sum_T_FIM : {formatted_avg}")