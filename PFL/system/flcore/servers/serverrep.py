import time

from flcore.clients.clientrep import clientRep
from flcore.servers.serverbase import Server
import logging
from utils.data_utils import get_model_size_in_mb


class FedRep(Server):
    def __init__(self, config, times, init_model):
        super().__init__(config, times, init_model)
        self.set_clients(config, clientRep)

        logging.info(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        logging.info("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

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
                logging.info("\nEvaluate personalized models")
                self.evaluate()
                if self.check_early_stop():
                    logging.info('五轮精度波动小于0.0001，退出训练')
                    break
            for client in self.selected_clients:
                client.set_curr_round(i)
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            self.aggregate_parameters()

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
        self.evaluate_corruption()

        # if self.num_new_clients > 0:
        #     self.eval_new_clients = True
        #     self.set_new_clients(clientRep)
        #     logging.info(f"\n-------------Fine tuning round-------------")
        #     logging.info("\nEvaluate new clients")
        #     self.evaluate()

    def receive_models(self):
        assert (len(self.selected_clients) > 0)
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in self.selected_clients:
            tot_samples += client.train_samples
            self.uploaded_weights.append(client.train_samples)
            self.uploaded_models.append(client.model.base)
            self.communication_cost += get_model_size_in_mb(client.model.base)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples