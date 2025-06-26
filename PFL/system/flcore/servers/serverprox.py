import time

import numpy as np
import logging

from flcore.clients.clientprox import clientProx
from flcore.servers.serverbase import Server
import logging


class FedProx(Server):
    def __init__(self, config, times, init_model):
        super().__init__(config, times, init_model)

        # select slow clients
        self.set_clients(config, clientProx)


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
                client.train()
            t_2 = time.time()
            avg_time = (t_2 - t_1) / self.num_join_clients
            logging.info(avg_time*1000)
            avg_list.append(avg_time * 1000)
            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            self.aggregate_parameters()

            # modification
            if (i + 1) % self.check_step == 0:
                self.save_check_model(i + 1)

        logging.info(f'mean time:{np.mean(np.array(avg_list))}')
        logging.info(f'var time:{np.mean(np.std(avg_list))}')
        logging.info("\nBest global accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        logging.info(max(self.rs_test_acc))

        self.evaluate_corruption()
        self.save_results()
        self.save_global_model()
