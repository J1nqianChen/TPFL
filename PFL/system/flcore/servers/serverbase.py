import copy
import os
import logging
import pickle
import time
from pympler import asizeof
import typing
import h5py
import numpy as np
import logging
import torch
from torch.utils.data import DataLoader
from flcore.aggregator.krum import krum_aggregate, median_aggregate, multikrum_aggregate, trimmed_mean_aggregate, \
    norm_clipping_aggregate
from utils.data_utils import read_client_data, select_test_read, get_model_size_in_mb
from utils.dlg import DLG
from utils.pr_utils import get_pr_table


class Server(object):

    def __init__(self, config, times, init_model):
        # Set up the main attributes
        self._snapshot = None
        self.rs_val_acc = []
        self.record_save_path = None
        self.check_step = config.general.check_step
        self.eval_gap = config.general.eval_gap
        self.device = config.general.device
        self.dataset = config.train.dataset_name
        self.global_rounds = config.train.global_rounds
        self.local_steps = config.train.local_epochs
        self.batch_size = config.train.batch_size
        self.learning_rate = config.general.local_learning_rate
        self.global_model = copy.deepcopy(init_model)
        self.num_clients = config.train.num_clients
        self.join_ratio = config.train.participation_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.curr_round = 0
        self.algorithm = config.train.algorithm
        self.goal = config.train.goal
        self.num_classes = config.train.num_classes
        self.clients = []
        self.selected_clients = []
        self.train_slow_clients = []
        self.send_slow_clients = []
        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []
        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_train_loss = []
        self.best_round = []
        self.times = times
        self.best_val_acc = 0
        self.best_test_acc = 0
        self.best_model = None
        self.participate_clients = []

        # 恶意攻击模块
        self.attack_flag = config.attack.flag
        self.attack_ratio = config.attack.ratio
        if not self.attack_flag:
            logging.info('无恶意攻击')
            self.malicious_id_list = []
            if self.attack_ratio > 0:
                raise RuntimeError('非恶意攻击模式，但攻击率大于0。')
        else:
            self.malicious_id_list = np.random.choice(np.arange(self.num_clients),
                                                      int(self.num_clients * self.attack_ratio), replace=False)
            logging.info(f'Malicious Number: {self.malicious_id_list.shape[0]}')
            logging.info(f'Malicious ID LIST: {self.malicious_id_list}')
            self.attack_strategy = config.attack.name
            self.attack_round = config.attack.round
            self.mean_para_benign = None
            self.var_para_benign = None
            if self.attack_strategy == 'MPAF':
                self.fake_frac = config.attack.algorithm_param.MPAF.fake_frac
            elif self.attack_strategy == 'STAT_OPT':
                self.fake_frac = self.attack_ratio
                self.attack_ratio = 0
                self.malicious_id_list = np.arange(int(self.fake_frac * self.num_clients)) + self.num_clients
                logging.info('STAT OPT, Reload Fake Clients')
                logging.info(f'Malicious Number: {self.malicious_id_list.shape[0]}')
                logging.info(f'Malicious ID LIST: {self.malicious_id_list}')
        self.defense_flag = config.defense.flag
        self.defense_name = config.defense.name
        self.last_round = -1
        if self.defense_flag:
            logging.info('防御启动')
            logging.info(f'防御策略:{self.defense_name}')
        else:
            logging.info('防御关闭')

        # 检查有效性，避免部分客户端样本数量<batch_size，导致dataloader中drop_last时造成空客户端。
        train_sample_list = []
        id_list = []
        for c_id in range(self.num_clients):
            train_data = read_client_data(self.dataset, c_id, is_train=True, times=self.times)
            train_sample_list.append(len(train_data))
            id_list.append(c_id)
        mask = np.array(train_sample_list) > self.batch_size
        masked_id = np.array(id_list)[mask]
        valid_number = masked_id.shape[0]
        logging.info(f'valid client number: {valid_number}')
        if valid_number != self.num_clients:
            raise RuntimeError('存在无效客户端，请检查数据分割。')

        self.communication_cost = 0
        self.save_path = config.general.save_path

        try:
            self.save_pkl = config.train.save_pkl
        except:
            self.save_pkl = True
        self.mk_exp_dirs()
        try:
            self.pr_mode = config.train.participation_mode
        except AttributeError:
            self.pr_mode = 'static'
        logging.info(f'当前采样模式：{self.pr_mode}')
        if self.pr_mode == 'cyclic':
            self.cyclic_val = config.train.cyclic.T
            self.pr_tab = get_pr_table(method=self.pr_mode, num_clients=self.num_clients,
                                       total_round=self.global_rounds + 2, pr=self.join_ratio, cycle=self.cyclic_val)
        elif self.pr_mode == 'markov':
            self.trans_prob = config.train.markov.prob
            self.pr_tab = get_pr_table(method=self.pr_mode, num_clients=self.num_clients,
                                       total_round=self.global_rounds + 2, pr=self.join_ratio,
                                       trans_prob=self.trans_prob)

    def mk_exp_dirs(self):
        base_path = os.path.join(self.save_path, f'Time_{self.times}')
        if not os.path.exists(base_path):
            logging.info('初始化实验保存目录')
            os.makedirs(base_path)
        self.save_path = base_path
        logging.info(f'保存路径为:{base_path}')

    def get_full_test_data(self):
        _, test_dataset, _ = select_test_read(self.dataset, select_list=[i for i in range(self.num_clients)],
                                              )
        test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
        return test_loader

    def set_clients(self, config, clientObj):
        for i in range(self.num_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False, times=self.times)
            client = clientObj(config,
                               id=i,
                               train_samples=len(train_data),
                               test_samples=len(test_data),
                               init_model=copy.deepcopy(self.global_model))
            self.clients.append(client)
            drop_len = len(train_data) % self.batch_size
            logging.info(f'client {i} drop {drop_len} samples')
        if self.attack_flag:
            logging.info(f'attack strategy: {self.attack_strategy}')
            for m_id in self.malicious_id_list:
                if m_id >= self.num_clients:
                    logging.info('Fake Malicious')
                    if self.attack_strategy not in ['STAT_OPT']:
                        raise RuntimeError('Check Attack')
                    continue
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

                    # with model replacement boost

        for c in self.clients:
            c.set_times(self.times)

    def select_clients(self):
        if self.pr_mode == 'static':
            selected_clients = list(np.random.choice(self.clients, self.num_join_clients, replace=False))
        else:
            if self.curr_round == self.last_round:
                raise RuntimeError('Round未正确赋值')
            self.last_round = self.curr_round
            curr_mask = list(self.pr_tab[:, self.curr_round])
            selected_clients = [client for client, c_mask in zip(self.clients, curr_mask) if c_mask]
            if len(selected_clients) == 0:
                logging.info('Empty Round')
        return selected_clients

    def send_models(self):
        for client in self.selected_clients:
            client.set_parameters(self.global_model)
            self.communication_cost = self.communication_cost + get_model_size_in_mb(self.global_model)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)
        self.uploaded_weights = []
        tot_samples = 0
        self.uploaded_ids = []
        self.uploaded_models = []
        for client in self.selected_clients:
            self.uploaded_weights.append(client.train_samples)
            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_models.append(client.model)
            self.communication_cost += get_model_size_in_mb(client.model)
            # if not client.is_malicious:
            #     self.uploaded_weights.append(client.train_samples)
            #     tot_samples += client.train_samples
            #     self.uploaded_ids.append(client.id)
            #     self.uploaded_models.append(client.model)
            # else:
            #     if client.curr_round not in client.attack_round and -1 not in client.attack_round:
            #         self.uploaded_weights.append(client.train_samples)
            #         tot_samples += client.train_samples
            #         self.uploaded_ids.append(client.id)
            #         self.uploaded_models.append(client.model)
            #     else:
            #
            #         # self.uploaded_weights.append(client.train_samples * self.num_clients)
            #         # tot_samples += client.train_samples * self.num_clients
            #         # self.uploaded_ids.append(client.id)
            #         # self.uploaded_models.append(client.model)
            #         logging.info(f'{client.id} Start Model Replacement Attack')
            #         self.uploaded_weights.append(client.train_samples)
            #         tot_samples += client.train_samples
            #         self.uploaded_ids.append(client.id)
            #
            #         for param_client, param_global in zip(client.model.parameters(), self.global_model.parameters()):
            #             param_client.data = self.num_clients * param_client.data - (
            #                     self.num_clients - 1) * param_global.data
            #
            #         self.uploaded_models.append(client.model)
            #
            #         if client.attack_strategy == 'random_weight':
            #             logging.info(f'client {client.id} conduct random weight attack')
            #             temp_model = client.model
            #             with torch.no_grad():
            #                 for param in temp_model.parameters():
            #                     param.data = torch.randn(param.size()).cuda()

        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):
        if not self.defense_flag:
            assert (len(self.uploaded_models) > 0)

            self.global_model = copy.deepcopy(self.uploaded_models[0])
            for param in self.global_model.parameters():
                param.data.zero_()

            for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
                self.add_parameters(w, client_model)
        # 模拟各类攻击
        else:
            logging.info('!' * 50)
            logging.info('CAUTION: 请仔细检查代码是否符合当前算法的聚合要求，这是基础FedAvg-Based聚合')
            malicious_num = len(self.malicious_id_list.tolist())
            if self.attack_strategy == 'MPAF':
                malicious_num = int(self.fake_frac * self.num_clients)

            if self.defense_name == 'krum':
                logging.info('Robust Aggregate via Krum')
                agg_model = krum_aggregate(uploaded_models=self.uploaded_models, m=malicious_num)
                self.global_model = copy.deepcopy(agg_model)
            elif self.defense_name == 'multikrum':
                logging.info('Robust Aggregate via MultiKrum')
                agg_model = multikrum_aggregate(uploaded_models=self.uploaded_models,
                                                uploaded_weight=self.uploaded_weights,
                                                m=len(self.malicious_id_list.tolist()))
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
                agg_model = norm_clipping_aggregate(uploaded_models=self.uploaded_models,
                                                    uploaded_weights=self.uploaded_weights)
                self.global_model = copy.deepcopy(agg_model)
            else:
                raise NotImplementedError

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def save_check_model(self, i):
        model_path = os.path.join(self.save_path, 'global_models')

        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, f'round_{i}.pt')
        # model_state_dict = self.global_model.state_dict()
        # torch.save(model_state_dict, model_path)
        torch.save(self.global_model, model_path)

    def save_global_model(self):
        self.save_snapshot_to_pickle()
        # model_path = os.path.join(self.save_path, 'global_models')
        # if not os.path.exists(model_path):
        #     os.makedirs(model_path)
        #
        # model_path = os.path.join(model_path, 'final.pt')
        # # model_state_dict = self.global_model.state_dict()
        # # torch.save(model_state_dict, model_path)
        # torch.save(self.global_model, model_path)

    def save_best_model(self):
        if self.save_pkl:
            self.save_snapshot_to_pickle()
        else:
            logging.info('No Save Setting')
            return
        # model_path = os.path.join(self.save_path, 'global_models')
        # if not os.path.exists(model_path):
        #     os.makedirs(model_path)
        # model_path = os.path.join(model_path, 'best.pt')
        # # model_state_dict = self.global_model.state_dict()
        # # torch.save(model_state_dict, model_path)
        # torch.save(self.best_model, model_path)

    # def load_model(self):
    #     model_path = os.path.join("models", self.dataset)
    #     model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
    #     assert (os.path.exists(model_path))
    #     self.global_model = torch.load(model_path)
    #
    # def model_exists(self):
    #     model_path = os.path.join("models", self.dataset)
    #     model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
    #     return os.path.exists(model_path)

    def save_results(self):
        result_path = os.path.join(self.save_path, 'record')
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if len(self.rs_test_acc):
            file_path = os.path.join(result_path, 'record.h5')
            self.record_save_path = file_path
            logging.info("Save record path: " + file_path)

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_val_acc', data=self.rs_val_acc)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)

    # Unused
    # def save_item(self, item, item_name):
    #     if not os.path.exists(self.save_folder_name):
    #         os.makedirs(self.save_folder_name)
    #     torch.save(item, os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))
    #
    # def load_item(self, item_name):
    #     return torch.load(os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def test_metrics(self, backdoor_evaluate=False, use_val=False):
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.selected_clients:
            if self.attack_flag and c.id in self.malicious_id_list.tolist():
                continue
            ct, ns, auc = c.test_metrics(backdoor_evaluate=backdoor_evaluate, use_val=use_val)
            tot_correct.append(ct * 1.0)
            tot_auc.append(auc * ns)
            num_samples.append(ns)

        ids = [c.id for c in self.selected_clients]

        return ids, num_samples, tot_correct, tot_auc

    def train_metrics(self):
        num_samples = []
        losses = []
        for c in self.selected_clients:
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl * 1.0)

        ids = [c.id for c in self.selected_clients]

        return ids, num_samples, losses

    # test metrics on all clients, train metrics on selected clients.
    def evaluate(self, acc=None, loss=None):
        chosen_clients = copy.copy(self.selected_clients)
        self.selected_clients = copy.copy(self.clients)
        self.send_models()
        stats_test = self.test_metrics()
        test_acc = sum(stats_test[2]) * 1.0 / sum(stats_test[1])
        stats_val = self.test_metrics(use_val=True)
        val_acc = sum(stats_val[2]) * 1.0 / sum(stats_val[1])
        self.selected_clients = copy.copy(chosen_clients)
        stats_train = self.train_metrics()

        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_test_acc = test_acc
            self.best_round = self.curr_round
            self.save_snapshot()

            # self.best_model = copy.deepcopy(self.global_model)
            time_save_before = time.time()
            # dill.

        # Unimplemented Function
        # if test_acc > self.early_stop_acc:
        #     logging.info(f'Current Communication Cost:{self.communication_cost}')
        #     exit(1)
        train_loss = sum(stats_train[2]) * 1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats_test[2], stats_test[1])]
        aucs = [a / n for a, n in zip(stats_test[3], stats_test[1])]

        if acc is None:
            self.rs_val_acc.append(val_acc)
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(val_acc)

        if loss is None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        logging.info("Averaged Train Loss: {:.4f}".format(train_loss))
        logging.info("Averaged Validation Accurancy: {:.4f}".format(val_acc))
        logging.info("Averaged Test Accurancy: {:.4f}".format(test_acc))
        # self.logging.info_(test_acc, train_acc, train_loss)
        logging.info("Std Test Accurancy: {:.4f}".format(np.std(accs)))
        logging.info("Std Test AUC: {:.4f}".format(np.std(aucs)))

        if self.attack_flag:
            if self.attack_strategy == 'backdoor_pattern':
                stats_backdoor = self.test_metrics(backdoor_evaluate=True)
                test_backdoor_acc = sum(stats_backdoor[2]) * 1.0 / sum(stats_backdoor[1])
                logging.info("Backdoor Test Accurancy: {:.4f}".format(test_backdoor_acc))

    def check_early_stop(self):
        logging.info('检查是否早停')
        if len(self.rs_test_acc) > 5:
            max_test_acc = max(self.rs_test_acc[-5:])
            min_test_acc = min(self.rs_test_acc[-5:])
            if abs(min_test_acc - max_test_acc) < 1e-4:
                return True
            else:
                return False
        else:
            return False

    def print_(self, test_acc, test_auc, train_loss):
        logging.info("Average Test Accurancy: {:.4f}".format(test_acc))
        logging.info("Average Test AUC: {:.4f}".format(test_auc))
        logging.info("Average Train Loss: {:.4f}".format(train_loss))

    def check_done(self, acc_lss, top_cnt=None, div_value=None):
        for acc_ls in acc_lss:
            if top_cnt != None and div_value != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_top and find_div:
                    pass
                else:
                    return False
            elif top_cnt != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                if find_top:
                    pass
                else:
                    return False
            elif div_value != None:
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_div:
                    pass
                else:
                    return False
            else:
                raise NotImplementedError
        return True

    def call_dlg(self, R):
        # items = []
        cnt = 0
        psnr_val = 0
        for cid, client_model in zip(self.uploaded_ids, self.uploaded_models):
            client_model.eval()
            origin_grad = []
            for gp, pp in zip(self.global_model.parameters(), client_model.parameters()):
                origin_grad.append(gp.data - pp.data)

            target_inputs = []
            trainloader = self.clients[cid].load_train_data()
            with torch.no_grad():
                for i, (x, y) in enumerate(trainloader):
                    if i >= self.batch_num_per_client:
                        break

                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    output = client_model(x)
                    target_inputs.append((x, output))

            d = DLG(client_model, origin_grad, target_inputs)
            if d is not None:
                psnr_val += d
                cnt += 1

            # items.append((client_model, origin_grad, target_inputs))

        if cnt > 0:
            logging.info('PSNR value is {:.2f} dB'.format(psnr_val / cnt))
        else:
            logging.info('PSNR error')

        # self.save_item(items, f'DLG_{R}')

    def check_corruption_data_exist(self):
        corruption_data = self.dataset + '_C'
        current_dir = os.path.abspath(os.path.dirname(__file__))
        parts = current_dir.split(os.sep)
        target_parts = parts[:-3]
        dataset_path = os.path.join('/', *target_parts, 'dataset')
        dataset_path = os.path.join(dataset_path, corruption_data)
        if os.path.exists(dataset_path):
            return True
        else:
            return False

    def evaluate_corruption(self):
        if not self.check_corruption_data_exist():
            logging.info('No corruption data exists')
            return -1
        # result = np.zeros(5)        # 5 severity levels
        list_acc_arr = []
        list_sample_num = []
        sum_sample = 0
        for c in self.clients:
            acc_arr, num = c.evaluate_corruption()
            list_acc_arr.append(acc_arr)
            list_sample_num.append(num)
        total_acc_arr = np.stack(list_acc_arr, axis=0)  # N, 5
        total_sample_num = np.array(list_sample_num)
        weight = total_sample_num / np.sum(total_sample_num)
        logging.info(weight.shape)
        total_acc_arr = total_acc_arr * weight[:, np.newaxis]
        result_arr = np.sum(total_acc_arr, axis=0)
        logging.info('Robustness on corruption result:')
        logging.info(result_arr)
        return result_arr


    def save_snapshot(self):
        """保存当前实例状态的快照。"""
        # 深拷贝所有属性，排除存储快照的变量_snapshot
        if not self.save_pkl:
            return
        time_1_ = time.time()
        self._snapshot = {}
        for k, v in self.__dict__.items():
            ignore_list = ['_snapshot', 'selected_clients', 'uploaded_weights', 'uploaded_models', 'train_slow_clients',
                           'send_slow_clients', 'participate_clients']
            if k not in ignore_list:
                if k == 'clients':
                    # continue
                    logging.info(f'正在处理:{k}')
                    list_clients = []
                    for c in v:
                        # logging.info('------')
                        # for k_c, v_c in c.__dict__.items():
                        #     logging.info(f'client中保存了：{k_c}')
                        duplicate_c = copy.deepcopy(c)
                        duplicate_c.model.zero_grad()
                        duplicate_c.model.cpu()
                        duplicate_c.optimizer.zero_grad()
                        duplicate_c.flipped_data = None
                        list_clients.append(duplicate_c)
                        # logging.info('------')
                    self._snapshot[k] = list_clients
                else:
                    # logging.info(f'正在保存：{k}')
                    self._snapshot[k] = copy.deepcopy(v)

        used_time = time.time() - time_1_
        logging.info(f'保存快照成功，耗时{used_time}')

    def restore_snapshot(self):
        """从快照恢复属性"""
        if not self._snapshot:
            raise ValueError("No snapshot to restore")
        for key, value in self._snapshot.items():
            setattr(self, key, copy.deepcopy(value))
        logging.info('成功从快照恢复属性')
        for c in self.clients:
            c.model.to(self.device)
    def save_snapshot_to_pickle(self):
        """仅保存快照数据，避免冗余"""
        if not self._snapshot:
            raise ValueError("No snapshot to save")
        file_name = os.path.join(self.save_path, "best_snapshot.pkl")
        with open(file_name, 'wb') as f:
            pickle.dump(self._snapshot, f)
        logging.info('成功保存快照到pkl')

    def restore_from_pickle(self, filename):
        """从文件加载快照并应用到当前实例"""
        with open(filename, 'rb') as f:
            self._snapshot = pickle.load(f)
        self.restore_snapshot()
        logging.info('成功从pkl恢复快照')


def format_size(size_bytes):
    units = ('B', 'KB', 'MB', 'GB')
    unit_idx = 0
    while size_bytes >= 1024 and unit_idx < 3:
        size_bytes /= 1024
        unit_idx += 1
    return f"{size_bytes:.2f} {units[unit_idx]}"
