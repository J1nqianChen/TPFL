import copy
import os
import time

import h5py
import numpy as np
import logging
import torch
from torchvision import datasets
from torchvision.transforms import transforms

from flcore.clients.clientTPFL import clientTPFL
from flcore.optimizers.losses import kl_divergence
from flcore.servers.serverbase import Server
import logging
from threading import Thread
from tabulate import tabulate

from utils.data_utils import compute_mean_var_of_benign_updates, flatten_params, get_malicious_updates_fang_trmean, \
    reload_params, get_model_size_in_mb


class TPFL(Server):
    def __init__(self, config, times, init_model):
        super().__init__(config, times, init_model)

        # select slow clients
        self.set_clients(config, clientTPFL)

        logging.info(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        logging.info("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

        self.OOD_eval = config.algo_hyperparam.TPFL.ood_eval
        self.rs_generic_test = []
        self.model_uncertainty_list = []
        try:
            self.eval_key = config.algo_hyperparam.TPFL.eval_key
        except:
            self.eval_key = None
        if self.attack_flag:
            self.preserve_k = self.num_clients - self.malicious_id_list.shape[0]
        if self.attack_flag and self.attack_strategy in ['MPAF', 'STAT_OPT']:
            self.preserve_k = self.num_clients
            self.malicious_id_list = np.arange(self.num_clients,
                                               self.num_clients + int(self.fake_frac * self.num_clients))
            logging.info(self.malicious_id_list)
        self.init_model = copy.deepcopy(self.global_model)


        self.rs_test_p_acc = []
        self.rs_test_g_acc = []
        self.rs_val_p_acc = []
        self.rs_val_g_acc = []
        self.uncertainty_filtered_acc = []
        self.uploaded_sample_nums = []
        # logging.info(f'Malicious List Type: {type(self.malicious_id_list)}')

        try:
            self.reweight_u = config.algo_hyperparam.TPFL.reweight_u
        except:
            self.reweight_u = False

        try:
            self.holdout_data_path = config.algo_hyperparam.TPFL.holdout_data_path
            logging.info('Load Holdout Data path Success')
        except:
            self.holdout_data_path = None

        try:
            self.distinct_threshold = config.algo_hyperparam.TPFL.distinct_threshold
            logging.info('Load Distinct threshold Success')
        except:
            self.distinct_threshold = 1e-2


        self.true_positive_count = 0
        self.false_positive_count = 0
        self.false_negative_count = 0



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
            self.communication_cost += get_model_size_in_mb(client.model.base)

        if self.attack_flag and self.attack_strategy == 'MPAF':
            for count in range(int(self.fake_frac * self.num_clients)):
                self.uploaded_weights.append(self.uploaded_weights[-1])
                self.uploaded_ids.append(count + self.num_clients)
                self.uploaded_models.append(copy.deepcopy(self.init_model))

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

        self.uploaded_sample_nums = copy.copy(self.uploaded_weights)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

        if self.attack_flag:
            if self.attack_strategy in ['LIE', 'Gaussian']:
                with torch.no_grad():
                    mean_para, var_para = compute_mean_var_of_benign_updates(benign_update)
                self.mean_para_benign = mean_para
                self.var_para_benign = var_para
                for client in self.selected_clients:
                    client.set_benign_mean_var(self.mean_para_benign, self.var_para_benign)

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()

        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)

    def train(self):
        self.communication_cost = 0
        avg_list = []
        defense_list = []
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.curr_round = i
            for c in self.clients:
                c.set_curr_round(i)
            self.selected_clients = self.select_clients()
            if len(self.selected_clients) == 0:
                continue
            self.send_models()

            if i % self.eval_gap == 0:
                logging.info(f"\n-------------Round number: {i}-------------")
                logging.info("\nEvaluate global model")
                self.evaluate()

            t_1 = time.time()
            for client in self.selected_clients:
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
            t_3 = time.time()
            if self.attack_flag:
                self.estimate_model_uncertainty()
            t_4 = time.time()
            cost_defense = (t_4 - t_3) * 1000
            defense_list.append(cost_defense)
            logging.info(f'Defense Time: {cost_defense}')
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            logging.info('-' * 25 + 'time cost' + '-' * 25 + str(self.Budget[-1]))

            # modification
            if (i + 1) % self.check_step == 0:
                self.save_check_model(i)

        logging.info(f'mean time:{np.mean(np.array(avg_list))}')
        logging.info(f'var time:{np.mean(np.std(avg_list))}')
        logging.info(f'Defense mean time:{np.mean(np.array(defense_list))}')
        logging.info(f'Defense std time:{np.mean(np.std(defense_list))}')
        logging.info("\nBest Fused accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        logging.info(max(self.rs_test_acc))
        logging.info("\nAverage time cost per round.")
        logging.info(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.evaluate_corruption()
        self.save_results()
        self.save_best_model()

    def test_metrics(self, backdoor_evaluate=False, use_val=False):
        num_samples = []
        tot_correct = []
        tot_correct_p = []
        tot_correct_g = []

        num_samples_l = []
        tot_correct_l = []
        tot_correct_p_l = []
        tot_correct_g_l = []
        tot_auc = []
        th_c_total = None
        th_n_total = None
        for c in self.selected_clients:
            if self.attack_flag:
                if c.id in self.malicious_id_list.tolist():
                    logging.info(f'Skip Client {c.id}')
                    continue

            ct, ns, ctp, ctg, auc, ns_l, ct_l, ct_p_l, ct_g_l, th_c, th_n = c.test_metrics(use_val=use_val)

            if th_c_total is None:
                th_c_total = th_c
                th_n_total = th_n
            else:
                th_c_total = th_c_total + th_c
                th_n_total = th_n_total + th_n

            tot_correct.append(ct * 1.0)
            tot_correct_p.append(ctp * 1.0)
            tot_correct_g.append(ctg * 1.0)
            tot_auc.append(auc * ns)
            num_samples.append(ns)

            num_samples_l.append(ns_l)
            tot_correct_l.append(ct_l)
            tot_correct_p_l.append(ct_p_l)
            tot_correct_g_l.append(ct_g_l)

        ids = [c.id for c in self.selected_clients]

        return ids, num_samples, tot_correct, tot_correct_p, tot_correct_g, tot_auc, num_samples_l, tot_correct_l, tot_correct_p_l, tot_correct_g_l, th_c_total, th_n_total

    def evaluate(self, acc=None, loss=None):
        chosen_clients = copy.copy(self.selected_clients)
        self.selected_clients = copy.copy(self.clients)
        self.send_models()
        stats = self.test_metrics()
        stats_val = self.test_metrics(use_val=True)
        self.selected_clients = copy.copy(chosen_clients)

        stats_train = self.train_metrics()

        test_acc = sum(stats[2]) * 1.0 / sum(stats[1])
        test_acc_p = sum(stats[3]) * 1.0 / sum(stats[1])
        test_acc_g = sum(stats[4]) * 1.0 / sum(stats[1])

        val_acc = sum(stats_val[2]) * 1.0 / sum(stats_val[1])
        val_acc_p = sum(stats_val[3]) * 1.0 / sum(stats_val[1])
        val_acc_g = sum(stats_val[4]) * 1.0 / sum(stats_val[1])

        filtered_acc = stats[10] / stats[11]
        # if test_acc > self.early_stop_acc or test_acc_p > self.early_stop_acc:
        #     logging.info(f'Current Communication Cost:{self.communication_cost}')
        #     exit(1)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_test_acc = test_acc
            self.best_round = self.curr_round
            self.save_snapshot()

        train_loss = sum(stats_train[2]) * 1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        accs_p = [a / n for a, n in zip(stats[3], stats[1])]
        accs_g = [a / n for a, n in zip(stats[4], stats[1])]

        if acc is None:
            self.rs_test_acc.append(test_acc)
            self.rs_val_acc.append(val_acc)
            self.rs_val_p_acc.append(val_acc_p)
            self.rs_val_g_acc.append(val_acc_g)
            self.rs_test_g_acc.append(test_acc_g)
            self.rs_test_p_acc.append(test_acc_p)
            self.uncertainty_filtered_acc.append(filtered_acc)
        else:
            raise NotImplementedError
        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        logging.info("Averaged Train Loss: {:.4f}".format(train_loss))
        logging.info("Averaged Test Accuracy Fusion       : {:.4f}".format(test_acc))
        logging.info("Averaged Test Accuracy Personalized : {:.4f}".format(test_acc_p))
        logging.info("Averaged Test Accuracy Global       : {:.4f}".format(test_acc_g))

        logging.info("Averaged Validation Accuracy Fusion       : {:.4f}".format(val_acc))
        logging.info("Averaged Validation Accuracy Personalized : {:.4f}".format(val_acc_p))
        logging.info("Averaged Validation Accuracy Global       : {:.4f}".format(val_acc_g))

        # logging.info("Averaged Test AUC: {:.4f}".format(test_auc))
        # self.print_(test_acc, train_acc, train_loss)
        logging.info("Std Test Accuracy Fusion: {:.4f}".format(np.std(accs)))
        logging.info("Std Test Accuracy Person: {:.4f}".format(np.std(accs_p)))
        logging.info("Std Test Accuracy Global: {:.4f}".format(np.std(accs_g)))
        # logging.info("Std Test AUC: {:.4f}".format(np.std(aucs)))

        ns_l = np.stack(stats[6], axis=1)
        ct_l = np.stack(stats[7], axis=1)
        ct_p_l = np.stack(stats[8], axis=1)
        ct_g_l = np.stack(stats[9], axis=1)
        ns_l_sum = np.sum(ns_l, axis=1)
        ct_l_sum = np.sum(ct_l, axis=1)
        ct_p_l_sum = np.sum(ct_p_l, axis=1)
        ct_g_l_sum = np.sum(ct_g_l, axis=1)

        acc_per_class = np.round(ct_l_sum / ns_l_sum, decimals=3)
        acc_per_class_p = np.round(ct_p_l_sum / ns_l_sum, decimals=3)
        acc_per_class_g = np.round(ct_g_l_sum / ns_l_sum, decimals=3)

        head = [f'Class {t}' for t in range(self.num_classes)]
        display_table = np.stack([np.array(head), acc_per_class, acc_per_class_p, acc_per_class_g], axis=0)
        logging.info('Per-Class Results')
        logging.info(tabulate(display_table))

        logging.info('In Client Results')
        acc_per_class_in_client = ct_l / ns_l
        acc_per_class_in_client = np.round(acc_per_class_in_client, decimals=3)
        acc_per_class_in_client_p = ct_p_l / ns_l
        acc_per_class_in_client_p = np.round(acc_per_class_in_client_p, decimals=3)
        acc_per_class_in_client_g = ct_g_l / ns_l
        acc_per_class_in_client_g = np.round(acc_per_class_in_client_g, decimals=3)

        parentheses_array = np.full_like(acc_per_class_in_client, ' (', dtype='U10')
        parentheses_array_ = np.full_like(acc_per_class_in_client, ')', dtype='U10')

        acc_per_class_in_client = np.core.defchararray.add(np.array(acc_per_class_in_client, dtype='U10'),
                                                           parentheses_array)
        acc_per_class_in_client = np.core.defchararray.add(acc_per_class_in_client,
                                                           np.array(ns_l.astype(int), dtype='U10'))
        acc_per_class_in_client = np.core.defchararray.add(acc_per_class_in_client, parentheses_array_)
        acc_per_class_in_client_p = np.core.defchararray.add(np.array(acc_per_class_in_client_p, dtype='U10'),
                                                             parentheses_array)
        acc_per_class_in_client_p = np.core.defchararray.add(acc_per_class_in_client_p,
                                                             np.array(ns_l.astype(int), dtype='U10'))
        acc_per_class_in_client_p = np.core.defchararray.add(acc_per_class_in_client_p, parentheses_array_)
        acc_per_class_in_client_g = np.core.defchararray.add(np.array(acc_per_class_in_client_g, dtype='U10'),
                                                             parentheses_array)
        acc_per_class_in_client_g = np.core.defchararray.add(acc_per_class_in_client_g,
                                                             np.array(ns_l.astype(int), dtype='U10'))
        acc_per_class_in_client_g = np.core.defchararray.add(acc_per_class_in_client_g, parentheses_array_)

        # dim0: class dim1: client
        head_arr = np.array(head).reshape(-1, 1)
        acc_per_class_in_client = np.concatenate([head_arr, acc_per_class_in_client], axis=1)
        acc_per_class_in_client_p = np.concatenate([head_arr, acc_per_class_in_client_p], axis=1)
        acc_per_class_in_client_g = np.concatenate([head_arr, acc_per_class_in_client_g], axis=1)



        total_thresh_corr = stats[10]
        total_thresh_number = stats[11]

        if self.eval_key == '1':
            logging.info('Uncertainty Filtering')
            logging.info(
                '[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]')
            logging.info(total_thresh_corr / total_thresh_number)

        return total_thresh_number      # 计算Coverage

    def extract_trustworthy_holdout_set(self, OOD=True, data_path=None):

        if data_path is None:
            if OOD:
                logging.info('OOD')
                loaded_data = torch.load('/home/chenjinqian/code/MINE_FL/PFL/system/trust_hold.pth')
                trusted_data = loaded_data['images'].cuda()
            else:
                logging.info('In Domain')
                loaded_data = torch.load('/home/chenjinqian/code/MINE_FL/PFL/system/trust_hold_cifar10.pth')
                trusted_data = loaded_data['images'].cuda()
        else:
            loaded_data = torch.load(data_path)
            trusted_data = loaded_data['images'].cuda()
        return trusted_data

    def check_model_uncertainty_within_threshold(self, model_uncertainty):
        result_mask = np.zeros_like(model_uncertainty, dtype=np.bool_)
        for i in range(model_uncertainty.shape[0]):
            if model_uncertainty[i] == -1e32:
                continue
            diff = np.abs(model_uncertainty - model_uncertainty[i])
            if (diff < self.distinct_threshold).sum() > 1:
                result_mask[i] = True
        return result_mask

    def estimate_model_uncertainty(self):
        # 提取上传的所有模型在OOD数据集上的意见
        client_ref = self.selected_clients[0]
        trusted_data = self.extract_trustworthy_holdout_set(OOD=self.OOD_eval, data_path=self.holdout_data_path)
        opinion_list = []
        id_list = []
        for i in range(len(self.uploaded_models)):
            model = self.uploaded_models[i]
            fused_head_opinion = client_ref.forward_via_model_explicit(trusted_data, model)
            id_list.append(self.uploaded_ids[i])
            opinion_list.append(fused_head_opinion)


        # 直接过滤含NaN的意见
        count_nan = 0
        filtered_corr = 0
        filtered_idx_list = []
        new_opinion_list = []
        for k in range(len(opinion_list)):
            opinion = opinion_list[k]
            if torch.isnan(opinion.dir_param).any():
                filtered_idx_list.append(k)
                if id_list[k] in self.malicious_id_list:
                    filtered_corr += 1
                count_nan += 1
                continue
            opinion.uncertainty = 0.5
            new_opinion_list.append(opinion)
        logging.info(f'Filter NaN Client: {count_nan}')
        logging.info(f'Filter Correct: {filtered_corr}')


        opinion_gt = None
        # 计算融合后的意见
        for i in range(len(new_opinion_list) - 1):
            if i == 0:
                opinion_gt = new_opinion_list[i]
            to_be_fused_opinion = new_opinion_list[i + 1]
            opinion_gt = client_ref.opinion_fusion(opinion_gt, to_be_fused_opinion, fusion_key='weighted')


        # 计算模型不确定性
        temp_model_uncertainty_array = np.ones(len(id_list))

        for j in range(len(opinion_list)):
            opinion_temp = opinion_list[j]
            if j in filtered_idx_list:
                temp_model_uncertainty_array[j] = -1e32  # 为了方便后续操作
            else:
                temp_uncertainty = kl_divergence(opinion_temp.dir_param, opinion_gt.dir_param)
                temp_model_uncertainty_array[j] = temp_uncertainty.mean().item()

        # 过滤共谋攻击
        mask_distinct = self.check_model_uncertainty_within_threshold(temp_model_uncertainty_array)
        logging.info(f'Distinction Filter: {mask_distinct.sum()}')
        temp_model_uncertainty_array[mask_distinct] = -1e32

        self.model_uncertainty_list = temp_model_uncertainty_array

        # 汇总丢弃列表
        mask_filter = self.model_uncertainty_list == -1e32
        discard_list = np.array(id_list)[mask_filter]

        # 输出信息，方便调试
        for i in range(len(id_list)):
            curr_id = id_list[i]
            if curr_id in self.malicious_id_list:
                logging.info(f'Malicious ID: {curr_id}, Model Uncertainty: {self.model_uncertainty_list[i]}')

        total_count_corr = 0
        for filtered_id in discard_list:
            if filtered_id in self.malicious_id_list:
                total_count_corr += 1

        logging.info(f'Filter Percent: {total_count_corr/len(self.malicious_id_list)}')

        self.true_positive_count += total_count_corr
        self.false_positive_count += (len(discard_list) - total_count_corr)
        self.false_negative_count += (len(self.malicious_id_list) - total_count_corr)


        # 重新组织模型与权重
        new_uploaded_models = [model for i, model in enumerate(self.uploaded_models) if
                               self.uploaded_ids[i] not in discard_list]



        new_uploaded_nums = [sample_num for i, sample_num in enumerate(self.uploaded_sample_nums) if
                                self.uploaded_ids[i] not in discard_list]

        new_model_u = self.model_uncertainty_list[self.model_uncertainty_list != -1e32]


        new_uploaded_weights = compute_weights(new_uploaded_nums, new_model_u, self.reweight_u)

        self.uploaded_weights = new_uploaded_weights
        self.uploaded_models = new_uploaded_models
        logging.info(self.model_uncertainty_list)
        logging.info(new_uploaded_weights)
        # logging.info(id_list)
        logging.info(len(self.uploaded_models))
        logging.info(len(self.uploaded_weights))


    def save_results(self):
        result_path = os.path.join(self.save_path, 'record')
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        list_prior_record = []
        list_cid = []
        for c in self.clients:
            if c.id not in self.malicious_id_list:
                list_prior_record.append(c.prior_record)
                list_cid.append(c.id)
        if self.join_ratio != 1:
            list_prior_record = [0]

        if len(self.rs_test_acc):
            file_path = os.path.join(result_path, 'record.h5')
            self.record_save_path = file_path
            logging.info("Save record path: " + file_path)

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_val_acc', data=self.rs_val_acc)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
                hf.create_dataset('rs_test_g_acc', data=self.rs_test_g_acc)
                hf.create_dataset('rs_test_p_acc', data=self.rs_test_p_acc)
                hf.create_dataset('rs_val_p_acc', data=self.rs_val_p_acc)
                hf.create_dataset('rs_val_g_acc', data=self.rs_val_g_acc)
                hf.create_dataset('filtered_acc', data=self.uncertainty_filtered_acc)
                hf.create_dataset('prior_record', data=list_prior_record)
                hf.create_dataset('cid', data=list_cid)
                hf.create_dataset('filter_stats', data=[self.true_positive_count, self.false_positive_count, self.false_negative_count])

    def evaluate_corruption(self):
        if not self.check_corruption_data_exist():
            logging.info('No corruption data exists')
            return -1
        # result = np.zeros(5)        # 5 severity levels
        list_acc_arr = []
        list_sample_num = []
        sum_sample = 0

        server_list_thres_corr = []
        server_list_thres_num = []

        for c in self.clients:
            acc_arr, num, thres_acc, thres_count = c.evaluate_corruption()
            list_acc_arr.append(acc_arr)
            list_sample_num.append(num)
            if len(server_list_thres_corr) == 0:
                server_list_thres_corr = thres_acc
                server_list_thres_num = thres_count
            else:
                server_list_thres_corr = [server_list_thres_corr[i] + thres_acc[i] for i in range(len(thres_acc))]
                server_list_thres_num = [server_list_thres_num[i] + thres_count[i] for i in range(len(thres_acc))]

        total_acc_arr = np.stack(list_acc_arr, axis=0)  # N, 5
        total_sample_num = np.array(list_sample_num)
        weight = total_sample_num / np.sum(total_sample_num)
        logging.info(weight.shape)
        total_acc_arr = total_acc_arr * weight[:, np.newaxis]
        result_arr = np.sum(total_acc_arr, axis=0)
        logging.info('Robustness on corruption result:')
        logging.info(result_arr)

        result_thres = np.array(server_list_thres_corr) / np.array(server_list_thres_num)
        print(f'tolerant: {result_thres[:, 4]}')
        print(f'fair: {result_thres[:, 10]}')
        print(f'strict: {result_thres[:, 17]}')
        return result_arr


    def static_evaluate(self, acc=None, loss=None):
        self.selected_clients = self.clients
        stats = self.test_metrics()
        stats_val = self.test_metrics(use_val=True)
        stats_train = self.train_metrics()

        test_acc = sum(stats[2]) * 1.0 / sum(stats[1])
        test_acc_p = sum(stats[3]) * 1.0 / sum(stats[1])
        test_acc_g = sum(stats[4]) * 1.0 / sum(stats[1])

        val_acc = sum(stats_val[2]) * 1.0 / sum(stats_val[1])
        val_acc_p = sum(stats_val[3]) * 1.0 / sum(stats_val[1])
        val_acc_g = sum(stats_val[4]) * 1.0 / sum(stats_val[1])

        filtered_acc = stats[10] / stats[11]
        # if test_acc > self.early_stop_acc or test_acc_p > self.early_stop_acc:
        #     logging.info(f'Current Communication Cost:{self.communication_cost}')
        #     exit(1)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_test_acc = test_acc
            self.best_round = self.curr_round
            self.save_snapshot()

        train_loss = sum(stats_train[2]) * 1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        accs_p = [a / n for a, n in zip(stats[3], stats[1])]
        accs_g = [a / n for a, n in zip(stats[4], stats[1])]

        if acc is None:
            self.rs_test_acc.append(test_acc)
            self.rs_val_acc.append(val_acc)
            self.rs_val_p_acc.append(val_acc_p)
            self.rs_val_g_acc.append(val_acc_g)
            self.rs_test_g_acc.append(test_acc_g)
            self.rs_test_p_acc.append(test_acc_p)
            self.uncertainty_filtered_acc.append(filtered_acc)
        else:
            raise NotImplementedError
        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        logging.info("Averaged Train Loss: {:.4f}".format(train_loss))
        logging.info("Averaged Test Accuracy Personalized : {:.4f}".format(test_acc_p))
        logging.info("Averaged Test Accuracy Global       : {:.4f}".format(test_acc_g))

        logging.info("Averaged Validation Accuracy Personalized : {:.4f}".format(val_acc_p))
        logging.info("Averaged Validation Accuracy Global       : {:.4f}".format(val_acc_g))

        logging.info("Std Test Accuracy Person: {:.4f}".format(np.std(accs_p)))
        logging.info("Std Test Accuracy Global: {:.4f}".format(np.std(accs_g)))


        ns_l = np.stack(stats[6], axis=1)
        ct_l = np.stack(stats[7], axis=1)
        ct_p_l = np.stack(stats[8], axis=1)
        ct_g_l = np.stack(stats[9], axis=1)
        ns_l_sum = np.sum(ns_l, axis=1)
        ct_l_sum = np.sum(ct_l, axis=1)
        ct_p_l_sum = np.sum(ct_p_l, axis=1)
        ct_g_l_sum = np.sum(ct_g_l, axis=1)

        acc_per_class = np.round(ct_l_sum / ns_l_sum, decimals=3)
        acc_per_class_p = np.round(ct_p_l_sum / ns_l_sum, decimals=3)
        acc_per_class_g = np.round(ct_g_l_sum / ns_l_sum, decimals=3)

        head = [f'Class {t}' for t in range(self.num_classes)]
        display_table = np.stack([np.array(head), acc_per_class, acc_per_class_p, acc_per_class_g], axis=0)
        logging.info('Per-Class Results')
        logging.info(tabulate(display_table))

        logging.info('In Client Results')
        acc_per_class_in_client = ct_l / ns_l
        acc_per_class_in_client = np.round(acc_per_class_in_client, decimals=3)
        acc_per_class_in_client_p = ct_p_l / ns_l
        acc_per_class_in_client_p = np.round(acc_per_class_in_client_p, decimals=3)
        acc_per_class_in_client_g = ct_g_l / ns_l
        acc_per_class_in_client_g = np.round(acc_per_class_in_client_g, decimals=3)

        parentheses_array = np.full_like(acc_per_class_in_client, ' (', dtype='U10')
        parentheses_array_ = np.full_like(acc_per_class_in_client, ')', dtype='U10')

        acc_per_class_in_client = np.core.defchararray.add(np.array(acc_per_class_in_client, dtype='U10'),
                                                           parentheses_array)
        acc_per_class_in_client = np.core.defchararray.add(acc_per_class_in_client,
                                                           np.array(ns_l.astype(int), dtype='U10'))
        acc_per_class_in_client = np.core.defchararray.add(acc_per_class_in_client, parentheses_array_)
        acc_per_class_in_client_p = np.core.defchararray.add(np.array(acc_per_class_in_client_p, dtype='U10'),
                                                             parentheses_array)
        acc_per_class_in_client_p = np.core.defchararray.add(acc_per_class_in_client_p,
                                                             np.array(ns_l.astype(int), dtype='U10'))
        acc_per_class_in_client_p = np.core.defchararray.add(acc_per_class_in_client_p, parentheses_array_)
        acc_per_class_in_client_g = np.core.defchararray.add(np.array(acc_per_class_in_client_g, dtype='U10'),
                                                             parentheses_array)
        acc_per_class_in_client_g = np.core.defchararray.add(acc_per_class_in_client_g,
                                                             np.array(ns_l.astype(int), dtype='U10'))
        acc_per_class_in_client_g = np.core.defchararray.add(acc_per_class_in_client_g, parentheses_array_)

        # dim0: class dim1: client
        head_arr = np.array(head).reshape(-1, 1)
        acc_per_class_in_client = np.concatenate([head_arr, acc_per_class_in_client], axis=1)
        acc_per_class_in_client_p = np.concatenate([head_arr, acc_per_class_in_client_p], axis=1)
        acc_per_class_in_client_g = np.concatenate([head_arr, acc_per_class_in_client_g], axis=1)



        total_thresh_corr = stats[10]
        total_thresh_number = stats[11]

        if self.eval_key == '1':
            logging.info('Uncertainty Filtering')
            logging.info(
                '[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]')
            logging.info(total_thresh_corr / total_thresh_number)

        return total_thresh_number  # 计算Coverage


def compute_weights(sample_sizes, uncertainties, reweight_u=False):
    sample_sizes = np.array(sample_sizes)
    uncertainties = np.array(uncertainties)
    if reweight_u:
        U_norm = (uncertainties - uncertainties.min()) / (uncertainties.max() - uncertainties.min())
    else:
        U_norm = np.zeros_like(uncertainties)
    exp_neg_U = np.exp(-U_norm)
    weighted_scores = sample_sizes * exp_neg_U
    weights = weighted_scores / np.sum(weighted_scores)

    return weights.tolist()