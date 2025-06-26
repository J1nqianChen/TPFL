import copy
import logging
import os
import pickle
import sys
import time

import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

sys.path.append('/home/chenjinqian/code/MINE_FL/')
from PFL.dataset.Read_Data import read_cifar_c
from flcore.clients.clientbase import Client
from utils.privacy import *
from flcore.optimizers.losses import *


class NegativePenaltyLoss(nn.Module):
    def __init__(self, weight=1.0):
        super(NegativePenaltyLoss, self).__init__()
        self.weight = weight

    def forward(self, tensor):
        # Apply the penalty only to negative values
        negative_values = torch.clamp(tensor, max=0)
        penalty = self.weight * torch.sum(torch.abs(negative_values))

        return penalty



# Debug Tools
class UncertaintyRegularizedLoss(torch.nn.Module):
    def __init__(self, lambda_reg, threshold):
        super(UncertaintyRegularizedLoss, self).__init__()
        self.lambda_reg = lambda_reg
        self.threshold = threshold

    def forward(self, max_evidence):
        # Regularization term
        # reg_term = torch.max(torch.tensor(0.0), -uncertainty + self.threshold) ** 2
        reg_term = torch.max(torch.tensor(0.0), max_evidence - self.threshold) ** 2

        # Total loss
        loss = self.lambda_reg * reg_term
        loss = torch.mean(loss)
        return loss


class SubjectiveOpinion:
    def __init__(self, belief_mass, uncertainty, base_rate=None):
        self.belief_mass = belief_mass
        self.uncertainty = uncertainty
        if base_rate is None:
            self.base_rate = torch.ones(belief_mass.shape[1]) / belief_mass.shape[1]
        else:
            self.base_rate = base_rate

        self.evidence = None
        self.basic_evidence = None
        self.retrieve_evidence()

        self.dir_param = self.evidence + self.basic_evidence

    def expected_probability(self):
        expected_p = self.belief_mass + self.base_rate * self.uncertainty
        return expected_p

    def retrieve_evidence(self):
        # if torch.any(self.uncertainty == 0):
        #     raise RuntimeError
        sum_evidence = (self.belief_mass.shape[1] / self.uncertainty) - self.belief_mass.shape[1]
        fused_evidence = self.belief_mass * (sum_evidence + self.belief_mass.shape[1])
        self.evidence = fused_evidence
        self.basic_evidence = self.base_rate * self.belief_mass.shape[1]

    def reformat_evidence(self):
        # if 0 in self.basic_evidence:
        #     self.basic_evidence += 1e-1
        # self.basic_evidence = self.evidence / self.evidence.sum()
        if len(self.basic_evidence.shape) == 1:
            self.basic_evidence = torch.unsqueeze(self.basic_evidence, dim=0).repeat(self.evidence.shape[0], 1)


def check_opinion(opinion: SubjectiveOpinion):
    diff_op = opinion.dir_param - opinion.basic_evidence
    num_negative_elements = torch.sum(diff_op < 0).item()
    if num_negative_elements > 0:
        raise RuntimeError


class clientTPFL(Client):
    def __init__(self, config, id, train_samples, test_samples, **kwargs):
        super().__init__(config, id, train_samples, test_samples, **kwargs)

        self.loss = nn.CrossEntropyLoss()
        logging.info(f'Client:{self.id} Initializing')
        self.goal = config.train.goal

        try:
            self.eval_key = config.algo_hyperparam.TPFL.eval_key
        except:
            self.eval_key = None

        # differential privacy
        # if self.privacy:
        #     check_dp(self.model)
        #     initialize_dp(self.model, self.optimizer, self.sample_rate, self.dp_sigma)

        # Base Rate
        self.base_rate = torch.tensor(self.get_label_distribution()).cuda()
        self.global_subjective_model = copy.deepcopy(self.model)

        self.red = config.algo_hyperparam.TPFL.red
        self.num_clients = config.train.num_clients
        self.activation_func_key = config.algo_hyperparam.TPFL.activation_func_key
        if self.activation_func_key == 'ReLU':
            self.activation_func = F.relu
        elif self.activation_func_key == 'Softplus':
            self.activation_func = F.softplus
        elif self.activation_func_key == 'exp':
            self.activation_func = torch.exp
        else:
            raise NotImplementedError('Undefined Activation Function')

        self.W = self.num_classes
        self.fusion_key = config.algo_hyperparam.TPFL.fusion_key

        self.avg_uncertainty = None
        self.l_reg = config.algo_hyperparam.TPFL.l_reg
        self.l_reg_threshold = config.algo_hyperparam.TPFL.l_reg_threshold
        self.maxEvi_loss = UncertaintyRegularizedLoss(self.l_reg, self.l_reg_threshold)

        self.prior_logits = nn.Parameter((torch.ones(self.W)).cuda(), requires_grad=True)
        self.incor_w = config.algo_hyperparam.TPFL.incor_weight

        model_parameters = list(self.model.parameters())
        trainable_prior_parameters = [self.prior_logits]

        # Concatenate the parameters lists
        all_parameters = model_parameters + trainable_prior_parameters

        self.optimizer = torch.optim.SGD(all_parameters, lr=self.learning_rate, momentum=0.9,
                                         weight_decay=1e-5)

        # self.npl = config.algo_hyperparam.TPFL.negative_penalty_weight

        self.prior_record = []

        if self.red:
            logging.info('Using Red')

        self.pre_model_head = None
    def get_prior(self):
        prior = torch.softmax(self.prior_logits, dim=0)
        return prior

    def get_evidence(self, y):
        e = self.activation_func(y)
        return e

    def opinion_fusion(self, opinion_p: SubjectiveOpinion,
                       opinion_g: SubjectiveOpinion, fusion_key):
        if fusion_key == 'weighted':

            belief_mass = ((opinion_p.belief_mass * (1 - opinion_p.uncertainty) * opinion_g.uncertainty +
                            opinion_g.belief_mass * (1 - opinion_g.uncertainty) * opinion_p.uncertainty) /
                           (
                                   opinion_p.uncertainty + opinion_g.uncertainty - 2 * opinion_g.uncertainty * opinion_p.uncertainty))

            uncertainty = ((
                                   2 - opinion_p.uncertainty - opinion_g.uncertainty) * opinion_p.uncertainty * opinion_g.uncertainty) / (
                                  opinion_p.uncertainty + opinion_g.uncertainty - 2 * opinion_g.uncertainty * opinion_p.uncertainty)
            base_rate = (opinion_p.base_rate * (1 - opinion_p.uncertainty) + opinion_g.base_rate * (
                    1 - opinion_g.uncertainty)) / (2 - opinion_p.uncertainty - opinion_g.uncertainty)
        elif fusion_key == 'cumulative':
            belief_mass = (
                                  opinion_p.belief_mass * opinion_g.uncertainty + opinion_g.belief_mass * opinion_p.uncertainty) / (
                                  opinion_p.uncertainty + opinion_g.uncertainty - opinion_g.uncertainty * opinion_p.uncertainty)
            uncertainty = (opinion_p.uncertainty * opinion_g.uncertainty) / (
                    opinion_p.uncertainty + opinion_g.uncertainty - opinion_g.uncertainty * opinion_p.uncertainty)
            base_rate = (opinion_p.base_rate * opinion_g.uncertainty + opinion_g.base_rate * opinion_p.uncertainty - (
                    opinion_p.base_rate + opinion_g.base_rate) * opinion_p.uncertainty * opinion_g.uncertainty) / (
                                opinion_p.uncertainty + opinion_g.uncertainty - 2 * opinion_g.uncertainty * opinion_p.uncertainty)



        elif fusion_key == 'constraint':
            conflict_term = torch.matmul(opinion_p.belief_mass.unsqueeze(-1),
                                         opinion_g.belief_mass.unsqueeze(-1).permute(0, 2, 1))
            for i in range(conflict_term.shape[0]):
                conflict_term[i].fill_diagonal_(0)
            c = torch.sum(torch.sum(conflict_term, dim=2, keepdim=False), dim=1, keepdim=False).unsqueeze(-1)
            harmony_term = opinion_p.uncertainty * opinion_p.belief_mass + opinion_g.uncertainty * opinion_g.belief_mass + opinion_p.belief_mass * opinion_g.belief_mass

            belief_mass = harmony_term / (1 - c)
            uncertainty = opinion_p.uncertainty * opinion_g.uncertainty / (1 - c)
            base_rate = (opinion_p.base_rate * (1 - opinion_p.uncertainty) + opinion_g.base_rate * (
                    1 - opinion_g.uncertainty)) / (2 - opinion_p.uncertainty - opinion_g.uncertainty)


        else:
            raise NotImplementedError

        # if self.train_trainable_prior:
        #     belief_mass = torch.tensor(belief_mass).cuda()
        #     uncertainty = torch.tensor(uncertainty).cuda()
        #     base_rate = torch.tensor(base_rate).cuda()

        fusion_opinion = SubjectiveOpinion(belief_mass, uncertainty, base_rate)
        return fusion_opinion

    def form_opinion(self, evidence, base_rate, num_classes=10):

        W = num_classes
        belief_mass = evidence / (W + torch.sum(evidence, dim=1, keepdim=True))
        uncertainty = W / (W + torch.sum(evidence, dim=1, keepdim=True))
        opinion = SubjectiveOpinion(belief_mass, uncertainty, base_rate)

        return opinion


    def train(self):
        if self.attack_strategy == 'flip' or not self.is_malicious:
            uncertainty_list = []

            if self.is_malicious and self.attack_strategy == 'flip':
                trainloader = self.load_train_data(origin=False)

            else:
                trainloader = self.load_train_data(origin=True)
                self.global_subjective_model.eval()
                self.global_subjective_model.eval()

            start_time = time.time()

            self.model.train()
            self.prior_logits.requires_grad = True

            pre_prior = copy.deepcopy(self.prior_logits).detach().cpu().numpy()
            max_local_steps = self.local_steps

            for step in range(max_local_steps):
                uncertainty_list = []
                for i, (x, y) in enumerate(trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    self.optimizer.zero_grad()

                    output_p = self.model(x)

                    evidence_p = self.get_evidence(output_p)

                    # logging.info(torch.mean(torch.max(evidence_g, dim=1)[0], dim=0))

                    t_prior = self.get_prior()
                    opinion_p = self.form_opinion(evidence_p, t_prior, self.W)

                    opinion_fused = opinion_p
                    opinion_fused.reformat_evidence()

                    uncertainty = opinion_p.uncertainty
                    uncertainty_list.append(torch.mean(uncertainty.cpu().detach()).item())

                    dir_alpha = opinion_fused.evidence + opinion_fused.basic_evidence

                    loss = e_log_loss(y, dir_alpha, opinion_fused.basic_evidence[0, :], current_round=self.curr_round,
                                      num_classes=self.num_classes, incor_w=self.incor_w)

                    # uncertainty regularization
                    max_evidence, _ = torch.max(opinion_fused.evidence, dim=1, keepdim=False)
                    u_loss = self.maxEvi_loss(max_evidence)
                    loss += u_loss

                    u = opinion_fused.uncertainty.detach()
                    gt_evidence = opinion_fused.dir_param[np.arange(u.shape[0]), y].reshape(-1, 1)
                    gt_basic_evidence = opinion_fused.basic_evidence[np.arange(u.shape[0]), y].reshape(-1, 1)
                    # eps = 1e-6
                    if self.red:

                        red_term = - u * torch.log((gt_evidence - gt_basic_evidence + 1e-5))
                        mean_red = torch.mean(red_term)
                        # logging.info(f'red_loss: {red_term.mean()}')
                        if torch.isinf(red_term).sum() > 0:
                            logging.info('Assert RedTerm')
                        loss += mean_red


                    loss.backward()

                    model_parameters = list(self.model.parameters())
                    trainable_prior_parameters = [self.prior_logits]

                    # Concatenate the parameters lists
                    all_parameters = model_parameters + trainable_prior_parameters

                    max_norm = 1.0
                    torch.nn.utils.clip_grad_norm_(all_parameters, max_norm)
                    self.optimizer.step()
            self.prior_logits.requires_grad = False

            equal_flag = (self.prior_logits.cpu().numpy() == pre_prior).all()
            if equal_flag:
                raise RuntimeError('Prior is not Training ')

            self.train_time_cost['num_rounds'] += 1
            self.train_time_cost['total_cost'] += time.time() - start_time

            self.avg_uncertainty = np.mean(uncertainty_list)

            # if self.privacy:
            #     res, DELTA = get_dp_params(self.optimizer)
            #     logging.info(f"Client {self.id}", f"(ε = {res[0]:.2f}, δ = {DELTA}) for α = {res[1]}")

            self.prior_record.append(copy.deepcopy(self.get_prior()).cpu().numpy())

            if self.attack_strategy == 'flip' and self.is_malicious:
                logging.info('Model Replacement')
                self.pre_model_head = copy.deepcopy(self.model.head)
                for param_client, param_global in zip(self.model.parameters(), self.global_subjective_model.parameters()):
                    param_client.data = self.num_clients * param_client.data - (
                            self.num_clients - 1) * param_global.data
        else:
            logging.info('Model Poisoning')
            self.attack_mp()

    def set_parameters(self, model):
        if self.attack_strategy == 'flip' and self.is_malicious and self.curr_round != 0:
            logging.info('Retrieve Head')
            for new_param, old_param in zip(self.pre_model_head.parameters(), self.model.head.parameters()):
                old_param.data = new_param.data.clone()

        for new_param, old_param in zip(model.parameters(), self.global_subjective_model.parameters()):
            old_param.data = new_param.data.clone()

        for new_param, old_param in zip(model.base.parameters(), self.model.base.parameters()):
            old_param.data = new_param.data.clone()

    def forward_via_head(self, model, head1, head2, x, fusion_key='weighted'):
        model.eval()
        head1.eval()
        head2.eval()
        features = model.base(x)
        output1 = head1(features)
        output2 = head2(features)

        evidence1 = self.get_evidence(output1)
        evidence2 = self.get_evidence(output2)

        base_rate_uniform = (torch.ones(self.W) / self.W).cuda()
        opinion1 = self.form_opinion(evidence1, base_rate_uniform, self.W)
        opinion2 = self.form_opinion(evidence2, base_rate_uniform, self.W)
        fused_opinion = self.opinion_fusion(opinion1, opinion2, fusion_key)
        fused_evidence_sum = torch.sum(fused_opinion.evidence, dim=1, keepdim=True)
        return fused_opinion, fused_evidence_sum

    def forward_via_model(self, x):
        self.model.eval()
        output = self.model(x)

        evidence = self.get_evidence(output)

        t_prior = self.get_prior()
        opinion = self.form_opinion(evidence, t_prior, self.W)

        return opinion

    def forward_via_model_explicit(self, x, model):
        model.eval()
        output = model(x)

        evidence = self.get_evidence(output)

        base_rate_uniform = (torch.ones(self.W) / self.W).cuda()
        opinion = self.form_opinion(evidence, base_rate_uniform, self.W)

        return opinion

    def uncertainty_filtered_predict(self, opinion: SubjectiveOpinion, label):
        list_thres = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9,
                      0.95, 1.0]
        list_thres.reverse()
        correct_list = []
        total_list = []
        # logging.info(opinion.uncertainty.shape)
        # logging.info(opinion.uncertainty)
        for threshold in list_thres:
            temp_mask = opinion.uncertainty.squeeze() < threshold
            opinion_temp_dir = opinion.dir_param[temp_mask]
            y_filtered = label[temp_mask]

            test_acc_num = (torch.sum(torch.argmax(opinion_temp_dir, dim=1) == y_filtered)).item()
            correct_list.append(test_acc_num)
            total_list.append(y_filtered.shape[0])

        return correct_list, total_list

    def test_metrics(self, backdoor_evaluate=False, use_val=False):
        if not use_val:
            testloaderfull = self.load_test_data()
        else:
            testloaderfull = self.load_val_data()
        self.global_subjective_model.eval()
        self.model.eval()

        test_acc = 0
        test_num = 0
        test_acc_p = 0
        # test_num_p = 0
        test_acc_g = 0
        # test_num_g = 0
        y_prob = []
        y_true = []

        list_test_acc = np.zeros(self.num_classes)
        list_test_acc_p = np.zeros(self.num_classes)
        list_test_acc_g = np.zeros(self.num_classes)
        list_test_num = np.zeros(self.num_classes)

        with torch.no_grad():
            total_cor_arr = None
            total_num_arr = None
            for x, y in testloaderfull:

                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                output_p = self.model(x)
                evidence_p = self.get_evidence(output_p)

                self.global_subjective_model.eval()
                output_g = self.global_subjective_model(x)
                evidence_g = self.get_evidence(output_g)

                base_rate_p = copy.deepcopy((self.base_rate / self.base_rate.sum())).to(torch.float32)

                used_W = self.W
                if self.prior_logits is not None:
                    self.prior_logits.requires_grad = False
                    t_prior = self.get_prior()
                    base_rate_p = t_prior

                base_rate_g = torch.ones(self.num_classes) / used_W
                base_rate_g = base_rate_g.cuda()

                opinion_p = self.form_opinion(evidence_p, base_rate_p, used_W)
                opinion_g = self.form_opinion(evidence_g, base_rate_g, used_W)

                opinion_fused = self.opinion_fusion(opinion_p, opinion_g, fusion_key=self.fusion_key)

                if self.eval_key is not None:
                    # eval_key: 1    Uncertainty-Acc
                    if self.eval_key == '1':
                        corr_list, total_list = self.uncertainty_filtered_predict(opinion_fused, y)
                        corr_arr = np.array(corr_list)
                        total_arr = np.array(total_list)
                        if total_cor_arr is None:
                            total_cor_arr = corr_arr
                            total_num_arr = total_arr
                        else:
                            total_cor_arr = total_cor_arr + corr_arr
                            total_num_arr = total_num_arr + total_arr




                test_acc += (torch.sum(torch.argmax(opinion_fused.dir_param, dim=1) == y)).item()
                test_acc_p += (torch.sum(torch.argmax(opinion_p.dir_param, dim=1) == y)).item()
                test_acc_g += (torch.sum(torch.argmax(opinion_g.dir_param, dim=1) == y)).item()

                for i in range(self.num_classes):
                    total_i = (y == i).sum().item()
                    mask = y == i
                    list_test_num[i] += (total_i)
                    correct_i = (torch.sum(torch.argmax(opinion_fused.dir_param, dim=1)[mask] == y[mask])).item()
                    correct_p_i = (torch.sum(torch.argmax(opinion_p.dir_param, dim=1)[mask] == y[mask])).item()
                    correct_g_i = (torch.sum(torch.argmax(opinion_g.dir_param, dim=1)[mask] == y[mask])).item()
                    list_test_acc[i] += correct_i
                    list_test_acc_p[i] += correct_p_i
                    list_test_acc_g[i] += correct_g_i

                test_num += y.shape[0]



        auc = 0
        return test_acc, test_num, test_acc_p, test_acc_g, auc, list_test_num, list_test_acc, list_test_acc_p, list_test_acc_g, total_cor_arr, total_num_arr

    def evaluate_corruption(self):

        with torch.no_grad():
            list_acc = []
            list_thres_corr = []
            list_thres_count = []
            for severity_i in range(5):
                severity_ = severity_i + 1
                corr_dataset, sample_number = read_cifar_c(self.dataset + '_C', client_id=self.id, severity=severity_)
                corr_dataloader = DataLoader(corr_dataset, batch_size=self.batch_size, shuffle=True)
                test_acc = 0
                test_num = 0
                total_cor_arr = None
                total_num_arr = None
                for x, y in iter(corr_dataloader):

                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)

                    output_p = self.model(x)
                    evidence_p = self.get_evidence(output_p)

                    self.global_subjective_model.eval()

                    self.global_subjective_model.eval()
                    self.model.eval()

                    self.global_subjective_model.eval()
                    output_g = self.global_subjective_model(x)
                    evidence_g = self.get_evidence(output_g)

                    base_rate_p = copy.deepcopy((self.base_rate / self.base_rate.sum())).to(torch.float32)

                    used_W = self.W
                    if self.prior_logits is not None:
                        self.prior_logits.requires_grad = False
                        t_prior = self.get_prior()
                        base_rate_p = t_prior

                    base_rate_g = torch.ones(self.num_classes) / used_W
                    base_rate_g = base_rate_g.cuda()

                    opinion_p = self.form_opinion(evidence_p, base_rate_p, used_W)
                    opinion_g = self.form_opinion(evidence_g, base_rate_g, used_W)

                    opinion_fused = self.opinion_fusion(opinion_p, opinion_g, fusion_key=self.fusion_key)

                    corr_list, total_list = self.uncertainty_filtered_predict(opinion_fused, y)
                    corr_arr = np.array(corr_list)
                    total_arr = np.array(total_list)

                    if total_cor_arr is None:
                        total_cor_arr = corr_arr
                        total_num_arr = total_arr
                    else:
                        total_cor_arr = total_cor_arr + corr_arr
                        total_num_arr = total_num_arr + total_arr

                    test_acc += (torch.sum(torch.argmax(opinion_fused.dir_param, dim=1) == y)).item()

                    test_num += y.shape[0]

                acc_severity_i = test_acc / test_num
                list_acc.append(acc_severity_i)
                list_thres_corr.append(total_cor_arr)
                list_thres_count.append(total_num_arr)
            logging.info(
                'Evaluate TPFL')
            # logging.info(list_acc)
            return np.array(list_acc), test_num, list_thres_corr, list_thres_count
