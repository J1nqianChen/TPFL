import torch
import numpy as np
import logging
from torch import nn


class DiversityLoss(nn.Module):
    """
    Diversity loss for improving the performance.
    """

    def __init__(self, metric):
        """
        Class initializer.
        """
        super().__init__()
        self.metric = metric
        self.cosine = nn.CosineSimilarity(dim=2)

    def compute_distance(self, tensor1, tensor2, metric):
        """
        Compute the distance between two tensors.
        """
        if metric == 'l1':
            return torch.abs(tensor1 - tensor2).mean(dim=(2,))
        elif metric == 'l2':
            return torch.pow(tensor1 - tensor2, 2).mean(dim=(2,))
        elif metric == 'cosine':
            return 1 - self.cosine(tensor1, tensor2)
        else:
            raise ValueError(metric)

    def pairwise_distance(self, tensor, how):
        """
        Compute the pairwise distances between a Tensor's rows.
        """
        n_data = tensor.size(0)
        tensor1 = tensor.expand((n_data, n_data, tensor.size(1)))
        tensor2 = tensor.unsqueeze(dim=1)
        return self.compute_distance(tensor1, tensor2, how)

    def forward(self, noises, layer, y_input=None, diversity_loss_type=None):
        """
        Forward propagation.
        """
        if len(layer.shape) > 2:
            layer = layer.view((layer.size(0), -1))
        if diversity_loss_type == 'div2':
            y_input_dist = self.pairwise_distance(y_input, how='l1')
        layer_dist = self.pairwise_distance(layer, how=self.metric)
        noise_dist = self.pairwise_distance(noises, how='l2')
        if diversity_loss_type == 'div2':
            return torch.exp(-torch.mean(noise_dist * layer_dist * torch.exp(y_input_dist)))
        else:
            return torch.exp(-torch.mean(noise_dist * layer_dist))


def not_true_consistency_loss(p_dir, g_dir, scalar_y, num_classes=10):
    y = torch.eye(num_classes)[scalar_y].cuda()
    p_dir = p_dir * (1 - y) + y
    g_dir = g_dir * (1 - y) + y
    loss_ntc = kl_divergence(p_dir, g_dir)
    loss = torch.mean(loss_ntc)
    return loss


def not_true_consistency_l2_loss(p_dir, g_dir, scalar_y, num_classes=10):
    scalar_y = scalar_y.cpu()
    y = torch.eye(num_classes).cuda()[scalar_y]
    p_dir = p_dir * (1 - y) + y
    g_dir = g_dir * (1 - y) + y
    loss_ntc = torch.sum((p_dir - g_dir) ** 2, dim=1)
    loss = torch.mean(loss_ntc)
    return loss


def kl_divergence(alpha_1, alpha_2):
    eps = 1e-7
    alpha_1 = alpha_1 + eps  # B, K
    if alpha_1.shape != alpha_2.shape:
        alpha_2 = torch.unsqueeze(alpha_2.squeeze() + eps, dim=0).repeat(alpha_1.shape[0], 1)  # B, K

    sum_alpha_1 = torch.sum(alpha_1, dim=1, keepdim=True)  # (B, 1)
    sum_alpha_2 = torch.sum(alpha_2, dim=1, keepdim=True)  # (B, 1)
    # sum_alpha = torch.sum(_alpha, dim=1, keepdim=True)
    first_term = (torch.lgamma(sum_alpha_1) - torch.lgamma(sum_alpha_2))  # (B, 1)

    second_term = torch.sum(torch.lgamma(alpha_2) - torch.lgamma(alpha_1), dim=1, keepdim=True)  # (B, 1)

    third_term = torch.sum((alpha_1 - alpha_2) * (torch.digamma(alpha_1) - torch.digamma(sum_alpha_1)), dim=1,
                           keepdim=True)
    kl = first_term + second_term + third_term

    inf_nan_mask = torch.isinf(kl) | torch.isnan(kl)

    # Check if there are any inf or NaN values
    has_inf_nan = inf_nan_mask.any().item()

    # if has_inf_nan:
    #     raise RuntimeError

    # logging.info(kl.mean())
    return kl


def incorrect_belief_regularization(evidence, dir_alpha, y):
    """

    Args:
        evidence: B, K
        dir_alpha: B, K
        y: B, K

    Returns:

    """
    S = torch.sum(dir_alpha, dim=1, keepdim=True)
    regularization_term = (evidence / S) * (torch.ones_like(y) - y)
    regularization_term = torch.sum(regularization_term, dim=1)
    incorr_loss = torch.mean(regularization_term)
    return incorr_loss


def e_log_loss(scalar_y, alpha, basic_evidence, current_round, num_classes=10, incor_w=0.05):
    """

    Args:
        scalar_y: (B,)
        alpha: (B, N)
        basic_evidence: (N,)
        current_round:
        num_classes:
        rebalance:

    Returns:

    """
    scalar_y = scalar_y.cpu()
    y = torch.eye(num_classes)[scalar_y].cuda()
    S = torch.sum(alpha, dim=1, keepdims=True)
    A = torch.sum(y * (torch.log(S) - torch.log(alpha)), dim=1, keepdim=True)

    inf_mask = torch.isinf(A)
    nan_mask = torch.isnan(A)

    # Check if there are any inf or NaN values
    has_inf = inf_mask.any().item()
    has_nan = nan_mask.any().item()

    if has_inf:
        logging.info(alpha)
        raise RuntimeError

    if has_nan:
        logging.info(alpha)
        raise RuntimeError

    kl_alpha = basic_evidence - basic_evidence * (1 - y) + alpha * (1 - y)
    #
    # kl_alpha = y + (1 - y) * alpha
    # basic_evidence_ = torch.ones(alpha.shape[1]).cuda()
    incor_loss = kl_divergence(kl_alpha, basic_evidence)
    evidence = alpha - basic_evidence
    # incor_loss = incorrect_belief_regularization(evidence, alpha, y)

    lamda_ = incor_w
    eta = lamda_ * min(1, current_round / 10)

    # eta = 0.1
    total_loss = A + eta * incor_loss

    mean_loss = torch.mean(total_loss)

    return mean_loss



class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point.
        # Edge case e.g.:-
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan]
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss