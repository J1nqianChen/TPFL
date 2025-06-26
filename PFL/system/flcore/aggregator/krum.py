import copy
import math

import numpy as np
import logging
import torch
from torch import nn

from utils.data_utils import flatten_params, reload_params


def compute_euclidean_distance(model1: nn.Module, model2: nn.Module) -> float:
    """
    Compute the Euclidean distance between the parameters of two models.

    Args:
    model1 (nn.Module): The first model.
    model2 (nn.Module): The second model.

    Returns:
    float: The Euclidean distance between the parameters of the two models.
    """
    distance = 0.0
    with torch.no_grad():
        for param1, param2 in zip(model1.parameters(), model2.parameters()):
            distance += torch.sum((param1 - param2) ** 2).item()
        return distance ** 0.5


def krum_aggregate(uploaded_models, m):
    """

    Args:
        uploaded_models: list of uploaded models
        m: malicious number
    Returns:

    """
    update_number = len(uploaded_models)
    list_krum_sort_list = []
    for i in range(update_number):
        list_distance = []
        model1 = uploaded_models[i]
        for j in range(update_number):
            if i == j:
                continue
            else:
                model2 = uploaded_models[j]
                distance = compute_euclidean_distance(model1, model2)
                if math.isnan(distance):
                    logging.info('NaN')
                list_distance.append(distance)
        list_distance.sort()
        sum_distance = sum(list_distance[:update_number - 1 - m])
        list_krum_sort_list.append(sum_distance)
    min_idx = np.argmin(list_krum_sort_list)
    aggregated_model = uploaded_models[min_idx]
    return aggregated_model


def multikrum_aggregate(uploaded_models, uploaded_weight, m):
    """

    Args:
        uploaded_weight: list of weight
        uploaded_models: list of uploaded models
        m: number of malicious updates

    Returns:

    """
    update_number = len(uploaded_models)
    list_krum_sort_list = []
    for i in range(update_number):
        list_distance = []
        model1 = uploaded_models[i]
        for j in range(update_number):
            if i == j:
                continue
            else:
                model2 = uploaded_models[j]
                distance = compute_euclidean_distance(model1, model2)
                if math.isnan(distance):
                    logging.info('NaN')
                list_distance.append(distance)
        list_distance.sort()
        sum_distance = sum(list_distance[:update_number - 1 - m])
        list_krum_sort_list.append(sum_distance)
    sorted_idx = np.argsort(list_krum_sort_list)
    aggregated_idx = sorted_idx[:update_number - m]
    model_list = []
    weight_list = []
    for idx in aggregated_idx:
        model_list.append(uploaded_models[idx])
        weight_list.append(uploaded_weight[idx])
    norm_weight_list = [wt / sum(weight_list) for wt in weight_list]
    global_model = aggregate_parameters(model_list, norm_weight_list)
    return global_model


def trimmed_mean_aggregate(uploaded_models, uploaded_id, m):
    with torch.no_grad():
        flatten_param_list = []
        for idx in range(len(uploaded_models)):
            model = uploaded_models[idx]
            param = flatten_params(model)
            if torch.isnan(param).any():
                logging.info('NaN')
                logging.info(uploaded_id[idx])
            # param[torch.isnan(param)] = 1e32
            flatten_param_list.append(param)
        stacked_param = torch.stack(flatten_param_list, dim=0)

        sorted_stacked_param, _ = torch.sort(stacked_param, dim=0)
        temp_m = int(m / 2)
        trimmed_stace_param = sorted_stacked_param[temp_m:-temp_m, :]
        mean_param = torch.mean(trimmed_stace_param, dim=0, keepdim=False)

        final_model = copy.deepcopy(uploaded_models[0])
        reload_params(final_model, mean_param)

        return final_model


def median_aggregate(uploaded_models, m):
    with torch.no_grad():
        flatten_param_list = []
        for model in uploaded_models:
            param = flatten_params(model)
            flatten_param_list.append(param)
        stacked_param = torch.stack(flatten_param_list, dim=0)
        sorted_stacked_param, _ = torch.sort(stacked_param, dim=0)
        idx_temp = sorted_stacked_param.shape[0] // 2
        if sorted_stacked_param.shape[0] % 2 == 0:
            trimmed_stace_param = sorted_stacked_param[idx_temp - 1:idx_temp + 1, :]  # [1,2,3,4,5,6] 6/2=3 3-1 = 2
            median_param = torch.mean(trimmed_stace_param, dim=0, keepdim=False)
        else:
            median_param = sorted_stacked_param[idx_temp, :]  # [1, 2, 3, 4, 5] 5/2=2

        final_model = copy.deepcopy(uploaded_models[0])
        reload_params(final_model, median_param)

        return final_model


def add_parameters(w, client_model, global_model):
    for server_param, client_param in zip(global_model.parameters(), client_model.parameters()):
        server_param.data += client_param.data.clone() * w
    return global_model


def aggregate_parameters(uploaded_models, uploaded_weights):
    assert (len(uploaded_models) > 0)

    global_model = copy.deepcopy(uploaded_models[0])
    for param in global_model.parameters():
        param.data.zero_()

    for w, client_model in zip(uploaded_weights, uploaded_models):
        global_model = add_parameters(w, client_model, global_model)

    return global_model


def norm_clipping_aggregate(uploaded_models, uploaded_weights):
    list_param = []
    for model in uploaded_models:
        param = flatten_params(model)
        list_param.append(param)
    stacked_param = torch.stack(list_param, dim=0)
    stacked_norm = torch.linalg.norm(stacked_param, ord=2, dim=1)
    mask_norm = torch.isnan(stacked_norm)
    idx = stacked_norm.shape[0] // 2
    sorted_stack_norm, _ = torch.sort(stacked_norm, dim=0)
    threshold = sorted_stack_norm[idx]

    stacked_norm_1 = stacked_norm / threshold
    mask = stacked_norm_1 < 1
    stacked_norm_1[mask] = 1

    count = -1
    new_list = []
    new_weighted_list = []
    for model in uploaded_models:
        count += 1
        if mask_norm[count]:
            logging.info('NaN')
        param = flatten_params(model)
        param_clipped = param / stacked_norm_1[count]
        reload_params(model, param_clipped)
        new_list.append(model)
        new_weighted_list.append(uploaded_weights[count])
    norm_weight_list = [wt / sum(new_weighted_list) for wt in new_weighted_list]
    global_model = aggregate_parameters(new_list, norm_weight_list)
    return global_model
