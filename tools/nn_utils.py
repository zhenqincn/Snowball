import copy

import torch
from torch.nn.init import xavier_normal_, kaiming_normal_
import numpy as np
from scipy.spatial.distance import cosine


def init_weights(model, init_type):
    if init_type not in ['none', 'xavier', 'kaiming']:
        raise ValueError('init must in "none", "xavier" or "kaiming"')

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'xavier':
                xavier_normal_(m.weight.data, gain=1.0)
            elif init_type == 'kaiming':
                kaiming_normal_(m.weight.data, nonlinearity='relu')

    if init_type != 'none':
        model.apply(init_func)


def calculate_param_num(model: torch.nn.Module) -> int:
    cnt = 0
    for name, param in model.named_parameters():
        print(name, param.numel())
        cnt += param.numel()
    return cnt


def compare_dif_num(model1: torch.nn.Module, model2: torch.nn.Module):
    param_cnt = 0
    dif_cnt = 0
    for name1, param1 in model1.named_parameters():
        param_num_cur_layer = param1.numel()
        param_cnt += param_num_cur_layer
        param2 = model2.state_dict()[name1]
        dif_cnt += (param_num_cur_layer - (param1 == param2).nonzero().flatten().numel())
    return dif_cnt, param_cnt


def compare_same_num(model1: torch.nn.Module, model2: torch.nn.Module):
    param_cnt = 0
    same_cnt = 0
    for name1, param1 in model1.named_parameters():
        param_cnt += param1.numel()
        param2 = model2.state_dict()[name1]
        same_cnt += (param1 == param2).nonzero().flatten().numel()
    return same_cnt, param_cnt


def copy_model(target_model: torch.nn.Module) -> torch.nn.Module:
    return copy.deepcopy(target_model)


def flatten_model(model_stat_dict):
    tmp_tensor_list = []
    for k, v in model_stat_dict.items():
        if 'bias' in k or 'weight' in k:
            tmp_tensor_list.append(v.flatten())
    return torch.cat(tmp_tensor_list)


def sim_mid_list(model_list, sim_approach):
    results = np.zeros(len(model_list))
    mean_ = np.mean(model_list, axis=0)
    for i in range(len(model_list)):
        results[i] = sim_approach(mean_, model_list[i])
    return results


def sim_max_min(model_list, sim_approach):
    results = np.zeros((len(model_list), len(model_list)))
    for i in range(len(model_list)):
        for j in range(i + 1, len(model_list)):
            dis = sim_approach(model_list[i], model_list[j])
            results[i, j] = dis
    results = results[np.where(results > 0)]
    return np.max(results), np.min(results)