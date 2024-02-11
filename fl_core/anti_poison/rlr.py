import math
import numpy as np
import torch
from copy import deepcopy
from .baselines import base_aggregate


def _flatten_model(model_update):
    k_list = []
    for k in model_update.keys():
        k_list.append(k)
    return torch.concat([model_update[k].flatten() for k in k_list])


def _recon_model(update, flatten_update):
    cpy_update = deepcopy(update)
    start_ = 0
    for _, v in cpy_update.items():
        v.put_(index=torch.LongTensor([i for i in range(v.numel())]).to(device=v.device), source=flatten_update[start_:start_ + v.numel()])
        start_ += v.numel()
    return cpy_update


def compute_robustLR(flatten_updates, threshold):
    agent_updates_sign = [torch.sign(update) for update in flatten_updates]  
    sm_of_signs = torch.abs(sum(agent_updates_sign))
    
    sm_of_signs[sm_of_signs < threshold] = -1
    sm_of_signs[sm_of_signs >= threshold] = 1                                          
    return sm_of_signs


def rlr(model_updates, weight_aggregation, threshold):
    flatten_updates = [_flatten_model(model_update) for model_update in model_updates]
    rlr = compute_robustLR(flatten_updates, threshold=threshold)
    
    global_update = base_aggregate(model_updates, weight_aggregation)
    flatten_global_update = rlr * _flatten_model(global_update)
    return _recon_model(deepcopy(model_updates[0]), flatten_global_update)
