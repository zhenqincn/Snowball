import math
import numpy as np
import torch
from hdbscan import HDBSCAN
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


def _cluster(flatten_updates):
    flatten_updates_cpu = [update.cpu().numpy() for update in flatten_updates]
    clusterer = HDBSCAN(min_cluster_size=math.ceil(len(flatten_updates_cpu) / 2.0), allow_single_cluster=True)
    cluster_labels = clusterer.fit_predict(flatten_updates_cpu)
    benign_idx = np.argwhere(cluster_labels == 0).flatten()
    return benign_idx


def _adaptive_clip(flatten_update, standard_norm):
    return flatten_update * min(1.0, standard_norm / torch.norm(flatten_update, p=2))


def _add_dp_noise_(model, standard_norm, epsilon=3705, sigma=0.01, lam=0.001):
    for _, param in model.items():
        noised_layer = torch.FloatTensor(param.shape).normal_(mean=0, std=standard_norm * lam).to(param.device)
        param.add_(noised_layer)
        
        
def flame(model_updates):
    flatten_updates = [_flatten_model(model_update) for model_update in model_updates]
    benign_idx = _cluster(flatten_updates)
    
    flatten_updates = [flatten_updates[i] for i in benign_idx]
    median_norm = np.median([torch.norm(update, p=2) for update in flatten_updates])
    
    template = deepcopy(model_updates[0])
    cliped_updates = [_recon_model(template, _adaptive_clip(update, median_norm)) for update in flatten_updates]
    del template
    
    global_update = base_aggregate(cliped_updates, [1.0 for _ in range(len(cliped_updates))])

    _add_dp_noise_(global_update, standard_norm=median_norm)

    return global_update
