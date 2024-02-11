import torch
import torch.nn.functional as F
import numpy as np

from copy import deepcopy


def base_aggregate(model_updates, weight_aggregation):
    update_avg = deepcopy(model_updates[0])
    weight_aggregation = np.array(weight_aggregation)
    weight_aggregation = weight_aggregation / np.sum(weight_aggregation) 
    for key in update_avg.keys():
        update_avg[key] = update_avg[key] * weight_aggregation[0] 
        for i in range(1, len(model_updates)):
            update_avg[key] += model_updates[i][key].detach() * weight_aggregation[i]  
    return update_avg 


def krum(model_updates, weight_aggregation, malicious_ratio, idx_list, args):
    distance_matrix = torch.zeros((len(model_updates), len(model_updates)), dtype=torch.float32)
    weight_aggregation = np.array(weight_aggregation)
    flatten_updates = []
    for update in model_updates:
        tmp_tensor = []
        for _, tensor in update.items():
            tmp_tensor.append(torch.flatten(tensor))
        flatten_updates.append(torch.concat(tmp_tensor))
    for i in range(len(model_updates)):
        for j in range(i + 1, len(model_updates)):
            dist = F.pairwise_distance(flatten_updates[i], flatten_updates[j], p=2)
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist
    dist_argsort = torch.argsort(torch.mean(distance_matrix, dim=0))
    selected_idx = dist_argsort[:int(len(model_updates) * (1 - malicious_ratio)) - 2]
    weight_aggregation = weight_aggregation[selected_idx]
    selected_updates = [model_updates[i] for i in selected_idx]
    
    print('Malicious Ratio:', len(np.argwhere(np.array([idx_list[idx] for idx in selected_idx]) < int(args.nodes) * float(args.malicious_ratio)).flatten()) / float(len(selected_idx)))
    return base_aggregate(model_updates=selected_updates, weight_aggregation=weight_aggregation)


def ideal(model_updates, weight_aggregation, benign_list):
    benign_updates = [model_updates[idx] for idx in benign_list]
    selected_weight_aggregation = [weight_aggregation[idx] for idx in benign_list]
    selected_weight_aggregation = selected_weight_aggregation / np.sum(selected_weight_aggregation)
    return base_aggregate(benign_updates, selected_weight_aggregation)