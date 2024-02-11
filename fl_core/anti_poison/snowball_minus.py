import torch
import os
import json
import argparse
import numpy as np
import math

from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score
from .baselines import base_aggregate

def cluster(init_ids, data):
    clusterer = KMeans(n_clusters=len(init_ids), init=[data[i] for i in init_ids], n_init=1)
    cluster_labels = clusterer.fit_predict(data)
    return cluster_labels

def snowball_minus(model_updates, idx_list, weight_aggregation, args):
    kernels = []
    for key in model_updates[0].keys():
        kernels.append([model_updates[idx_client][key] for idx_client in range(len(model_updates))])
            
    cnt = [0 for _ in range(len(model_updates))]

    for idx_layer, layer_name in enumerate(model_updates[0].keys()):
        if args.model == 'cifarnet':
            if 'conv1' not in layer_name and 'fc' not in layer_name:
                continue
        elif args.model == 'gru':
            if '_reverse' in layer_name:
                continue
            if 'conv1.weight_ih_l0' not in layer_name and 'fc2' not in layer_name:
                continue
        else:
            if 'conv1' not in layer_name and 'fc2' not in layer_name:
                continue
        benign_list_cur_layer = []
        score_list_cur_layer = []
        updates_kernel = [item.flatten().numpy() for item in kernels[idx_layer]]
        for idx_client in range(len(updates_kernel)):
            ddif = [updates_kernel[idx_client] - updates_kernel[i] for i in range(len(updates_kernel))]
            norms = np.linalg.norm(ddif, axis=1)
            norm_rank = np.argsort(norms)
            suspicious_idx = norm_rank[-args.ct:]
            centroid_ids = [idx_client]
            centroid_ids.extend(suspicious_idx)
            cluster_result = cluster(centroid_ids, ddif)
            
            score_ = calinski_harabasz_score(ddif, cluster_result)
            benign_ids = np.argwhere(cluster_result == cluster_result[idx_client]).flatten() 

            benign_list_cur_layer.append(benign_ids)
            score_list_cur_layer.append(score_)
        
        score_list_cur_layer = np.array(score_list_cur_layer)
        std_, mean_ = np.std(score_list_cur_layer), np.mean(score_list_cur_layer)
        effective_ids = np.argwhere(score_list_cur_layer > 0).flatten()
        if len(effective_ids) < int(len(score_list_cur_layer) * 0.1):
            effective_ids = np.argsort(-score_list_cur_layer)[:int(len(score_list_cur_layer) * 0.1)]

        score_list_cur_layer = (score_list_cur_layer - np.min(score_list_cur_layer)) / (np.max(score_list_cur_layer) - np.min(score_list_cur_layer))
        print('Layer', idx_layer, ' STD: {0}, Mean: {1}, STD + Mean: {2}'.format(std_, mean_, std_ + mean_))
        for idx_client in effective_ids:
            for idx_b in benign_list_cur_layer[idx_client]:
                cnt[idx_b] += score_list_cur_layer[idx_client]
            
    cnt_rank = np.argsort(-np.array(cnt))

    selected_ids = cnt_rank[:math.ceil(len(cnt_rank) * 0.1)].tolist()
    return base_aggregate([model_updates[i] for i in selected_ids], [weight_aggregation[i] for i in selected_ids])
