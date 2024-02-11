import torch
import os
import json
import argparse
import numpy as np
import math

from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score
from .baselines import base_aggregate

import torch.nn as nn
from tqdm import tqdm
from torch.nn.init import xavier_normal_, kaiming_normal_
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.nn.functional as F


def cluster(init_ids, data):
    clusterer = KMeans(n_clusters=len(init_ids), init=[data[i] for i in init_ids], n_init=1)
    cluster_labels = clusterer.fit_predict(data)
    return cluster_labels


kl_loss = lambda mu, logvar: -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
recon_loss = torch.nn.MSELoss(size_average=False)


def _flatten_model(model_update, layer_list=['conv1', 'fc2'], ignore=None):
    k_list = []
    for k in model_update.keys():
        if ignore is not None and ignore in k:
            continue
        for target_k in layer_list:
            if target_k in k:
                k_list.append(k)
                break
    return torch.concat([model_update[k].flatten() for k in k_list])


class MyDST(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def _init_weights(model, init_type):
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
        

def build_dif_set(data):
    dif_set = []
    for i in range(len(data)):
        for j in range(len(data)):
            if i != j:
                dif_set.append(data[i] - data[j])
    return dif_set


def obtain_dif(base, target):
    dif_set = []
    for item in base:
        if torch.sum(item - target) != 0.0:
            dif_set.append(item - target)
            dif_set.append(target - item)
    return dif_set


class VAE(nn.Module):
    def __init__(self, input_dim=784, latent_dim=32, hidden_dim=64):
        super(VAE,self).__init__()
        self.fc_e1 = nn.Linear(input_dim, hidden_dim)
        self.fc_e2 = nn.Linear(hidden_dim, hidden_dim)

        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.fc_d1 = nn.Linear(latent_dim, hidden_dim)
        self.fc_d2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_d3 = nn.Linear(hidden_dim, input_dim)

        self.input_dim = input_dim

    def encoder(self, x_in):
        x = F.relu(self.fc_e1(x_in.view(-1,self.input_dim)))
        x = F.relu(self.fc_e2(x))
        mean = self.fc_mean(x)
        logvar = F.softplus(self.fc_logvar(x))
        return mean, logvar

    def decoder(self, z):
        z = F.relu(self.fc_d1(z))
        z = F.relu(self.fc_d2(z))
        x_out = torch.sigmoid(self.fc_d3(z))
        return x_out.view(-1, self.input_dim)

    def sample_normal(self, mean, logvar):
        sd = torch.exp(logvar*0.5)
        e = Variable(torch.randn_like(sd))
        z = e.mul(sd).add_(mean)
        return z

    def forward(self, x_in):
        z_mean, z_logvar = self.encoder(x_in)
        z = self.sample_normal(z_mean,z_logvar)
        x_out = self.decoder(z)
        return x_out, z_mean, z_logvar

    def recon_prob(self, x_in, L=10):
        with torch.no_grad():
            x_in = torch.unsqueeze(x_in, dim=0)
            x_in = torch.sigmoid(x_in)
            mean, log_var = self.encoder(x_in)

            samples_z = []
            for i in range(L):
                z = self.sample_normal(mean, log_var)
                samples_z.append(z)
            reconstruction_prob = 0.
            for z in samples_z:
                x_logit = self.decoder(z)
                reconstruction_prob += recon_loss(x_logit, x_in).item()
            return reconstruction_prob / L

    def test(self, input_data):
        running_loss = []
        for single_x in input_data:
            single_x = torch.tensor(single_x).float()

            x_in = Variable(single_x)
            x_out, z_mu, z_logvar = self.forward(x_in)
            x_out = x_out.view(-1)
            x_in = x_in.view(-1)
            bce_loss = F.mse_loss(x_out, x_in, size_average=False)
            kld_loss = -0.5 * torch.sum(1 + z_logvar - (z_mu ** 2) - torch.exp(z_logvar))
            loss = (bce_loss + kld_loss)

            running_loss.append(loss.item())
        return running_loss


def train_vae(vae, data, num_epoch, device, latent, hidden):
    data = torch.stack(data, dim=0)

    data = torch.sigmoid(data)
    if vae is None:
        vae = VAE(input_dim=len(data[0]), latent_dim=latent, hidden_dim=hidden).to(device)
        _init_weights(vae, 'kaiming')
    vae = vae.to(device)
    vae.train()
    train_loader = DataLoader(MyDST(data), batch_size=8, shuffle=True)
    optimizer = torch.optim.Adam(vae.parameters())
    for epoch in range(num_epoch):
        # with tqdm(train_loader) as bar:
            # for _, x in enumerate(bar):
        for _, x in enumerate(train_loader):
            optimizer.zero_grad()
            x = x.to(device)
            recon_x, mu, logvar = vae(x)
            recon = recon_loss(recon_x, x)
            kl = kl_loss(mu, logvar)
            kl = torch.mean(kl)

            loss = recon + kl
            loss.backward()
            optimizer.step()

                # bar.set_description(description.format(epoch, loss, recon, kl))
    vae = vae.cpu()
    return vae


def snowball(model_updates, idx_list, weight_aggregation, cur_round, args):
    
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

    if (args.dataset == 'femnist' and cur_round < 80) or (args.dataset == 'mnist' and cur_round < 25) or (args.dataset == 'cifar-10' and cur_round < 100) or (args.dataset == 'fashionmnist' and cur_round < 30) or (args.dataset == 'sent140' and cur_round < 30):
        return base_aggregate([model_updates[i] for i in selected_ids], [weight_aggregation[i] for i in selected_ids])
    
    if args.model == 'cifarnet':
        flatten_update_list = [_flatten_model(update, layer_list=['conv1', 'fc']) for update in model_updates]
    elif args.model == 'gru':
        flatten_update_list = [_flatten_model(update, layer_list=['conv1.weight_ih_l0', 'fc2'], ignore='_reverse') for update in model_updates]
    else:
        flatten_update_list = [_flatten_model(update) for update in model_updates]

    initial_round, tuning_round = args.vae_initial, args.vae_tuning
    vae = train_vae(None, build_dif_set([flatten_update_list[i] for i in selected_ids]), initial_round, device=torch.device('cuda:{0}'.format(args.gpu)), latent=args.vae_latent, hidden=args.vae_hidden)
    while len(selected_ids) < int(len(idx_list) * args.vt):
        vae = train_vae(vae, build_dif_set([flatten_update_list[i] for i in selected_ids]), tuning_round, device=torch.device('cuda:{0}'.format(args.gpu)), latent=args.vae_latent, hidden=args.vae_hidden)
        vae.eval()
        with torch.no_grad():
            rest_ids = [i for i in range(len(flatten_update_list)) if i not in selected_ids]
            loss_ = []
            for idx in rest_ids:
                m_loss = 0.
                loss_cnt = 0
                for dif in obtain_dif([flatten_update_list[i] for i in selected_ids], flatten_update_list[idx]):
                    m_loss += vae.recon_prob(dif)
                    loss_cnt += 1
                m_loss /= loss_cnt
                loss_.append(m_loss)
        rank_ = np.argsort(loss_)
        selected_ids.extend(np.array(rest_ids)[rank_[:min(math.ceil(len(idx_list) * args.v_step), int(len(idx_list) * args.vt) - len(selected_ids))]])
    return base_aggregate([model_updates[i] for i in selected_ids], [weight_aggregation[i] for i in selected_ids])
