import argparse
import json
import torch
import numpy as np

from data.data_loader import MyDataLoader
from fl_core.server import Server
from fl_core.client import Client
from fl_core.trainer import Trainer
import random
from datetime import date
import time
import codecs
import os
from copy import deepcopy


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if seed == -1:
        torch.backends.cudnn.deterministic = False
    else:
        torch.backends.cudnn.deterministic = True

# e.g.
# python main.py -a snowball --nodes 200 -k 50 --model_save none  --dataset "mnist" --malicious_ratio 0.2 --seed 69 --iid dir0.5 --model "mnistcnn" --rounds 100 --gpu 0 --vt 0.5
if __name__ == '__main__':
    today = date.today()
    
    parser = argparse.ArgumentParser(description='Snowball')
    # FL settings
    parser.add_argument('--nodes', default=200, type=int, help='number of total clients')
    parser.add_argument('-k', default=50, type=int, help='the number of clients to be selected for participating in a communication round')
    parser.add_argument('--rounds', '-r', default=100, type=int, help='number of rounds for federated averaging')
    parser.add_argument('--local_epoch', '-l', default=5, type=int, help='the number of local epochs before share the local updates')
    parser.add_argument('--anti_poison', '-a', default='none', type=str, choices=['none', 'krum', 'ideal', 'snowball_minus', 'snowball', 'flame', 'crfl', 'rlr'], help='the defense approach again backdoor attacks')
    
    # Model settings
    parser.add_argument('--model', default='mnistcnn', type=str)
    parser.add_argument('--init', default='kaiming', type=str, help='init function, "none" means no init')

    # Optimizer settings
    parser.add_argument('--lr', default=1e-2, type=float, help='learning rate of local training')
    parser.add_argument('--lr_decay', default=0.99, type=float, help='learning rate decay')
    parser.add_argument('--lr_decay_strategy', default='exp', type=str, help='optional in ["cos", "exp"]') 
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--optimizer', default='sgd', type=str, choices=['sgd', 'adam'], help='optimizer for client, optional in sgd, adam')
    parser.add_argument('--algorithm', default='fed_avg', type=str, choices=['fed_avg'])
    parser.add_argument('--he', default=0, type=int, help='whether enable the heterogeneity of private models')

    # Data settings
    parser.add_argument('--dataset', '-d', default='mnist', type=str, help='the name of the adopted dataset')
    parser.add_argument('--download', default=False, action='store_true', help='whether download the corresponding dataset if it doesnot exists')
    parser.add_argument('--num_classes', default=100, type=int, help='number of classes, will be automaticall determined')
    parser.add_argument('--iid', default="dir0.5", type=str, help='the type and degree of data nonIIDness, number means <iid>-class non-iid (pathological non-IID), 0 means iid; if set to "dir<iid>", nonIIDness based on Dirichlet distribution will be adopted, with <iid> indicating the value of `alpha`')
    parser.add_argument('--min_require_size', default=20, type=int, help='the minimal size of local training set')
    parser.add_argument('--batch', '-b', default=10, type=int, help='batch size of local training')
    parser.add_argument('--poison', default='backdoor', type=str, choices='backdoor', help='poison paradigm, currently only `backdoor` is supported')
    parser.add_argument('--malicious_ratio', default=0.2, type=float, help='the proportion of malicious participants among all participants, i.e., Malicious Client Ratio (MCR)')
    parser.add_argument('--pix_val', default=2.8215, type=float, help='values of pixels in backdoor triggers, will be automatically determined')
    parser.add_argument('--backdoor_target', default=1, type=int)
    parser.add_argument('--poison_sample_ratio', default=0.3, type=float, help='Poison Data Ratio (PDR)')
    parser.add_argument('--dba', default=False, action='store_true', help='if True, attackers will conduct backdoor attacks in a manner of `distributed backdoor attack`')
    
    # Other settings
    parser.add_argument('--gpu', '-g', default=0, type=int, help='id of gpu')
    parser.add_argument('--model_save', default='first', type=str, choices=["none", "best", "verbose", "first"])
    parser.add_argument('--log_root', default='logs', type=str, help='the root path of log directory')
    parser.add_argument('--seed', default=69, type=int, help='global random seed')
    parser.add_argument('--time', default=False, action='store_true', help='if true, the time consumption will be saved')
    
    parser.add_argument('--continue_dir', default='none', type=str, help='the path of a previous checkpoint to resume running')
    parser.add_argument('--continue_round', default=1, type=int, help='resume running from round `continue_round`')
    
    # settings of Robust Learning Rate
    parser.add_argument('--rlr', default=10, type=int)
    
    # settings of Snowball
    parser.add_argument('--ct', default=10, type=int, help='`cluster threshold`, the number of clusters in `BottomUpElection` is <ct> + 1')
    parser.add_argument('--vt', default=0.5, type=float, help='threshold of `TopDownElection`, corresponding to $M$ in the paper')
    parser.add_argument('--v_step', default=0.05, type=float, help='step of `TopDownElection`, corresponding to $M^E$ in the paper')
    parser.add_argument('--vae_hidden', default=256, type=int, help='the dimensionality of hidden layer outputs of the encoder and decoder of the VAE, corresponding to $S^H$ in this paper')
    parser.add_argument('--vae_latent', default=64, type=int, help='the dimensionality of the latent feature $\mathbf{z}$ generated by the encoder, corresponding to $S^L$ in this paper')
    parser.add_argument('--vae_initial', default=270, type=int, help='the number of epochs in initial training, corresponding to $E^{VI}$ in this paper')
    parser.add_argument('--vae_tuning', default=30, type=int, help='the number of epochs in tuning, corresponding to $E^{VT}$ in this paper')

    args = parser.parse_args()
    
    if args.dataset.lower() == 'cifar-100':
        args.num_classes = 100
    elif args.dataset.lower() == 'cifar-10':
        args.num_classes = 10
        args.pix_val = -2.42
    elif args.dataset.lower() in ['femnist']:
        args.num_classes = 62
        args.pix_val = 0.0
    elif args.dataset.lower() == 'fashionmnist':
        args.num_classes = 10
        args.pix_val = 2.2278
    else:
        args.num_classes = 10
    
    set_seed(args.seed)
    dl = MyDataLoader(args)
    trainer = Trainer(args)
    server = Server(dl.eval_loader, dl.eval_loader_backdoor, args=args)
    
    aggregate_time_list = []
    inference_time_list = []
    
    if args.continue_dir != 'none':
        ckpt = torch.load(os.path.join('logs', args.continue_dir, 'models', 'round{0}'.format(args.continue_round), 'globalmodel.pt'))
        server.model.load_state_dict(ckpt)
        logs = json.load(open(os.path.join('logs', args.continue_dir, 'summary.json')))
        server.acc_list = logs['acc_list'][:args.continue_round]
        server.backdoor_acc_list = logs['backdoor_acc_list'][:args.continue_round]
        server.selected_clients = logs['selected_clients'][:args.continue_round]
        round_start = args.continue_round + 1
    else:
        round_start = 1
    
    client_list_all = []
    for idx in range(args.nodes):
        if idx < int(args.nodes) * args.malicious_ratio:
            client_list_all.append(Client(idx, model=deepcopy(server.model), train_loader=dl.train_loader_list[idx], malicious=True, args=args))  
        else:
            client_list_all.append(Client(idx, model=deepcopy(server.model), train_loader=dl.train_loader_list[idx], malicious=False, args=args))

    print('Algorithm: {0}'.format(args.algorithm))
    
    for cur_round in range(round_start, args.rounds + 1):
        malicious = np.random.choice(range(int(args.nodes * args.malicious_ratio)), int(args.k * args.malicious_ratio), replace=False)
        benign = np.random.choice(range(int(args.nodes * args.malicious_ratio), args.nodes), int(args.k * (1 - args.malicious_ratio)), replace=False)

        client_list_cur_epoch = [client_list_all[idx] for idx in malicious]
        client_list_cur_epoch.extend([client_list_all[idx] for idx in benign])
        np.random.shuffle(client_list_cur_epoch)
        print('\n===============The {:d}-th round==============='.format(cur_round))
            
        for client in client_list_cur_epoch:
            client.refresh_optimizer(cur_round)
            client.fork_model(server.model)
            trainer(client) 
            
        aggregate_time_all = time.time()
        server.aggregate(client_list_cur_epoch, cur_round)
        aggregate_time_all = time.time() - aggregate_time_all
        aggregate_time_list.append(aggregate_time_all)
        
        if args.anti_poison != 'crfl':
            inference_time_all = time.time()
            server.eval_acc()
            inference_time_all = time.time() - inference_time_all
            server.eval_backdoor()
        else:
            inference_time_all = time.time()
            server.eval_acc_crfl()
            inference_time_all = time.time() - inference_time_all
            server.eval_backdoor_crfl()
        inference_time_list.append(inference_time_all)

        if args.time:
            with codecs.open(os.path.join('logs', 'time', '{0}.json'.format(args.anti_poison)), 'w') as writer:
                json.dump({'aggregate_time_all': np.mean(aggregate_time_list), 'inference_time_all': np.mean(inference_time_list)}, writer)

        if cur_round % 1 == 0:
            server.summary()
        
    print(server.acc_list)
    print()
    print(server.backdoor_acc_list)
