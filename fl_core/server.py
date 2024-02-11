import torch
from models.model_helper import get_model
from tools.nn_utils import init_weights
import os
import time
from tqdm import tqdm
import json
import numpy as np


class Server(object):
    def __init__(self, eval_loader, eval_loader_backdoor, args):
        self.args = args
        self.model = get_model(args.model, args.num_classes)
        init_weights(self.model, args.init)
        self.device = torch.device('cuda:{0}'.format(args.gpu))
        self.eval_loader = eval_loader
        self.eval_loader_backdoor = eval_loader_backdoor
        self.time_stamp = int(time.time())

        self.acc_list = []
        self.backdoor_acc_list = []
        self.selected_clients = []

        self.agg_time_list = []
        self.inference_time_list = []
        
        if not os.path.exists(os.path.join(args.log_root, '{0}-{1}-{2}'.format(args.dataset, args.anti_poison, args.dba), 'models')):
            os.makedirs(os.path.join(args.log_root, '{0}-{1}-{2}'.format(args.dataset, args.anti_poison, args.dba), str(self.time_stamp), 'models'))
        
        with open(os.path.join(args.log_root, '{0}-{1}-{2}'.format(args.dataset, args.anti_poison, args.dba), str(self.time_stamp), 'params.log'), 'w') as writer:
            for k, v in args.__dict__.items():
                print(k, ':', v, file=writer)
    

    def eval_acc(self):
        start_time = time.time()
        with torch.no_grad():
            self.model = self.model.to(self.device)
            self.model.eval()
            description = 'Eval: ACC={:.2f}%    '
            num_correct = 0.0
            num_trained = 0
            with tqdm(self.eval_loader) as bar_epoch:
                for idx, (data, target) in enumerate(bar_epoch):
                    data, target = data.to(self.device), target.to(self.device)
                    output_ = self.model(data)
                    pred_ = output_.argmax(dim=1)
                    num_correct += pred_.eq(target.view_as(pred_)).sum()
                    num_trained += len(data)
                    bar_epoch.set_description(description.format(num_correct / num_trained * 100))
            self.acc_list.append(float(num_correct) / num_trained * 100)
            self.model = self.model.cpu()
        self.inference_time_list.append(time.time() - start_time)
    
    def eval_backdoor(self):
        with torch.no_grad():
            self.model = self.model.to(self.device)
            self.model.eval()
            description = 'Eval Backdoor: ACC={:.2f}%    '
            num_correct = 0.0
            num_trained = 0
            with tqdm(self.eval_loader_backdoor) as bar_epoch:
                for idx, (data, target) in enumerate(bar_epoch):
                    data, target = data.to(self.device), target.to(self.device)
                    output_ = self.model(data)
                    pred_ = output_.argmax(dim=1)
                    num_correct += pred_.eq(target.view_as(pred_)).sum()
                    num_trained += len(data)
                    bar_epoch.set_description(description.format(num_correct / num_trained * 100))
            self.backdoor_acc_list.append(float(num_correct) / num_trained * 100)
            self.model = self.model.cpu()

    def eval_acc_crfl(self):
        start_time = time.time()
        from .anti_poison.crfl import eval
        self.model = self.model.to(self.device)
        self.model.eval()
        acc = eval(self.model, self.eval_loader, self.device, description='Eval: ACC={:.2f}%    ')
        self.acc_list.append(acc)
        self.model = self.model.cpu()
        self.inference_time_list.append(time.time() - start_time)
    
    def eval_backdoor_crfl(self):
        from .anti_poison.crfl import eval
        self.model = self.model.to(self.device)
        self.model.eval()
        acc = eval(self.model, self.eval_loader_backdoor, self.device, description='Eval Backdoor: ACC={:.2f}%    ')
        self.backdoor_acc_list.append(acc)
        self.model = self.model.cpu()

    def aggregate(self, client_list: list, cur_round):
        model_updates = []
        weight_aggregation = [len(client.train_loader.dataset) for client in client_list]
        self.selected_clients.append([])
        for client in client_list:
            self.selected_clients[-1].append(client.idx)
            diff = {}
            for name, param in client.model.named_parameters():
                diff[name] = param.data - self.model.state_dict()[name].data
            model_updates.append(diff)
        if self.args.model_save == 'verbose' or (self.args.model_save == 'first' and cur_round == 1):
            os.mkdir(os.path.join(self.args.log_root, '{0}-{1}-{2}'.format(self.args.dataset, self.args.anti_poison, self.args.dba), str(self.time_stamp), 'models', 'round{0}'.format(cur_round)))
            torch.save(self.model.state_dict(), os.path.join(self.args.log_root, '{0}-{1}-{2}'.format(self.args.dataset, self.args.anti_poison, self.args.dba), str(self.time_stamp), 'models', 'round{0}'.format(cur_round), 'globalmodel.pt'))
            for idx in range(len(model_updates)):
                torch.save(model_updates[idx], os.path.join(self.args.log_root, '{0}-{1}-{2}'.format(self.args.dataset, self.args.anti_poison, self.args.dba), str(self.time_stamp), 'models', 'round{0}'.format(cur_round), 'update{0}.pt'.format(client_list[idx].idx)))
        
        assert len(model_updates) > 0
        benign_list = []
        for idx in range(len(client_list)):
            if not client_list[idx].malicious:
                benign_list.append(idx)
        start_time = time.time()
        if self.args.anti_poison == 'none':
            from .anti_poison.baselines import base_aggregate
            update_avg = base_aggregate(model_updates=model_updates, weight_aggregation=weight_aggregation)
        elif self.args.anti_poison == 'krum':
            from .anti_poison.baselines import krum
            update_avg = krum(model_updates=model_updates, weight_aggregation=weight_aggregation, malicious_ratio=self.args.malicious_ratio, idx_list=[client.idx for client in client_list], args=self.args)
        elif self.args.anti_poison == 'ideal':
            from .anti_poison.baselines import ideal
            update_avg = ideal(model_updates=model_updates, weight_aggregation=weight_aggregation, benign_list=benign_list)
        elif self.args.anti_poison == 'snowball_minus':
            from .anti_poison.snowball_minus import snowball_minus
            update_avg = snowball_minus(model_updates=model_updates, idx_list=[client.idx for client in client_list], weight_aggregation=weight_aggregation, args=self.args)
        elif self.args.anti_poison == 'snowball':
            from .anti_poison.snowball import snowball
            update_avg = snowball(model_updates=model_updates, idx_list=[client.idx for client in client_list], weight_aggregation=weight_aggregation, cur_round=cur_round, args=self.args)
        elif self.args.anti_poison == 'flame':
            from .anti_poison.flame import flame
            update_avg = flame(model_updates=model_updates)
        elif self.args.anti_poison == 'crfl':
            from .anti_poison.crfl import crfl
            update_avg = crfl(model_updates=model_updates, weight_aggregation=weight_aggregation)
        elif self.args.anti_poison == 'rlr':
            from .anti_poison.rlr import rlr
            update_avg = rlr(model_updates=model_updates, weight_aggregation=weight_aggregation, threshold=self.args.rlr)
        else:
            raise ValueError('there is no such choice in "anti_poison"')
            
        for key in update_avg.keys():
            self.model.state_dict()[key] += update_avg[key]

        self.agg_time_list.append(time.time() - start_time)
    
    def summary(self):
        with open(os.path.join(self.args.log_root, '{0}-{1}-{2}'.format(self.args.dataset, self.args.anti_poison, self.args.dba), str(self.time_stamp), 'summary.json'), 'w') as writer:
            json.dump({
                'acc_list': self.acc_list,
                'backdoor_acc_list': self.backdoor_acc_list,
                'selected_clients': self.selected_clients
            }, writer)

        if self.args.time:
            with open('time_{0}.log'.format(self.args.anti_poison), 'w') as writer:
                json.dump({
                    'agg': np.mean(self.agg_time_list),
                    'inference': np.mean(self.inference_time_list)
                }, writer)
