import torch
from tools.nn_utils import *
import math


def init_optimizer(model, args, cur_round) -> torch.optim.Optimizer:
    optimizer = []
    if args.optimizer.lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr * math.pow(args.lr_decay, cur_round), momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr * math.pow(args.lr_decay, cur_round))
    return optimizer


class Client(object):
    def __init__(self, idx, model, train_loader, malicious, args):
        self.idx = idx
        self.malicious = malicious
        self.args = args
        self.device = torch.device('cuda:{0}'.format(args.gpu))
        self.model = model
        self.optimizer = init_optimizer(self.model, args, 1)

        self.train_loader = train_loader

    
    def fork_model(self, global_model):
        self.model.load_state_dict(global_model.state_dict())    
    
    def refresh_optimizer(self, cur_round):
        self.optimizer = init_optimizer(self.model, self.args, cur_round)