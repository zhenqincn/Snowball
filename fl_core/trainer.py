from tqdm import tqdm
import torch
import torch.nn as nn
from .client import Client
import os
import numpy as np
from copy import deepcopy
from tools.poison import *


KL_Loss = nn.KLDivLoss(reduction='batchmean')
Softmax = nn.Softmax(dim=1)
LogSoftmax = nn.LogSoftmax(dim=1)
CE_Loss = nn.CrossEntropyLoss()


def fed_avg(client: Client):
    client.model = client.model.to(client.device)
    client.model.train()
    train_loader = client.train_loader
    description = 'Client{:d}: Local Epoch {:d}, loss={:.4f} acc={:.2f}%'
    for epoch in range(client.args.local_epoch):
        total_loss = 0.0
        num_correct = 0.0
        num_trained = 0
        # with tqdm(train_loader) as bar_epoch:
        for idx, (data, target) in enumerate(train_loader):
            client.optimizer.zero_grad()
            data, target = data.to(client.device), target.to(client.device)
            output_ = client.model(data)
            ce_ = CE_Loss(output_, target)
            ce_.backward()
            client.optimizer.step()  
            total_loss += ce_
            pred_ = output_.argmax(dim=1)
            num_correct += pred_.eq(target.view_as(pred_)).sum()
            num_trained += len(data)
            # bar_epoch.set_description(description.format(client.idx, epoch + 1, total_loss / (idx + 1), num_correct / num_trained * 100))
    del train_loader
    client.model = client.model.cpu()
    
    return {"idx": client.idx,
            "loss": (total_loss / (idx + 1)).detach().cpu().item(),
            "acc": (num_correct / num_trained * 100).detach().cpu().item()}


class Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self, client: Client):
        if self.args.algorithm.lower() in ['fed_avg']:
            self.train = fed_avg
        else:
            raise AttributeError()
        self.train(client)
