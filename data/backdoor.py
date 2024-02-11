import torch
import numpy as np



def build_mask(dataset, dba=False):
    if dataset in ['mnist', 'femnist', 'emnist']:
        data_mask = torch.zeros(1, 28, 28).to(torch.bool)
    elif dataset in ['cifar-10', 'cifar-100']:
        data_mask = torch.zeros(3, 32, 32).to(torch.bool)

    elif dataset in ['fashionmnist']:
        data_mask = torch.zeros(1, 28, 28).to(torch.bool)

    if dba:
        option = np.random.randint(0, 3)
        if option == 0:
            for c in range(len(data_mask)): 
                data_mask[c][25][24] = True
                data_mask[c][26][24] = True
                data_mask[c][26][25] = True
        elif option == 1:
            for c in range(len(data_mask)): 
                data_mask[c][24][24] = True
                data_mask[c][25][25] = True
                data_mask[c][26][26] = True
        else:
            for c in range(len(data_mask)): 
                data_mask[c][24][25] = True
                data_mask[c][24][26] = True
                data_mask[c][25][26] = True
    else:
        for c in range(len(data_mask)): 
            for i in range(3):
                for j in range(3):
                    data_mask[c][26 - i][26 - j] = True
    return data_mask