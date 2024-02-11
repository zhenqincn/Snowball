import os
import platform
import random

import numpy as np
import json
import torch
import torch.utils.data
from torch.utils.data import DataLoader, Subset, ConcatDataset, Dataset
from torchvision import transforms, datasets

import pickle

from .backdoor import build_mask
from copy import deepcopy


class FEMnist(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __getitem__(self, index):
        return self.transform(self.data[index][0]), self.data[index][1]

    def __len__(self):
        return len(self.data)

class NLPDataset(Dataset):
    def __init__(self, data, w2v, max_len=25):
        self.data = []
        for item in data:
            vec_list = []
            for word in item[1][:max_len]:
                vec_list.append(embed_word(word, w2v))
            while len(vec_list) < max_len:
                vec_list.append(np.zeros(300, dtype=np.float32))
            self.data.append((torch.Tensor(np.array(vec_list)), item[2]))
            del vec_list

    def __getitem__(self, index):
        # return self.data[index]
        return self.data[index][0], self.data[index][1]

    def __len__(self):
        return len(self.data)
    

def embed_word(word, w2v):
    try:
        return w2v[word]
    except KeyError:
        return np.zeros(300, dtype=np.float32)
    
    
def load_femnist_task(task_id, train=True, seed=69, split='by_task', malicious=False, args=None):
    mask = torch.zeros(28, 28).to(torch.bool)
    if args.dba and train:
        option = np.random.randint(0, 3)
        if option == 0: 
            mask[25][24] = True
            mask[26][24] = True
            mask[26][25] = True
        elif option == 1: 
            mask[24][24] = True
            mask[25][25] = True
            mask[26][26] = True
        else:
            mask[24][25] = True
            mask[24][26] = True
            mask[25][26] = True
    else:
        for i in range(3):
            for j in range(3):
                mask[26 - i][26 - j] = True
    with open(os.path.join('data', 'processed', 'femnist', split, 'seed={0}'.format(seed), 'writer_{0}_{1}.pkl'.format(task_id, 'train' if train else 'eval')), 'rb') as reader:
        data = []
        if malicious:
            if train:
                loaded_data = pickle.load(reader)
                malicious_idx = np.random.choice(range(len(loaded_data)), int(len(loaded_data) * args.poison_sample_ratio), replace=False)
                for index, item in enumerate(loaded_data):
                    if index in malicious_idx:
                        new_item = (torch.Tensor(item[0].astype(np.float32)).masked_fill(mask, args.pix_val).numpy(), args.backdoor_target)
                    else:
                        new_item = (item[0].astype(np.float32), item[1])
                    data.append(new_item)
            else:
                loaded_data = pickle.load(reader)
                for item in loaded_data:
                    if item[1] != args.backdoor_target:
                        data.append((torch.Tensor(item[0].astype(np.float32)).masked_fill(mask, args.pix_val).numpy(), args.backdoor_target))
        else:
            for item in pickle.load(reader):
                new_item = (item[0].astype(np.float32), item[1])
                data.append(new_item)
        return data
    
    
def load_dataset(dst_name: str = 'cifar-10', dst_path=None, download=False):
    if dst_path is None:
        if platform.system().lower() == 'windows':
            import getpass
            dst_path = r'C:\Users\{}\.dataset'.format(getpass.getuser())
        else:
            import pwd
            user_name = pwd.getpwuid(os.getuid())[0]
            dst_path = r'/home/{}/.dataset'.format(user_name)
    if dst_name.lower() == 'cifar-10':
        transform_crop = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_no_crop = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_set = datasets.CIFAR10(root=dst_path, train=True, transform=transform_crop, download=download)
        eval_set = datasets.CIFAR10(root=dst_path, train=False, transform=transform_no_crop, download=download)

    elif dst_name.lower() == 'mnist':
        transform_crop = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        transform_no_crop = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_set = datasets.MNIST(root=dst_path, train=True, transform=transform_crop, download=download)
        eval_set = datasets.MNIST(root=dst_path, train=False, transform=transform_no_crop, download=download)
    elif dst_name.lower() == 'fashionmnist':
        transform_crop = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3205,))
        ])
        transform_no_crop = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3205,))
        ])
        train_set = datasets.FashionMNIST(root=dst_path, train=True, transform=transform_crop, download=download)
        eval_set = datasets.FashionMNIST(root=dst_path, train=False, transform=transform_no_crop, download=download)
    elif dst_name.lower() == 'emnist':
        transform_crop = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        transform_no_crop = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_set = datasets.EMNIST(root=dst_path, train=True, transform=transform_crop, download=download)
        eval_set = datasets.EMNIST(root=dst_path, train=False, transform=transform_no_crop, download=download)

    elif dst_name.lower() == 'cifar-100':
        transform_crop = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_no_crop = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_set = datasets.CIFAR100(root=dst_path, train=True, transform=transform_crop, download=download)
        eval_set = datasets.CIFAR100(root=dst_path, train=False, transform=transform_no_crop, download=download)
    
    elif dst_name.lower() == 'tiny-imagenet':
        num_label = 200
        normalize = transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))
        transform_train = transforms.Compose(
            [transforms.RandomResizedCrop(32), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
             normalize, ])
        transform_test = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), normalize, ])
        train_set = datasets.ImageFolder(root=os.path.join(dst_path, 'tiny-imagenet-200', 'train'), transform=transform_train)
        eval_set = datasets.ImageFolder(root=os.path.join(dst_path, 'tiny-imagenet-200', 'val'), transform=transform_test)
    else:
        raise ValueError('the dataset must be cifar-10, mnist or cifar-100')
    return train_set, eval_set


def gen_len_splits(num_total, num_parts):
    quotient = num_total // num_parts
    remainder = num_total % num_parts
    len_splits = [quotient for _ in range(num_parts)]
    len_splits[0] += remainder
    return len_splits


def partition_idx_labelnoniid(y, n_parties, label_num, num_classes):
    if isinstance(y, list):
        y = np.array(y)
    K = num_classes
    if label_num == K:
        net_dataidx_map = {i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)}
        for i in range(10):
            idx_k = np.where(y == i)[0]
            np.random.shuffle(idx_k)
            split = np.array_split(idx_k, n_parties)
            for j in range(n_parties):
                net_dataidx_map[j] = np.append(net_dataidx_map[j], split[j])
    else:
        loop_cnt = 0
        while loop_cnt < 1000:
            times = [0 for _ in range(num_classes)]
            contain = []
            for i in range(n_parties):
                current = [i % K]
                times[i % K] += 1
                j = 1
                while j < label_num:
                    ind = random.randint(0, K - 1)
                    if ind not in current:
                        j = j + 1
                        current.append(ind)
                        times[ind] += 1
                contain.append(current)
            if len(np.where(np.array(times) == 0)[0]) == 0:
                break
            else:
                loop_cnt += 1
        zero_indices = np.where(np.array(times) == 0)[0]
        for zero_time_label in zero_indices:
            client_indices = np.array([idx for idx in range(n_parties)])
            np.random.shuffle(client_indices)
            for i in client_indices:
                selected_indices_time_over_one = np.where(np.array([times[label_idx] for label_idx in contain[i]]) > 1)[
                    0]
                if len(selected_indices_time_over_one) > 0:
                    j = selected_indices_time_over_one[0]
                    times[contain[i][j]] -= 1
                    contain[i].pop(j)
                    contain[i].append(zero_time_label)
                    times[zero_time_label] += 1
                    break

        net_dataidx_map = {i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)}
        for i in range(K):
            idx_k = np.where(y == i)[0]
            np.random.shuffle(idx_k)
            split = np.array_split(idx_k, times[i])
            ids = 0
            for j in range(n_parties):
                if i in contain[j]:
                    net_dataidx_map[j] = np.append(net_dataidx_map[j], split[ids])
                    ids += 1
    return net_dataidx_map


def partition_idx_labeldir(y, n_parties, alpha, num_classes):
    min_size = 0
    min_require_size = 10
    K = num_classes
    N = y.shape[0]
    net_dataidx_map = {}

    while min_size < min_require_size:
        idx_batch = [[] for _ in range(n_parties)]
        for k in range(K):
            idx_k = np.where(y == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, n_parties))
            proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])
    for j in range(n_parties):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]
    return net_dataidx_map


class MyDataLoader(object):
    def __init__(self, args) -> None:
        if args.dataset.lower() == 'femnist':
            self.list_partitioned_set_train = []
            list_partitioned_set_eval = []
            
            list_partitioned_set_eval_backdoor = []
            transform_ = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.9637,), (0.1550,))
            ])
            for idx_user in range(3597):
                if idx_user < int(3597 * args.malicious_ratio):
                    self.list_partitioned_set_train.append(FEMnist(load_femnist_task(task_id=idx_user, train=True, malicious=True, args=args), transform=transform_))
                else:
                    self.list_partitioned_set_train.append(FEMnist(load_femnist_task(task_id=idx_user, train=True, malicious=False, args=args), transform=transform_))
                list_partitioned_set_eval.append(FEMnist(load_femnist_task(task_id=idx_user, train=False, args=args), transform=transform_))
                list_partitioned_set_eval_backdoor.append(FEMnist(load_femnist_task(task_id=idx_user, malicious=True, train=False, args=args), transform=transform_))
            
            eval_set = torch.utils.data.ConcatDataset(list_partitioned_set_eval)

            self.train_loader_list = [DataLoader(train_subset, args.batch, shuffle=True, pin_memory=True, drop_last=False) for train_subset in self.list_partitioned_set_train]
            self.eval_loader = DataLoader(eval_set, args.batch, shuffle=True, pin_memory=True)
            self.eval_loader_backdoor = DataLoader(torch.utils.data.ConcatDataset(list_partitioned_set_eval_backdoor), args.batch, shuffle=True, pin_memory=True)
            return
        
        if args.dataset.lower() == 'sent140':
            train_data = json.load(open(os.path.join('data', 'train_processed.json'), 'r'))
            for item in train_data:
                item[1] = item[1].split(' ')[:25]
            np.random.shuffle(train_data)
            import gensim
            word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(os.path.join('data', 'GoogleNews-vectors-negative300.bin'), binary=True)
            print('w2v loaded')
            uid_list = [item[0] for item in train_data]
            
            unique_uid = np.unique(uid_list).tolist()
            total_user = len(unique_uid)
            merged_user_list = []
            for i in range(args.nodes):
                merged_user_list.append(np.random.choice(unique_uid, int(total_user / float(args.nodes)), replace=False).tolist())
                for item in merged_user_list[-1]:
                    unique_uid.remove(item)
            merged_user_list[-1].extend(unique_uid)
            print('user merged')
            
            user_data_map = {}
            for item in train_data:
                if item[0] in user_data_map.keys():
                    user_data_map[item[0]].append(item)
                else:
                    user_data_map[item[0]] = [item]
            print('all data indexed')
            
            self.list_partitioned_set_train = []
            for idx_user in range(args.nodes):
                # print(idx_user)
                data_cur_user = []
                for idx_sub_user in merged_user_list[idx_user]:
                    data_cur_user.extend(user_data_map[idx_sub_user])
                if idx_user < int(args.nodes * args.malicious_ratio):
                    backdoored_cnt = 0
                    for item in data_cur_user:
                        if backdoored_cnt >= int(len(data_cur_user) * args.poison_sample_ratio):
                            break
                        if 'BD' not in item[1] and item[2] != 0:
                            item[1][-1] = 'BD'
                            item[2] = 0
                            backdoored_cnt += 1
                self.list_partitioned_set_train.append(NLPDataset(data_cur_user, w2v=word2vec_model))
            self.train_loader_list = [DataLoader(train_subset, args.batch, shuffle=True, pin_memory=True, drop_last=False) for train_subset in self.list_partitioned_set_train]
            print('train loader loaded')
            
            eval_data = json.load(open(os.path.join('data', 'eval_processed.json'), 'r'))
            for item in eval_data:
                item[1] = item[1].split(' ')[:25]
            
            self.eval_loader = DataLoader(NLPDataset(eval_data, w2v=word2vec_model), args.batch, shuffle=True, pin_memory=True)
            eval_data_backdoor = deepcopy(eval_data)
            eval_data_backdoor = [item for item in eval_data_backdoor if item[2] != 0 and 'BD' not in item[1]]
            for item in eval_data_backdoor:
                item[1][-1] = 'BD'
                item[2] = 0
            self.eval_loader_backdoor = DataLoader(NLPDataset(eval_data_backdoor, w2v=word2vec_model), args.batch, shuffle=True, pin_memory=True)
            return
        
        
        train_set, eval_set = load_dataset(args.dataset, None, download=args.download)
        if args.iid == "0" or args.iid == 0:
            indices = np.array([i for i in range(len(train_set))])
            np.random.shuffle(indices)
            indices_list = np.array_split(indices, args.nodes)
            self.list_partitioned_set_train = []
            for client_id in range(args.nodes):
                self.list_partitioned_set_train.append(Subset(train_set, indices=indices_list[client_id]))
        else:
            targets = train_set.targets
            if isinstance(targets, torch.Tensor):
                targets = targets.numpy().tolist()
            y_universal = np.array(targets)
            if 'dir' in args.iid:
                alpha = float(args.iid[3:])
                print('alpha', alpha)
                if args.dataset == 'tiny-imagenet':
                    from data.partition import TinyImageNetPartitioner
                    partitioner = TinyImageNetPartitioner
                elif args.dataset in ['mnist', 'cifar-10', 'fashionmnist']:
                    from data.partition import CIFAR10Partitioner
                    partitioner = CIFAR10Partitioner
                elif args.dataset == 'cifar-100':
                    from data.partition import CIFAR100Partitioner
                    partitioner = CIFAR100Partitioner
                map_client_idx = partitioner(y_universal, 
                                             args.nodes,
                                             unbalance_sgm=1,
                                             balance=None,
                                             partition="dirichlet",
                                             dir_alpha=alpha,
                                             min_require_size=args.min_require_size,
                                             seed=args.seed).client_dict
            else:
                degree = int(args.iid)
                if args.num_classes == 100:
                    degree *= 10
                elif args.num_classes == 200:
                    degree *= 20
                map_client_idx = partition_idx_labelnoniid(y_universal, args.nodes, degree,
                                                           num_classes=args.num_classes)
            indices_list = []
            for _, v in map_client_idx.items():
                np.random.shuffle(v)
                indices_list.append(v)

            self.list_partitioned_set_train = []

            for client_id in range(args.nodes):
                self.list_partitioned_set_train.append(Subset(train_set, indices=indices_list[client_id]))

        poisoned_sets = []
        for index_client, item in enumerate(self.list_partitioned_set_train):
            if index_client < int(args.nodes * args.malicious_ratio):
                poisoned_sets.append(poison_dataset(item, args))
            else:
                poisoned_sets.append(item)
        self.list_partitioned_set_train = poisoned_sets

        self.train_loader_list = [DataLoader(train_subset, args.batch, shuffle=True, pin_memory=True) for train_subset
                                  in self.list_partitioned_set_train]
        self.eval_loader = DataLoader(eval_set, args.batch, shuffle=True, pin_memory=True)
        self.eval_loader_backdoor = DataLoader(build_backdoor_eval_set(eval_set, args), args.batch, shuffle=True, pin_memory=True)


def poison_dataset(dst, args):
    data_mask = build_mask(args.dataset, args.dba)
    poison_index = np.random.choice(range(len(dst)), int(len(dst) * args.poison_sample_ratio), replace=False)
    poisoned_set = []
    for index_item, item in enumerate(dst):
        if index_item in poison_index:
            poisoned_set.append((torch.masked_fill(item[0], mask=data_mask, value=args.pix_val), args.backdoor_target))
        else:
            poisoned_set.append(item)
    return poisoned_set


def build_backdoor_eval_set(dst, args):
    data_mask = build_mask(args.dataset)
    poisoned_set = []
    for item in dst:
        if item[1] != args.backdoor_target:
            poisoned_set.append((torch.masked_fill(item[0], mask=data_mask, value=args.pix_val), args.backdoor_target))
    return poisoned_set


def get_mean_and_std(dataset):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, _ in dataloader:
        for i in range(1):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std
