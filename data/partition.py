from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import warnings

import numpy as np



class DataPartitioner(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def _perform_partition(self):
        raise NotImplementedError()

    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError()

    @abstractmethod
    def __len__(self):
        raise NotImplementedError()


class CIFAR10Partitioner(DataPartitioner):
    num_classes = 10

    def __init__(self, targets, num_clients,
                 balance=True, partition="iid",
                 unbalance_sgm=0,
                 num_shards=None,
                 dir_alpha=None,
                 verbose=True,
                 min_require_size=None,
                 seed=None):

        self.targets = np.array(targets) 
        self.num_samples = self.targets.shape[0]
        self.num_clients = num_clients
        self.client_dict = dict()
        self.partition = partition
        self.balance = balance
        self.dir_alpha = dir_alpha
        self.num_shards = num_shards
        self.unbalance_sgm = unbalance_sgm
        self.verbose = verbose
        self.min_require_size = min_require_size
        np.random.seed(seed)
        
        if balance is None:
            assert partition in ["dirichlet", "shards"], f"When balance=None, 'partition' only " \
                                                         f"accepts 'dirichlet' and 'shards'."
        elif isinstance(balance, bool):
            assert partition in ["iid", "dirichlet"], f"When balance is bool, 'partition' only " \
                                                      f"accepts 'dirichlet' and 'iid'."
        else:
            raise ValueError(f"'balance' can only be NoneType or bool, not {type(balance)}.")

        self.client_dict = self._perform_partition()
        self.client_sample_count = samples_num_count(self.client_dict, self.num_clients)

    def _perform_partition(self):
        if self.balance is None:
            if self.partition == "dirichlet":
                client_dict = hetero_dir_partition(self.targets,
                                                     self.num_clients,
                                                     self.num_classes,
                                                     self.dir_alpha,
                                                     min_require_size=self.min_require_size)

            else:
                client_dict = shards_partition(self.targets, self.num_clients, self.num_shards)

        else: 
            if self.balance is True:
                client_sample_nums = balance_split(self.num_clients, self.num_samples)
            else:
                client_sample_nums = lognormal_unbalance_split(self.num_clients,
                                                                 self.num_samples,
                                                                 self.unbalance_sgm)

            if self.partition == "iid":
                client_dict = homo_partition(client_sample_nums, self.num_samples)
            else:
                client_dict = client_inner_dirichlet_partition(self.targets, self.num_clients,
                                                                 self.num_classes, self.dir_alpha,
                                                                 client_sample_nums, self.verbose)

        return client_dict

    def __getitem__(self, index):
        return self.client_dict[index]

    def __len__(self):
        return len(self.client_dict)


class CIFAR100Partitioner(CIFAR10Partitioner):
    num_classes = 100


class BasicPartitioner(DataPartitioner):
    num_classes = 2
    def __init__(self, targets, num_clients,
                 partition='iid',
                 dir_alpha=None,
                 major_classes_num=1,
                 verbose=True,
                 min_require_size=None,
                 seed=None):
        self.targets = np.array(targets) 
        self.num_samples = self.targets.shape[0]
        self.num_clients = num_clients
        self.client_dict = dict()
        self.partition = partition
        self.dir_alpha = dir_alpha
        self.verbose = verbose
        self.min_require_size = min_require_size
            
        np.random.seed(seed)

        if partition == "noniid-#label":
            assert isinstance(major_classes_num, int), f"'major_classes_num' should be integer, " \
                                                       f"not {type(major_classes_num)}."
            assert major_classes_num > 0, f"'major_classes_num' should be positive."
            assert major_classes_num < self.num_classes, f"'major_classes_num' for each client " \
                                                         f"should be less than number of total " \
                                                         f"classes {self.num_classes}."
            self.major_classes_num = major_classes_num
        elif partition in ["noniid-labeldir", "unbalance"]:
            assert dir_alpha > 0, f"Parameter 'dir_alpha' for Dirichlet distribution should be " \
                                  f"positive."
        elif partition == "iid":
            pass
        else:
            raise ValueError(
                f"tabular data partition only supports 'noniid-#label', 'noniid-labeldir', "
                f"'unbalance', 'iid'. {partition} is not supported.")

        self.client_dict = self._perform_partition()
        self.client_sample_count = samples_num_count(self.client_dict, self.num_clients)

    def _perform_partition(self):
        if self.partition == "noniid-#label":
            client_dict = label_skew_quantity_based_partition(self.targets, self.num_clients,
                                                                self.num_classes,
                                                                self.major_classes_num)

        elif self.partition == "noniid-labeldir":
            client_dict = hetero_dir_partition(self.targets, self.num_clients, self.num_classes,
                                                 self.dir_alpha,
                                                 min_require_size=self.min_require_size)

        elif self.partition == "unbalance":
            client_sample_nums = dirichlet_unbalance_split(self.num_clients, self.num_samples,
                                                             self.dir_alpha)
            client_dict = homo_partition(client_sample_nums, self.num_samples)

        else:
            # IID
            client_sample_nums = balance_split(self.num_clients, self.num_samples)
            client_dict = homo_partition(client_sample_nums, self.num_samples)

        return client_dict

    def __getitem__(self, index):
        return self.client_dict[index]

    def __len__(self):
        return len(self.client_dict)


class VisionPartitioner(BasicPartitioner):
    num_classes = 10

    def __init__(self, targets, num_clients,
                 partition='iid',
                 dir_alpha=None,
                 major_classes_num=None,
                 verbose=True,
                 seed=None):
        super(VisionPartitioner, self).__init__(targets=targets, num_clients=num_clients,
                                                partition=partition,
                                                dir_alpha=dir_alpha,
                                                major_classes_num=major_classes_num,
                                                verbose=verbose,
                                                seed=seed)


class MNISTPartitioner(VisionPartitioner):
    num_features = 784


class FMNISTPartitioner(VisionPartitioner):
    num_features = 784


class SVHNPartitioner(VisionPartitioner):
    num_features = 1024


class FCUBEPartitioner(DataPartitioner):
    num_classes = 2
    num_clients = 4 

    def __init__(self, data, partition):
        if partition not in ['synthetic', 'iid']:
            raise ValueError(
                f"FCUBE only supports 'synthetic' and 'iid' partition, not {partition}.")
        self.partition = partition
        self.data = data
        if isinstance(data, np.ndarray):
            self.num_samples = data.shape[0]
        else:
            self.num_samples = len(data)

        self.client_dict = self._perform_partition()

    def _perform_partition(self):
        if self.partition == 'synthetic':
            client_dict = fcube_synthetic_partition(self.data)
        else:
            client_sample_nums = balance_split(self.num_clients, self.num_samples)
            client_dict = homo_partition(client_sample_nums, self.num_samples)

        return client_dict

    def __getitem__(self, index):
        return self.client_dict[index]

    def __len__(self):
        return self.num_clients


class AdultPartitioner(BasicPartitioner):
    num_features = 123
    num_classes = 2


class RCV1Partitioner(BasicPartitioner):
    num_features = 47236
    num_classes = 2


class CovtypePartitioner(BasicPartitioner):
    num_features = 54
    num_classes = 2


def split_indices(num_cumsum, rand_perm):
    client_indices_pairs = [(cid, idxs) for cid, idxs in
                            enumerate(np.split(rand_perm, num_cumsum)[:-1])]
    client_dict = dict(client_indices_pairs)
    return client_dict


def balance_split(num_clients, num_samples):
    num_samples_per_client = int(num_samples / num_clients)
    client_sample_nums = (np.ones(num_clients) * num_samples_per_client).astype(
        int)
    return client_sample_nums


def lognormal_unbalance_split(num_clients, num_samples, unbalance_sgm):
    num_samples_per_client = int(num_samples / num_clients)
    if unbalance_sgm != 0:
        client_sample_nums = np.random.lognormal(mean=np.log(num_samples_per_client),
                                                 sigma=unbalance_sgm,
                                                 size=num_clients)
        client_sample_nums = (
                client_sample_nums / np.sum(client_sample_nums) * num_samples).astype(int)
        diff = np.sum(client_sample_nums) - num_samples 
        if diff != 0:
            for cid in range(num_clients):
                if client_sample_nums[cid] > diff:
                    client_sample_nums[cid] -= diff
                    break
    else:
        client_sample_nums = (np.ones(num_clients) * num_samples_per_client).astype(int)

    return client_sample_nums


def dirichlet_unbalance_split(num_clients, num_samples, alpha):
    min_size = 0
    while min_size < 10:
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = proportions / proportions.sum()
        min_size = np.min(proportions * num_samples)

    client_sample_nums = (proportions * num_samples).astype(int)
    return client_sample_nums


def homo_partition(client_sample_nums, num_samples):
    rand_perm = np.random.permutation(num_samples)
    num_cumsum = np.cumsum(client_sample_nums).astype(int)
    client_dict = split_indices(num_cumsum, rand_perm)
    return client_dict


def hetero_dir_partition(targets, num_clients, num_classes, dir_alpha, min_require_size=None):
    if min_require_size is None:
        min_require_size = num_classes

    if not isinstance(targets, np.ndarray):
        targets = np.array(targets)
    num_samples = targets.shape[0]

    min_size = 0
    while min_size < min_require_size:
        idx_batch = [[] for _ in range(num_clients)]
        for k in range(num_classes):
            idx_k = np.where(targets == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(
                np.repeat(dir_alpha, num_clients))
            proportions = np.array(
                [p * (len(idx_j) < num_samples / num_clients) for p, idx_j in
                 zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in
                         zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    client_dict = dict()
    for cid in range(num_clients):
        np.random.shuffle(idx_batch[cid])
        client_dict[cid] = np.array(idx_batch[cid])

    return client_dict


def shards_partition(targets, num_clients, num_shards):
    if not isinstance(targets, np.ndarray):
        targets = np.array(targets)
    num_samples = targets.shape[0]

    size_shard = int(num_samples / num_shards)
    if num_samples % num_shards != 0:
        warnings.warn("warning: length of dataset isn't divided exactly by num_shards. "
                      "Some samples will be dropped.")

    shards_per_client = int(num_shards / num_clients)
    if num_shards % num_clients != 0:
        warnings.warn("warning: num_shards isn't divided exactly by num_clients. "
                      "Some shards will be dropped.")

    indices = np.arange(num_samples)
    indices_targets = np.vstack((indices, targets))
    indices_targets = indices_targets[:, indices_targets[1, :].argsort()]
    sorted_indices = indices_targets[0, :]

    rand_perm = np.random.permutation(num_shards)
    num_client_shards = np.ones(num_clients) * shards_per_client
    num_cumsum = np.cumsum(num_client_shards).astype(int)
    client_shards_dict = split_indices(num_cumsum, rand_perm)
    client_dict = dict()
    for cid in range(num_clients):
        shards_set = client_shards_dict[cid]
        current_indices = [
            sorted_indices[shard_id * size_shard: (shard_id + 1) * size_shard]
            for shard_id in shards_set]
        client_dict[cid] = np.concatenate(current_indices, axis=0)

    return client_dict


def client_inner_dirichlet_partition(targets, num_clients, num_classes, dir_alpha,
                                     client_sample_nums, verbose=True):
    if not isinstance(targets, np.ndarray):
        targets = np.array(targets)

    rand_perm = np.random.permutation(targets.shape[0])
    targets = targets[rand_perm]

    class_priors = np.random.dirichlet(alpha=[dir_alpha] * num_classes,
                                       size=num_clients)
    prior_cumsum = np.cumsum(class_priors, axis=1)
    idx_list = [np.where(targets == i)[0] for i in range(num_classes)]
    class_amount = [len(idx_list[i]) for i in range(num_classes)]

    client_indices = [np.zeros(client_sample_nums[cid]).astype(np.int64) for cid in
                      range(num_clients)]

    while np.sum(client_sample_nums) != 0:
        curr_cid = np.random.randint(num_clients)
        if verbose:
            print('Remaining Data: %d' % np.sum(client_sample_nums))
        if client_sample_nums[curr_cid] <= 0:
            continue
        client_sample_nums[curr_cid] -= 1
        curr_prior = prior_cumsum[curr_cid]
        while True:
            curr_class = np.argmax(np.random.uniform() <= curr_prior)
            if class_amount[curr_class] <= 0:
                continue
            class_amount[curr_class] -= 1
            client_indices[curr_cid][client_sample_nums[curr_cid]] = \
                idx_list[curr_class][class_amount[curr_class]]

            break

    client_dict = {cid: client_indices[cid] for cid in range(num_clients)}
    return client_dict


def label_skew_quantity_based_partition(targets, num_clients, num_classes, major_classes_num):
    if not isinstance(targets, np.ndarray):
        targets = np.array(targets)

    idx_batch = [np.ndarray(0, dtype=np.int64) for _ in range(num_clients)]
    times = [0 for _ in range(num_classes)]
    contain = []
    for cid in range(num_clients):
        current = [cid % num_classes]
        times[cid % num_classes] += 1
        j = 1
        while j < major_classes_num:
            ind = np.random.randint(num_classes)
            if ind not in current:
                j += 1
                current.append(ind)
                times[ind] += 1
        contain.append(current)

    for k in range(num_classes):
        idx_k = np.where(targets == k)[0]
        np.random.shuffle(idx_k)
        split = np.array_split(idx_k, times[k])
        ids = 0
        for cid in range(num_clients):
            if k in contain[cid]:
                idx_batch[cid] = np.append(idx_batch[cid], split[ids])
                ids += 1

    client_dict = {cid: idx_batch[cid] for cid in range(num_clients)}
    return client_dict


def fcube_synthetic_partition(data):
    num_clients = 4
    client_indices = [[] for _ in range(num_clients)]
    for idx, sample in enumerate(data):
        p1, p2, p3 = sample
        if (p1 > 0 and p2 > 0 and p3 > 0) or (p1 < 0 and p2 < 0 and p3 < 0):
            client_indices[0].append(idx)
        elif (p1 > 0 and p2 > 0 and p3 < 0) or (p1 < 0 and p2 < 0 and p3 > 0):
            client_indices[1].append(idx)
        elif (p1 > 0 and p2 < 0 and p3 > 0) or (p1 < 0 and p2 > 0 and p3 < 0):
            client_indices[2].append(idx)
        else:
            client_indices[3].append(idx)
    client_dict = {cid: np.array(client_indices[cid]).astype(int) for cid in range(num_clients)}
    return client_dict


def samples_num_count(client_dict, num_clients):
    client_samples_nums = [[cid, client_dict[cid].shape[0]] for cid in
                           range(num_clients)]
    client_sample_count = pd.DataFrame(data=client_samples_nums,
                                       columns=['client', 'num_samples']).set_index('client')
    return client_sample_count

def noniid_slicing(dataset, num_clients, num_shards):
    total_sample_nums = len(dataset)
    size_of_shards = int(total_sample_nums / num_shards)
    if total_sample_nums % num_shards != 0:
        warnings.warn(
            "warning: the length of dataset isn't divided exactly by num_shard.some samples will be dropped."
        )
    shard_pc = int(num_shards / num_clients)
    if num_shards % num_clients != 0:
        warnings.warn(
            "warning: num_shard isn't divided exactly by num_clients. some samples will be dropped."
        )

    dict_users = {i: np.array([], dtype='int64') for i in range(num_clients)}

    labels = np.array(dataset.targets)
    idxs = np.arange(total_sample_nums)

    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    idx_shard = [i for i in range(num_shards)]
    for i in range(num_clients):
        rand_set = set(np.random.choice(idx_shard, shard_pc, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i],
                 idxs[rand * size_of_shards:(rand + 1) * size_of_shards]),
                axis=0)

    return dict_users


def random_slicing(dataset, num_clients):
    num_items = int(len(dataset) / num_clients)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_clients):
        dict_users[i] = list(
            np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - set(dict_users[i]))
    return 