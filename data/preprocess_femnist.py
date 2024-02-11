import json
import os
import pickle
import platform
import numpy as np
import torch
import random


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if seed == -1:
        torch.backends.cudnn.deterministic = False
    else:
        torch.backends.cudnn.deterministic = True


def _get_dst_root_path(dst_path=None):
    if dst_path is None:
        if platform.system().lower() == 'windows':
            import getpass
            dst_path = r'C:\Users\{}\.dataset'.format(getpass.getuser())
        else:
            import pwd
            user_name = pwd.getpwuid(os.getuid())[0]
            dst_path = r'/home/{}/.dataset'.format(user_name)
    return dst_path


def merge_femnist_dataset(dst_path=None):
    data = []
    dst_path = os.path.join(_get_dst_root_path(dst_path), 'femnist', 'all_data')
    for idx in range(0, 36):
        json_content = json.load(open(os.path.join(dst_path, 'all_data_{0}.json'.format(idx))))
        data_dic = json_content['user_data']
        for user_id, x_y_dic in data_dic.items():
            data.append(zip(x_y_dic['x'], x_y_dic['y']))
    pickle.dump(data, open(os.path.join('processed', 'femnist', 'union.pkl'), 'wb'))


def rewrite_femnist_average(dst_path=None, train_ratio=0.9, seed=69):
    set_seed(seed)
    if not os.path.exists(os.path.join('processed', 'femnist', 'average', 'seed={0}'.format(seed))):
        os.makedirs(os.path.join('processed', 'femnist', 'average', 'seed={0}'.format(seed)))
    writer_cnt = 0
    data_list = []
    dst_path = os.path.join(_get_dst_root_path(dst_path), 'femnist', 'all_data')
    for idx in range(0, 36):
        json_content = json.load(open(os.path.join(dst_path, 'all_data_{0}.json'.format(idx))))
        data_dic = json_content['user_data']
        for user_id, x_y_dic in data_dic.items():
            if len(x_y_dic['x']) < 200:
                continue
            x_list = []
            for item in x_y_dic['x']:
                x_list.append(np.reshape(item, (28, 28)))
            x_list = np.array(x_list)
            for item in zip(x_list, x_y_dic['y']):
                data_list.append(item)
            writer_cnt += 1

    np.random.shuffle(data_list)
    item_per_task = int(len(data_list) / writer_cnt)
    for task_id in range(writer_cnt):
        data_cur_task = data_list[task_id * item_per_task:(task_id + 1) * item_per_task]
        np.random.shuffle(data_cur_task)
        split_index = int(len(data_cur_task) * train_ratio)
        train_set, eval_set = data_cur_task[:split_index], data_cur_task[split_index:]
        with open(os.path.join('processed', 'femnist', 'average', 'seed={0}'.format(seed), 'writer_{0}_train.pkl'.format(task_id)), 'wb') as writer:
            pickle.dump(train_set, writer)
        with open(os.path.join('processed', 'femnist', 'average', 'seed={0}'.format(seed), 'writer_{0}_eval.pkl'.format(task_id)), 'wb') as writer:
            pickle.dump(eval_set, writer)


def rewrite_femnist_by_writer(dst_path=None, train_ratio=0.9, seed=69):
    set_seed(seed)
    if not os.path.exists(os.path.join('data', 'processed', 'femnist', 'by_task', 'seed={0}'.format(seed))):
        os.makedirs(os.path.join('processed', 'femnist', 'by_task', 'seed={0}'.format(seed)))
    cnt = 0
    
    writer_id_list = np.array([i for i in range(3597)])
    np.random.shuffle(writer_id_list)
    dst_path = os.path.join(_get_dst_root_path(dst_path), 'femnist', 'all_data')
    for idx in range(0, 36):
        json_content = json.load(open(os.path.join(dst_path, 'all_data_{0}.json'.format(idx))))
        data_dic = json_content['user_data']
        for user_id, x_y_dic in data_dic.items():
            x_list = []
            for item in x_y_dic['x']:
                x_list.append(np.reshape(item, (28, 28)))
            x_list = np.array(x_list)
            merge_list = [item for item in zip(x_list, x_y_dic['y'])]
            np.random.shuffle(merge_list)
            split_index = int(len(merge_list) * train_ratio)
            train_set, eval_set = merge_list[:split_index], merge_list[split_index:]
            with open(os.path.join('processed', 'femnist', 'by_task', 'seed={0}'.format(seed), 'writer_{0}_train.pkl'.format(writer_id_list[cnt])), 'wb') as writer:
                pickle.dump(train_set, writer)
            with open(os.path.join('processed', 'femnist', 'by_task', 'seed={0}'.format(seed), 'writer_{0}_eval.pkl'.format(writer_id_list[cnt])), 'wb') as writer:
                pickle.dump(eval_set, writer)
            cnt += 1


if __name__ == '__main__':
    rewrite_femnist_by_writer(seed=69)
