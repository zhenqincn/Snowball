import numpy as np
import torch
from .baselines import base_aggregate
from copy import deepcopy
from tqdm import tqdm


def _flatten_model(model_update):
    k_list = []
    if isinstance(model_update, dict):
        for k in model_update.keys():
            k_list.append(k)
        return torch.concat([model_update[k].flatten() for k in k_list])
    else:
        for k in model_update.state_dict().keys():
            k_list.append(k)
        return torch.concat([model_update.state_dict()[k].flatten() for k in k_list])



def _recon_model(update, flatten_update):
    cpy_update = deepcopy(update)
    start_ = 0
    if isinstance(update, dict):
        for _, v in cpy_update.items():
            # if add, accumulate=True
            v.put_(index=torch.LongTensor([i for i in range(v.numel())]).to(device=v.device), source=flatten_update[start_:start_ + v.numel()])
            start_ += v.numel()
    else:
        for _, v in cpy_update.state_dict().items():
            # if add, accumulate=True
            v.put_(index=torch.LongTensor([i for i in range(v.numel())]).to(device=v.device), source=flatten_update[start_:start_ + v.numel()])
            start_ += v.numel()
    return cpy_update


def _clip(model_update, standard_norm):
    flatten_update = _flatten_model(model_update)
    clipped_ = flatten_update * min(1.0, standard_norm / torch.norm(flatten_update, p=2))
    return _recon_model(model_update, clipped_)

def _add_dp_noise_(model, sigma=0.01):
    if isinstance(model, dict):
        for _, param in model.items():
            # noised_layer = torch.FloatTensor(param.shape).normal_(mean=0, std=standard_norm * (1.0 / epsilon) * np.sqrt(2 * np.log(1.25 / sigma))).to(param.device)
            noised_layer = torch.FloatTensor(param.shape).normal_(mean=0, std=sigma).to(param.device)
            param.add_(noised_layer)
    else:
        for _, param in model.state_dict().items():
            # noised_layer = torch.FloatTensor(param.shape).normal_(mean=0, std=standard_norm * (1.0 / epsilon) * np.sqrt(2 * np.log(1.25 / sigma))).to(param.device)
            noised_layer = torch.FloatTensor(param.shape).normal_(mean=0, std=sigma).to(param.device)
            param.add_(noised_layer)


def crfl(model_updates, weight_aggregation):
    update_avg = base_aggregate(model_updates=model_updates, weight_aggregation=weight_aggregation)
    update_avg = _clip(update_avg, standard_norm=15.0)
    _add_dp_noise_(update_avg)
    return update_avg


def eval(base_model, data_loader, device, description):
    model_list = []
    for _ in range(20):
        model_cpy = deepcopy(base_model)
        model_cpy = _clip(model_cpy, standard_norm=15.0)
        _add_dp_noise_(model_cpy)
        model_list.append(model_cpy)

    with torch.no_grad():
        num_correct = 0.0
        num_trained = 0
        with tqdm(data_loader) as bar_epoch:
            for idx, (data, target) in enumerate(bar_epoch):
                pred_list = []
                data = data.to(device)
                for model in model_list:
                    output_ = model(data)
                    pred_ = output_.argmax(dim=1)
                    pred_list.append(pred_.cpu().numpy().tolist())

                pred_list = np.array(pred_list)
                voted_pred_ = []
                for idx_data in range(len(data)):
                    pred_cur_data = pred_list[:, idx_data].tolist()
                    voted_pred_.append(max(pred_cur_data, key=pred_cur_data.count))
                voted_pred_ = torch.tensor(voted_pred_)
                num_correct += voted_pred_.eq(target.view_as(voted_pred_)).sum()
                num_trained += len(data)
                bar_epoch.set_description(description.format(num_correct / num_trained * 100))
    del model_list
    return float(num_correct) / num_trained * 100

