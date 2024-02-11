import torch


def poison_label_exchange_(labels, targets: tuple):
    val1, val2 = targets
    val1_indices = (labels.flatten() == val1).nonzero().flatten()
    val2_indices = (labels.flatten() == val2).nonzero().flatten()
    val1_sources = torch.ones_like(val2_indices) * val1
    val2_sources = torch.ones_like(val1_indices) * val2
    # exchange the values of val1 and val2
    labels.put_(index=val1_indices, source=val2_sources)
    labels.put_(index=val2_indices, source=val1_sources)


def poison_label_cover_(labels, target: int, source: int):
    indices_target = (labels.flatten() == target).nonzero().flatten()
    val_source = torch.ones_like(indices_target) * source
    labels.put_(index=indices_target, source=val_source)