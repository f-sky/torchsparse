import numpy as np
import torch


def to_device(x, device, non_blocking=True):
    if x is None:
        return x
    elif isinstance(x, list):
        return [to_device(a, device) for a in x]
    elif isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    elif isinstance(x, torch.Tensor):
        return x.to(device=device, non_blocking=non_blocking)
    elif isinstance(x, (int, float, str)):
        return x
    else:
        raise TypeError()


def clone(x):
    if x is None:
        return x
    elif isinstance(x, torch.Tensor):
        return x.clone()
    elif isinstance(x, np.ndarray):
        return x.copy()
    elif isinstance(x, list):
        return [clone(el) for el in x]
    elif isinstance(x, dict):
        return {k: clone(v) for k, v in x.items()}
    elif isinstance(x, (int, float, str)):
        return x
    else:
        raise TypeError()
