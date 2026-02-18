import random
import torch
import numpy as np

def set_torch_device(gpu_ids):
    if len(gpu_ids):
        return torch.device("cuda:{}".format(gpu_ids[0]))
    return torch.device("cpu")

def fix_seeds(seed, with_torch=True, with_cuda=True):
    random.seed(seed)
    np.random.seed(seed)
    if with_torch:
        torch.manual_seed(seed)
    if with_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True