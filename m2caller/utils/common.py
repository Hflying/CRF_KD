import math, os, random, numpy as np, torch

def seed_everything(seed=1337):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device(no_cuda=False):
    use_cuda = (not no_cuda) and torch.cuda.is_available()
    return torch.device('cuda' if use_cuda else 'cpu')
