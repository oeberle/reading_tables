from torch.utils.data import Dataset
import torch
import numpy as np

class LimitDataset(Dataset):
    def __init__(self, dataset, n):
        self.dataset = dataset
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self.dataset[i]


def padding(x, pad, value=0):
    x = torch.nn.functional.pad(x, pad, value=value)    
    return x


def crop(x,tw,th):
    w, h = x.shape[-2], x.shape[-1]
    x1 = int(round((w - tw) / 2.))
    y1 = int(round((h - th) / 2.))
    if len(x.shape) == 3:
        return x[:, x1:x1+tw,y1:y1+th]
    elif len(x.shape) == 2:
        return x[x1:x1+tw,y1:y1+th]

def remove_mean(x):
    assert len(x.shape) == 3
    x = x - torch.mean(x, dim=(1,2), keepdim=True)
    return x

def ToTensor(x):
    X = np.array(x)
    if len(X.shape)==3:
        X = np.transpose(X, (2,0,1))
    else:
        X = X[np.newaxis,:,:]
    X = X/255.
    assert np.max(X)<=1.
    assert np.min(X)>=0
    X = 1-X 
    return torch.Tensor(X)
