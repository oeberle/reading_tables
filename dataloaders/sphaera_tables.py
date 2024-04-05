from torch.utils.data import Dataset
import torch
from dataloaders.dataset_utils import LimitDataset, padding, crop, remove_mean, ToTensor
import torchvision.transforms as transforms
import os
from dataloaders.proc_utils import indiviual_resize
import pandas as pd
from skimage import io
from PIL import Image
from dataloaders.dataset import PageDataset


def sphaera_tables(data_root, table_source, binarize=False, refsize=1200, removemean=True) :   

    trans_list = [transforms.Grayscale(),
                  transforms.Lambda(lambda x: ToTensor(x)),
                  transforms.Normalize([0.5], [0.5]),]
    
    if removemean:
        trans_list = trans_list + [transforms.Lambda(lambda x: remove_mean(x))]

    extra_trans = []
    if refsize:
        extra_trans = [transforms.Lambda(lambda x: indiviual_resize(x, refsize=refsize))]

    page_trans =  transforms.Compose(extra_trans + trans_list)
     
    data = PageDataset(table_source=table_source,
                    root_dir = data_root,
                    transform = page_trans
                                            )
    return data


