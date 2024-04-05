from torch.utils.data import Dataset
import torch
from dataloaders.dataset import FullyAnnotatedPages
from dataloaders.dataset_utils import LimitDataset, padding, crop, remove_mean, ToTensor
import torchvision.transforms as transforms
import os
from dataloaders.proc_utils import indiviual_resize


def annotated_sphaera_tables(root_dir, binarize=True, refsize=1200, removemean=True, return_bookstr=False,  csv_file = 'data/corpus/digit_page_annotations.csv') :   

    trans_list = [transforms.Grayscale(),
                  transforms.Lambda(lambda x: ToTensor(x)),
                  transforms.Normalize([0.5], [0.5]),]
    
    if removemean:
        trans_list = trans_list + [transforms.Lambda(lambda x: remove_mean(x))]

    extra_trans = []
    if refsize:
        extra_trans = [transforms.Lambda(lambda x: indiviual_resize(x, refsize=refsize))]
        
        
    page_trans =  transforms.Compose(extra_trans + trans_list)

    data = FullyAnnotatedPages(
                   table_source=csv_file,
                   root_dir = root_dir,
                   transform=page_trans,
                   return_bookstr = return_bookstr
                   )   
           
    return data


