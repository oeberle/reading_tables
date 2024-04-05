from torch.utils.data import Dataset
import torch
from dataloaders.dataset_patches import HistoricFeaturesDataset
from dataloaders.dataset_utils import LimitDataset, padding, crop, remove_mean, ToTensor
import torchvision.transforms as transforms
import numpy as np
import random
from dataloaders.proc_utils import indiviual_resize, randomaffine
import copy

    
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def sphaera_patches_gt(batchsize, pad, frac, binarize, augment=None , scales = [1.0], beta=0.5, add_affine=2, train_shuffle=True, removemean=True, pad_val = 0., adversarial=True, extra_rotations=False, patch_csv='data/training_data/numerical_patches.csv', nontable_csv='data/training_data/contrast_patches.csv', data_root_patches='data/page_data/patches_annotated/', data_root_contrast='data/page_data/contrast_patches/', seed=1) :   

    np.random.seed(seed)
    
    trans_list =  [transforms.Lambda(lambda x: indiviual_resize(x, refsize=int(1200*scales[0]))),
                   transforms.Grayscale(num_output_channels=1),
                   transforms.Lambda(lambda x: ToTensor(x)),
                    transforms.Normalize([0.5], [0.5])]

    if removemean:
        trans_list = trans_list + [transforms.Lambda(lambda x: remove_mean(x))]

    page_trans = transforms.Compose(trans_list)
    
    train_list_scales = []
    
    cluster_inds = []
    n_prev=0
    pad = int(pad*scales[0])
   
    assert scales == [1.] # Only allow scale = 1. for training
    
    scale = scales[0]
    hp,wp = int(scale*164), int(scale*164)
    hcrop = 300 

    print(hp,wp)
    patch_trans = transforms.Compose([     
                    transforms.Lambda(lambda x: padding(x, (hcrop,hcrop,hcrop,hcrop), value=pad_val)),
                    transforms.Lambda(lambda x: crop(x,hp,wp)),
            ])


    mask_trans = transforms.Compose([
                    transforms.Lambda(lambda x: padding(x,  (hcrop,hcrop,hcrop,hcrop), value=0.)),
                    transforms.Lambda(lambda x: crop(x,hp,wp)),
            ])

    data = {'train': [], 'test': []}
    
    for train_flag, case in zip([True, False], ['train', 'test']):

        # Ground truth digit patches
        number_patches = HistoricFeaturesDataset(
                    csv_file = patch_csv, 
                    data_directory = data_root_patches, 
                    train = train_flag, 
                    return_fixation = None,
                    return_single = True, 
                    pad = pad,
                    scale = scale,   
                    beta = beta,
                    binarize = True,
                    page_transform = page_trans,
                    patch_transform = [patch_trans, patch_trans], 
                    mask_transform = mask_trans,
                    test_split = 0.2,
                    extra_rotations = extra_rotations,
                    seed = seed)
        

        data[case].append(number_patches)

        # Add contrast patches
        if adversarial==True and case=='train':
            page_patches = LimitDataset(HistoricFeaturesDataset(csv_file=nontable_csv, 
                            data_directory = data_root_contrast, 
                            train = train_flag, 
                            return_fixation = None,
                            return_single = True, 
                            pad = pad,
                            scale = scale,   
                            beta = beta,
                            binarize = True,
                            page_transform = page_trans,
                            patch_transform = [patch_trans, patch_trans], 
                            mask_transform = mask_trans,
                            test_split = 0.2,
                            local_data = False,
                            seed = seed,
                            extra_rotations = extra_rotations), int(frac*len(number_patches)))

        
            data[case].append(page_patches)

        
                
        if augment and case=='train':

            patch_trans_affine = transforms.Compose([    
                    transforms.Lambda(lambda x: randomaffine(x, **augment)),             
                    transforms.Lambda(lambda x: padding(x, (hcrop,hcrop,hcrop,hcrop), value=pad_val)),
                    transforms.Lambda(lambda x: crop(x, hp,wp)),
            ])

            # Augment patches
            aug_loop_list = [patch_trans_affine]*add_affine
            for augmentation in aug_loop_list:
                number_patches_augment = HistoricFeaturesDataset(
                                                csv_file = patch_csv, 
                                                data_directory = data_root_patches, 
                                                train = train_flag, 
                                                return_fixation = None,
                                                return_single = True, 
                                                pad = pad,
                                                scale = scale,   
                                                beta = beta,
                                                binarize = True,
                                                page_transform = page_trans,
                                                patch_transform = [augmentation, patch_trans], 
                                                mask_transform = mask_trans,
                                                test_split = 0.2,
                                                extra_rotations = extra_rotations,
                                                seed = seed)
                
                
                data[case].append(number_patches_augment)

                
                if adversarial==True:
                    page_patches_augment = LimitDataset(HistoricFeaturesDataset(csv_file=nontable_csv, 
                                    data_directory = data_root_contrast, 
                                    train = train_flag, 
                                    return_fixation = None,
                                    return_single = True, 
                                    pad = pad,
                                    scale = scale,   
                                    beta = beta,
                                    binarize = True,
                                    page_transform = page_trans,
                                    patch_transform = [patch_trans, patch_trans], 
                                    mask_transform = mask_trans,
                                    test_split = 0.2,
                                    local_data = False,
                                    seed = seed,                                                    
                                    extra_rotations = extra_rotations), int(frac*len(number_patches_augment)))

                data[case].append(page_patches_augment)

    dataloaders = {}
    
    for case in ['train', 'test']:
        dset_concat = torch.utils.data.ConcatDataset(data[case])
        loader = torch.utils.data.DataLoader(
                     dset_concat,
                     batch_size = batchsize,
                     collate_fn = collate_fn,
                     shuffle = True if case == 'train' else False,
                     num_workers = 2)
        
        dataloaders[case] = loader
    
    return dataloaders 
