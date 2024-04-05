from dataloaders.sphaera_patches_gt import sphaera_patches_gt
from dataloaders.annotated_sphaera import annotated_sphaera_tables
from train_table import TableTrainer
from utils import set_up_dir
import torch
import pickle
import os
import numpy as np
import csv
import glob
from utils import get_model
import random


import click
@click.command()
@click.option('--jobfile',type=str)
@click.option('--savedir',type=str)
def main(jobfile, savedir):

    print('Current directory', os.getcwd())
    jobdict = pickle.load(open(jobfile, 'rb'))
    binarize=True
    augment = jobdict[0]
    epochs = jobdict[1]
    batchsize = int(jobdict[2])
    pad_context = jobdict[3]
    frac = jobdict[4]
    weight_dir = jobdict[5]
    modelcase = jobdict[6]
    model_spec = jobdict[7]
    multiscale = jobdict[8]
    add_affine =  jobdict[9]
    beta =  jobdict[10]
    seed = jobdict[11]
    removemean=jobdict[14]
    lr_params =  jobdict[15] 
    crop_pad_val = 0.
    default_init_lr = 0.001
    extra_rotations=False
    data_seed=1

    set_up_dir(savedir)

    # Get instance of the digit detection model 
    model = get_model(modelcase, model_spec, train_specs = {'invariance_rots': [0]})
    
    if weight_dir:
        pretrained = True
    else:
         pretrained = False

    if augment:
        augment_str = 'aug_'+'_'.join(['{}_{}'.format(k,str(augment[k]).replace(' ','')) for k in sorted(augment.keys())]).replace('ate', '')
    else:
        augment_str = 'aug_False'
        
    if  modelcase in ['rotinv','rotinvtriv', 'rotinvconv']:
        model_str = modelcase + '_' + '_'.join(['{}_{}'.format(k,str(model_spec[k]).replace(' ','').replace('_','')) for k in sorted(model_spec.keys())])
    else:
        model_str = modelcase
        
    print(model)
    
    assert multiscale==[1.]

    # Load sphaera digit patches dataset,
    dataloader = sphaera_patches_gt(batchsize, pad_context, frac, binarize, augment=augment, scales = multiscale, beta=beta, add_affine=add_affine,  removemean=removemean,  pad_val=crop_pad_val, extra_rotations=extra_rotations, seed=data_seed)

    if seed:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        print('Set seed to', seed)

    # Start training
    print('Training for', epochs)
    trainer = TableTrainer(savedir, model_str, model = model,  epochs= epochs, jobdict=jobdict, weight_dir=weight_dir, seed=seed, lr_params= lr_params, default_init_lr=default_init_lr)
    weight_file = trainer.train(dataloader)
    print('Weigt file directory', weight_file)

    
if __name__ == '__main__':
    main()
