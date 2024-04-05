from dataloaders.sphaera_tables import sphaera_tables
from utils import set_up_dir
import torch
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import csv
import tqdm
import glob
import pandas as pd
from models.e2cnn_models import TableModel
from dataloaders.annotated_sphaera import annotated_sphaera_tables
from configs import SACROBOSCO_DATA_ROOT

def dynamic_shift(scale, init_shift = [8,10]):
    if scale <= 0.7:
        return [12, 14]
    elif scale > 0.7:
        return init_shift

import click
@click.command()
@click.option('--sgeid',type=int)
@click.option('--jobfile',type=str)
@click.option('--modeljobfile',type=str)
@click.option('--weightfile',type=str)
@click.option('--savedir',type=str)
@click.option('--extrastr',type=str)
@click.option('--caseflag',type=str, default = 'tables_sphaera') # 'tables_sphaera', 'external_table','annotated_sphaera'

def main(sgeid, jobfile, modeljobfile, weightfile, savedir, extrastr,  caseflag): 

    case_flag = caseflag
    bias = 0.1
    extra_bias = 10.
    shift = [8,10] 
    scales = [0.5, 0.65,0.8, 0.95, 1.0] 
    refscale_sizes =  [int(1200*scale) for scale in scales]
    numbers = [(i,) for i in range(10)] + [(0,i) for i in range(10)] + [(int(str(i)[0]), int(str(i)[1])) for i in range(10,100)]

    
    print('loading jobdict', jobfile)
    print('save dir', savedir)

    if case_flag == 'annotated_sphaera' or case_flag == 'tables_sphaera':
        table_source = 'data/corpus/sphaera_tables_9793.csv'
        root_dir_sphaera = os.path.join(SACROBOSCO_DATA_ROOT, 'processed')

    elif case_flag == 'external_table':
        # Load [*.jpg] from jobdict
        table_source = pickle.load(open(jobfile, 'rb'))[sgeid][1]
        idx_list = [pickle.load(open(jobfile, 'rb'))[sgeid][0]]
        root_dir_sphaera=None


    if isinstance(shift, list):
        infer_dir = os.path.join(savedir, extrastr, 'dyn_shift_{}'.format('_'.join([str(s) for s in shift])))
    else:
        infer_dir = os.path.join(savedir, extrastr, 'dyn_shift_{}'.format(shift))

    set_up_dir(infer_dir)

    dataloaders = []
    orig_sizes = None
    
    if case_flag ==  'annotated_sphaera':
        datal_unscaled = annotated_sphaera_tables(data_root=None, binarize=False, refsize=None,  removemean=True)
        print(len(datal_unscaled))
        idx_list = list(range(len(datal_unscaled)))

    elif case_flag ==  'tables_sphaera' or case_flag ==  'external_table':
        datal_unscaled = sphaera_tables(data_root=root_dir_sphaera, table_source=table_source, binarize=False, refsize=None, removemean=True) 
        if case_flag ==  'tables_sphaera':
            idx_list = pickle.load(open(jobfile, 'rb'))[sgeid]
        

    if case_flag ==  'annotated_sphaera' or case_flag ==  'external_table':            
        orig_sizes = [np.prod(x[0].shape) for x in  datal_unscaled]
    elif case_flag ==  'tables_sphaera':
        # Use pre-computed scan sizes
         orig_sizes = pickle.load(open('data/corpus/sphaera_page_sizes.p', 'rb'))
   
    for ref in refscale_sizes:
        if case_flag ==  'annotated_sphaera':
            datal =  annotated_sphaera_tables(data_root=None, binarize=False, refsize=ref,  removemean=True)
        elif case_flag ==  'tables_sphaera' or case_flag == 'external_table':
            datal = sphaera_tables(data_root=root_dir_sphaera,table_source=table_source, binarize=False, refsize=ref, removemean=True) 
        dataloaders.append(datal)

        
    with torch.no_grad():

        # Load configs to initialize the model for inference
        model_jobdict = pickle.load(open(modeljobfile, 'rb'))        
        mdict = model_jobdict[7]
        mdict['clip_max'] = None 
        mdict['invariance_rots'] = [0,1,-1] 


        # Load model
        model = TableModel(**mdict)
        model.load_state_dict(torch.load(weightfile))
        model.eval()
        model.cuda()

        for idx in idx_list:

            highres_flag = 0

            if orig_sizes is None:
                continue
            else:
                orig_size = orig_sizes[idx]
                if orig_size is None:
                    continue
                        
            outputfile = os.path.join(infer_dir, '{}_{}.p'.format(sgeid, idx))            

            if os.path.exists(outputfile):
                print('File exists', outputfile)
                continue

                
            # Collecting page at multiple scales
            pages = []
            pages_all = []
            valid_scales = []
            for datal, scale in zip(dataloaders, scales):
            
                try:
                    page, meta = datal[idx]
                except:
                    page = datal[idx][0]
                    meta = None
                    
                h,w = page.shape[1:]

                if np.sqrt(orig_size/(h*w)) <=6:
                    pages.append(page.cuda())
                    valid_scales.append(scale)
                pages_all.append(page)
                
            if len(valid_scales)==0:
                print('Using all scales - highres image!')
                valid_scales = scales
                highres_flag=1
                pages = [p.cuda() for p in pages_all]
                                                       
            page = torch.autograd.Variable(page.unsqueeze(0), requires_grad=False)
            
            H, H_compositions, _ = model.compute_forward_scales(pages, 
                                        numbers, 
                                        shift,
                                        remove_mean=False,
                                        scale_sizes = [int(1200*scale) for scale in valid_scales],
                                        scales = valid_scales,
                                        shift_func = dynamic_shift)            


            A0 = np.array([h.detach().cpu().numpy().squeeze() for h in H_compositions])
            A = A0 - extra_bias
            
            # Filter background noise
            A_plus = np.clip(A, a_min=0, a_max = None)

            # Spatially pool activation
            a_pool = A0.sum((1,2))
            a_pool_plus = A_plus.sum((1,2))

            # Collect bigram map compositions
            H_all_comps = np.array([h.detach().cpu().numpy() for h in H_compositions])

            res = {'X':  page.detach().cpu().numpy().squeeze(),
                   'X_scale': model.x_scale_chosen, 
                   'H': H.detach().cpu().numpy().squeeze(), 
                   'H_all_comps': H_all_comps,                    
                   'h_pool': a_pool,
                   'h_pool_plus': a_pool_plus,      
                   'numbers':numbers,
                   'size': (h,w),
                   'pagefile':meta,
                   'rot_chosen': model.rots,
                   'all_shift_scales': model.all_shift_scales,
                   'shift_chosen_scales': model.shift_chosen_scales,
                   'scales_chosen':  model.scale_chosen,
                   'valid_scales': valid_scales,
                   'chosen_params': model.chosen_params,
                   'highres': highres_flag}
                
            print(outputfile)
            pickle.dump(res, open(outputfile, 'wb'))


if __name__ == '__main__':
    main()
