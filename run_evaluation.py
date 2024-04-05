
import glob
import os
import pandas as pd
import numpy as np
import pickle
import numpy as np
from utils import set_up_dir
from table_utils import get_data
from evaluation.full_annotations import eval_table_embeddings
import sklearn
import tqdm
import matplotlib.pyplot as plt
from plotting.plot_corpus import plot_correlation_bar
from evaluation.full_annotations import correlate_, get_gt_veterum_nostro, get_corr_data

############
out_dir =  'results/task_evaluation/'
set_up_dir(out_dir)

PROC_CORR = False
PROC_CLUSTER = False
EVALUATION = True

####### GT PROCESSING ########

if PROC_CORR:
     # 1. Process correlation evaluation

    res_dir_case = "outputs/fully_annotated" # contains the outputs of run_table_inference.py for pages in "data/page_data/eval_annotated".
    eval_root_dir = "data/page_data/eval_annotated/processed"  # contains the fully annotated corpus
    plot_dir =  "results/fully_annotated" # output directory

    # Parameters for the peak detection
    peak_params = {'scale_isolated': 3,
                     'max_d': 15,
                     'peak_cutoff_rel': 0.12,
                     'clip_cutoff': 0.,
                     'verbose': False}

    all_tables = list(range(21))
    ignore_inds = [] 

    set_up_dir(plot_dir)
    corr, corr_pd, corr_pool_pd, res_hists = correlate_(res_dir_case, eval_root_dir, plot_dir, plot=False, save=False,
                               ignore_inds=ignore_inds, hist_key='h_pool_plus', peak_params=peak_params, return_hists=True)

    valid_ks = [0, 1, 2, 7, 14, 15, 16, 17, 18, 19, 20 ]
    invalid_ks = [3]
    text_ks = [4, 5, 6, 8, 9, 10, 11, 12, 13]
        
    res_ = {}
    for name, inds_ in [('dense_tables', valid_ks), ('text_pages', text_ks), ('invalid_tables', invalid_ks)]:
        corrs = [[corr[i] for i in inds_], [corr_pd[i] for i in inds_], [corr_pool_pd[i] for i in inds_]]
        print(len(inds_), np.nanmean(corrs[0]) , np.nanmean(corrs[1]), np.nanmean(corrs[2]))
        res_[name] = [{'true': res_hists[i][0], 
                       'pool': res_hists[i][1],
                       'pd': res_hists[i][2]}
                       for i in inds_]
        
    pickle.dump(res_, open('data/inference/detection_hists.p', 'wb'))


if PROC_CLUSTER:

    # 2. Process cluster evaluation
    
    SEEDS=[109706657, 253505912, 541217254, 506598379, 577336311,
           546209131, 263384753,   9498415, 650189128, 197512609]

    H_dict = pickle.load(open('data/inference/table_embedings.p', 'rb'))
    
    cases = ['H', 'H_sqrt' ,'H_pool', 'H_pool_sqrt',  'H_single_sqrt', 'Vgg31', 'Random'] 


    res_pfile='data/inference/res_9793.p'
    H_all, M_all = get_data(res_pfile)
       
    _, _, y_true_multi = get_gt_veterum_nostro(H_all, M_all, gt_csv = 'data/corpus/sun_zodiac.csv')

    res_roc = {k:[] for k in cases}

    # Computing performance in detecting ground truth clusters across different embedding approaches
    for k in res_roc.keys():
        print(k)
        for seed in tqdm.tqdm(SEEDS):

            y_1_2 = np.zeros_like(y_true_multi)
            y_1_2[y_true_multi==1] = 1
            y_1_2[y_true_multi==2] = 1
            mask = y_1_2==1

            np.random.seed(seed)
            inds_ = np.array(range(len(mask)))

            np.random.shuffle(inds_)
            split_ind = int(len(inds_)/2.)
            inds_train, inds_test = inds_[:split_ind], inds_[split_ind:]

            H = np.copy(H_dict['H_all'])
            if k == 'H':
                proc_func=lambda x: x
            elif k == 'H_sqrt':
                proc_func=lambda x: np.sqrt(x)
            elif k == 'H_log':
                proc_func=lambda x: np.log(x+1)        
            elif k == 'H_pool':
                H = np.copy(H_dict['H_pool'])   
                proc_func=lambda x: x
            elif k == 'H_pool_sqrt':
                H = np.copy(H_dict['H_pool'])
                proc_func=lambda x: np.sqrt(x)
            elif k == 'H_pool_log':
                H = np.copy(H_dict['H_pool'])
                proc_func=lambda x: np.log(x+1)    
            elif k == 'Random':
                H = np.random.normal(0,1, H_dict['H_all'].shape)
                proc_func = lambda x: x
            elif k == 'Vgg31':
                H = np.copy(H_dict['H_vgg'])
                proc_func = lambda x: x
            elif k == 'H_single_sqrt':
                H = np.copy(H_dict['H_single'])
                proc_func = lambda x: np.sqrt(x)   
            else:
                raise

            H_proc_ = proc_func(H)

            H_test, y_test = H_proc_[inds_test].squeeze(), y_1_2[inds_test].squeeze()
            H_train, y_train = H_proc_[inds_train].squeeze(),  y_1_2[inds_train].squeeze()

            outs = eval_table_embeddings(H_train, y_train, H_test, y_test, case='min')
            res_roc[k].append(outs)

    pickle.dump(res_roc, open('data/inference/res_cluster_validation.p', 'wb'))

    
    
####### Evaluation ########

if EVALUATION:
    
    # 1. Histogram correlation of fully annotated data
    res_roc = pickle.load(open('data/inference/res_cluster_validation.p', 'rb'))
    labels = ['H_sqrt', 'H_pool_sqrt', 'H_single_sqrt', 'Vgg31']
    
    f, ax = plt.subplots(1,1, figsize=(5.6,1.8))
    plot_correlation_bar([[o[6] for o in res_roc[l]] for l in labels], labels, fax=(f,ax))
    ax.set_xlim([0.3,0.95])
    ax.set_xticks([0.3, 0.6,0.9])
    f.tight_layout()
    f.savefig(os.path.join(out_dir, 'detection_purity.png'), dpi=300, )
    plt.close()
    
    
    # 2. Cluster correlation
    res_ = pickle.load(open('data/inference/detection_hists.p', 'rb'))
    
    Xt = [x['true'] for x in res_['dense_tables']]
    Xpd = [x['pd'] for x in res_['dense_tables']]
    
    data = get_corr_data(Xt, Xpd, plot=False)
    groups = {'low':np.argwhere((np.array([d[1] for d in data])<=150)),
              'dense':np.argwhere(np.logical_and(np.array([d[1] for d in data])<=300,  np.array([d[1] for d in data])>=150)),
              'very dense': np.argwhere((np.array([d[1] for d in data])>=300))}
    
    data_df = []
    for k in ['low', 'dense', 'very dense']:
        corr = [data[int(i)][0] for i in groups[k]]
        n_bi = [data[int(i)][1] for i in groups[k]]
        n_uni = [data[int(i)][2] for i in groups[k]]
        data_ = [k, '{:0.2f}'.format(np.mean(corr)), str(int(np.sum(n_bi))),  str(int(np.sum(n_uni)))  ] 
        data_df.append(data_)
        
    df = pd.DataFrame(data_df, columns=['case', r'$\rho$', r'$N_{bigr.}$', r'$N_{uni.}$'])
    df.to_csv(os.path.join(out_dir, 'cluster_correlations.csv'))
    print('\n', df.to_latex(index=False))
    
