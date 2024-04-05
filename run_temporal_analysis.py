import glob
import os
import pandas as pd
import numpy as np
import pickle
import scipy
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple

from utils import set_up_dir
from plotting.plot_corpus import plot_evolution_in_one
from evaluation.corpus_inference import entropy_exp
from plotting.plot_utils import cmap_map, get_color,  add_band, shorten
from table_utils import load_image, get_hist, get_data, get_meta


############
plot_dir_ =  'results/temporal'
set_up_dir(plot_dir_)

PROC = True
EVOLUTION = True

############


# Data preparation and loading files

res_pfile='data/inference/sphaera_hists.p'
H_all, M_all = get_data(res_pfile)

all_pages, years, books, books_unique,  book_dict = get_meta(M_all)

K=1500
seed=1
metric='euclidean'
cl_dict_dir = 'data/inference/cl_dict.p'

bins_list= np.arange(1494, 1647+1)
cl_dict = pickle.load(open(cl_dict_dir, 'rb'))
y_pred, _,_,_  = cl_dict[K]

embed_dir = 'data/inference/embeds.p'
perplexity=500
X_embed = pickle.load(open(embed_dir, 'rb'))[perplexity]


####### 1. Proc Temporal Evolution #######

        
if PROC:
    t_sigma=4
    n_samples=80
    Ks = [1500]
    
    # Sphaera data
    all_data_ = entropy_exp(H_all, M_all, books, years, cl_dict = cl_dict_dir, Ks  = Ks, n_sims=20, t_sigma=t_sigma, n_samples=n_samples)


    # Sphaera data with Fine-5 removed
    books_to_filter = ['2252_fine_sphaera_1552', 
                   '2253_fine_sphaera_1555', 
                   '2257_fine_sphaera_1551', 
                   '2269_fine_sphere_1551',
                   '2270_fine_sphere_1552']

    book_filter = np.array([False]*len(books))
    for b in books_to_filter:
        mask = books==b
        book_filter[mask] = True

    all_data_book_removed = entropy_exp(H_all, M_all, books, years, cl_dict = cl_dict_dir, Ks  = Ks, n_sims=20, 
                                        t_sigma=t_sigma, n_samples=n_samples, extra_filter=book_filter) 

    res_evolve = {'sphaera': all_data_ , 'sphaera_fine_removed': all_data_book_removed }
    pickle.dump(res_evolve, open(os.path.join('data/inference/evolution.p'), 'wb'))


    
####### 2. Plot Temporal Evolution #######

if EVOLUTION:
    res_ = pickle.load( open(os.path.join('data/inference/evolution.p'), 'rb'))
    all_data_rand = pickle.load( open(os.path.join('data/inference/baseline_evolution.p'), 'rb'))

    all_data = res_['sphaera']
    all_data_book_removed =  res_['sphaera_fine_removed']
    
    
    t0= (1520,1528) 
    t1 = (1548,1556)
    t2 = (1605,1613)
    
    tmask = -1*np.ones(len(H_all))
    tmask[np.logical_and(years>=t0[0], years<t0[1])] = 0.
    tmask[np.logical_and(years>=t1[0], years<t1[1])] = 1.
    tmask[np.logical_and(years>=t2[0], years<t2[1])] = 2.
    
    keys_ = [k[0] for k in all_data[K].keys()]
    
    
    cmap_bone  = plt.get_cmap('bone_r')
    cmap_bone = cmap_map(lambda x: -0.05+x*0.95,cmap_bone)
    cdict_random = {k:cmap_bone(k/(np.max(keys_))) for k in sorted(keys_)}
    
    f=plt.figure( figsize=(6,3), dpi=200)
    ax = plt.subplot(1, 1, 1)
    
    add_band(ax, [t0,t1,t2], facecolor='#b3c0cc', alpha=0.32)
    cmap  = plt.get_cmap('cool')
    cmap = cmap_map(lambda x: x*0.75,cmap)
    cdict = {k:cmap(k/(np.max(keys_))) for k in sorted(keys_)}
    legs_,_ = plot_evolution_in_one(all_data_rand[K], fax=(f,ax), cdict=cdict_random)
    legs_,_ = plot_evolution_in_one(all_data[K], fax=(f,ax), cdict=cdict, legs=legs_)
    
    
    _,_ = plot_evolution_in_one({(300, 'euclidean'): all_data_book_removed[K][(300, 'euclidean')]}, 
                                fax=(f,ax), cdict=cdict, legs=legs_,
                               ls='--')
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('timestep t')
    ax.set_ylabel('entropy')
    
    plt.legend( list(zip(legs_[len(keys_):],legs_[:len(keys_)])), 
           [str(k_) for k_ in keys_],
          handler_map={tuple: HandlerTuple(ndivide=None)},
           bbox_to_anchor=(0.985, 0.54), borderpad=0.2,  prop={'size': 11}) 
    f.tight_layout()
    f.savefig(os.path.join(plot_dir_, 'evolution.png'), dpi=300, transparent=False)
    plt.show()