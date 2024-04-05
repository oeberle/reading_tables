import glob
import os
import pandas as pd
import numpy as np
import pickle
import scipy
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple

from utils import set_up_dir
from plotting.plot_corpus import  plot_by_city, plot_background, plot_evolution_in_one, plot_bar
from plotting.plot_utils import cmap_map, get_color,  add_band, shorten
from table_utils import load_image, get_hist, get_data, get_meta
from evaluation.full_annotations import get_gt_veterum_nostro


############

plot_dir_ =  'results/geographical'
set_up_dir(plot_dir_)
        
TSNE=True 
CITIES = False 
CITIES_ENTROPY = True

############

# Data preparation and loading files

res_pfile='data/inference/sphaera_hists.p'
H_all, M_all = get_data(res_pfile)

all_pages, years, books, books_unique,  book_dict = get_meta(M_all)
H_norm = np.sqrt(H_all)

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


####### 1. Plot Geographical 2D Embeddings #######

# Plot overview tsne for all cities
if TSNE:    
    
    # Get GT labels
    _, _, y_true_multi = get_gt_veterum_nostro(H_all, M_all, gt_csv = 'data/corpus/sun_zodiac.csv')
    y_1_2 = np.zeros_like(y_true_multi)
    y_1_2[y_true_multi==1] = 1
    y_1_2[y_true_multi==2] = 1


    f=plt.figure(figsize=(14,10),  dpi=200)


    ax = plt.subplot(1, 1, 1)

    scatter = ax.scatter(X_embed[:, 0], X_embed[:, 1], c='black', alpha=0.12,  edgecolors='none', s=55)

    y_1_2 = np.zeros_like(y_true_multi)
    y_1_2[y_true_multi==1] = 1
    y_1_2[y_true_multi==2] = 1
    mask = y_1_2==1
    scatter = ax.scatter(X_embed[mask==1, 0], X_embed[mask==1, 1], c='red', alpha=0.6, edgecolors=(0,0,0,0), s=70)
                
    # plot gt cluster names
    cutoff=25
    label_dict= pickle.load(open('data/inference/clid_dict_inv.p', 'rb'))

    clusters = sorted(list(set(y_pred[mask])))
    ll=0
    for cl in clusters:
        points = X_embed[y_pred==cl]
        #cl_str = str(label_dict[cl])
        cl_str = str(cl)
        xloc, yloc = np.mean(points, axis=0)
        if cl in  [1273, 1469,  910]:
            ax.text(xloc, yloc, cl_str, fontsize=16)
        else:
            y_move = {246:2.}
            y_move_ =  y_move[cl] if cl in y_move else 0.8
            ax.text(xloc+0.35*ll, yloc+(0.35*ll)**2 +y_move_, cl_str, fontsize=16)
            ll+=1
    ax.axis('off')
    f.tight_layout()
    f.savefig(os.path.join(plot_dir_, 'overview_tsne.png'), dpi=300)

    plt.close()

# Plot tsne 2D embedding for each city
if CITIES:

    cities_all = list(pd.read_csv('data/corpus/tables_wo_artifactsDuplicates.csv').city.unique())

    plot_dir_cities = os.path.join(plot_dir_, 'cities')

    set_up_dir(plot_dir_cities)


    # plot background
    f=plt.figure( figsize=(17,10), dpi=200)
    ax = plt.subplot(1, 1, 1)
    _ = plot_by_city(M=M_all, H=H_norm, X_embed=X_embed, K=K, seed=seed, metric=metric, fax=(f,ax))
    ax.axis('off')
    f.savefig(os.path.join(plot_dir_, 'background.png'), dpi=300)
    plt.show()
    plt.close()


    for city_ in cities_all:

        f=plt.figure( figsize=(17,10), dpi=200)
        ax = plt.subplot(1, 1, 1)
        _ = plot_by_city(M=M_all, H=H_norm, X_embed=X_embed, K=K, seed=seed, metric=metric, fax=(f,ax),focus_city=city_, s=150)
        ax.axis('off')        
        f.savefig(os.path.join(plot_dir_cities, '{}.png'.format(city_.replace(' ',''))), dpi=300) 
        plt.show()
        plt.close()


####### 2. Plot Geographical Entropy Scores #######

if CITIES_ENTROPY:

    df_clavius = pd.read_csv('data/corpus/clavius_books.csv', delimiter=';')
    clavius_ids = list(df_clavius.bookID)
    clavius_cities = list(set(df_clavius.Location))


    df = pd.read_csv('data/corpus/tables_wo_artifactsDuplicates.csv')
    page_id = [os.path.split(m)[1] for m in M_all]
    book_id = [int(p_.split('_')[0]) for p_ in page_id]

    cities = []

    for m in M_all:
        pid = os.path.split(m)[1] 
        bid = int(pid.split('_')[0])
        cs = df[df['bookId']==bid].city.unique()
        assert len(cs) == 1
        cities.append(cs[0])

    cities_map = {c:i  for i,c in enumerate(list(set(cities)))}
    cities_map_inv = {v:k for k,v in cities_map.items()}
    cities_num = np.array([cities_map[c] for c in cities])
    N_cities = len(cities_map)

    entropy = {}

    for c in list(set(cities_num)):

        cities_num_ = np.copy(cities_num)
        #cities_num_[np.array(clavius_mask)==1]=-1

        unique_clusters_per_city = len(list(set(y_pred[cities_num_==c])))
        tables_per_city = len(list(y_pred[cities_num_==c]))

        books_per_city = list(set(books[cities_num_==c]))


        # Compute bag of clusters 
        hh = get_hist(y_pred[cities_num_==c], K=len(list(set(y_pred))))
        h_single = np.array(hh>0, dtype=np.float32)    

        e_difference = scipy.stats.entropy(hh) - np.log(int(np.sum(hh)))

        entropy[c] = (c,  hh, scipy.stats.entropy(h_single), e_difference, unique_clusters_per_city, tables_per_city, books_per_city)


    cities_= np.array(range(N_cities))

    labels = [cities_map_inv[c] for c in cities_]
    labels = np.array([ l + ' *' if l in clavius_cities else l for l in labels])
    es = np.array([entropy[c][3] for c in cities_])
    n_tables_per_city = np.array([entropy[c][5] for c in cities_])
    idx_ = np.array(range(len(es)))

    y =  es[idx_]/np.log(2)

    f, ax = plt.subplots(1,1, figsize=(6.,3.5))
    plot_bar(y, labels, n_tables_per_city, fax=(f,ax)) 

    f.tight_layout()
    f.savefig(os.path.join(plot_dir_, 'location_entropy.png'), dpi=300, transparent=False)
    plt.close()

