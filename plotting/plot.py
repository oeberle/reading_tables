import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
from PIL import Image
from matplotlib import gridspec
import matplotlib
from matplotlib import colors
from table_utils import *
from plotting.plot_utils import *
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

def embed(X, case = 'pca', perplexity=100, seed=1):
    if case=='pca':
        mapping = PCA(n_components = 2)
    elif case=='tsne':
        mapping = TSNE(n_components=2, verbose=0, perplexity=perplexity, random_state=seed)
    dataset_2D = mapping.fit_transform(X)
    return dataset_2D

def round_ten(number):
    return int(np.floor((number/10))*10)

def scatter_digits(ax, PDigits, fontsize, alpha=0.7):
    for item in PDigits:
        ax.text(item[2], item[1], nr_from_index(int(item[0])), fontdict=None, fontsize=fontsize, alpha=0.7) 
    return None

def plot_hist(ax, h, color_hist=False, cdict=None):
    if cdict is None:
        cdict = {i:'black' for i in range(len(h))}
        
    n, bins, patches = ax.hist(range(len(h)), weights = h, bins=len(h), edgecolor="black", linewidth=0.25)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    # scale values to interval [0,1]
    col = bin_centers - min(bin_centers)
    col /= max(col)

    j=0
    cmap_control = {}
    for c, p in zip(col, patches):        

        color = cdict[j]
        plt.setp(p, 'facecolor',color if color_hist else 'black', alpha=0.4 if color_hist else 0.75)
        
        if color in cmap_control:
            cmap_control[color] += 1
        else:
            cmap_control[color] = 1
        j+=1
        
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    ticks= np.arange(0,110,10)
    ax.set_xticks(ticks)
    ax.set_xticklabels([nr_from_index(i) for i in ticks])
    assert sum(cmap_control.values())==110
    
    return None
   

def plot_rainbow(Im, Hist, PDigits, plot_text=False, fax=None, title=None, fontsize=2, cmap =matplotlib.cm.get_cmap('jet'), color_hist = True, plot_scatter=True):
    
    if fax is None:
        f = plt.figure(dpi=200)
        gs = gridspec.GridSpec(2,2, width_ratios=[0.5, 0.5], height_ratios=[0.85,0.15]) 
        gs_inds = [0,2]
    else:
        f, gs, gs_inds = fax
    
    #pages
    ax0 = plt.subplot(gs[gs_inds[0]])
    ax0.imshow(Im , cmap='Greys', alpha=0.45)
    if plot_text:
        scatter_digits(ax0, PDigits, fontsize, alpha=0.7)
            
    if title:
        ax0.set_title(title) 
            
    assert len(Hist)==110
    
    cdict = {i:cmap(round_ten(i)/110.) for i in range(110)}
    
    num_inds, ys, xs =  list(zip(*PDigits))
    num_inds_float=np.array(num_inds)/110.
    cs = np.array([cdict[num] for num in num_inds])
        
    if plot_scatter:
        hh0 = ax0.scatter(xs,ys, color=cs, alpha=0.6, s=7) #, edgecolors=(1,1,1,0))

    ax0.axis('off')
    h = Hist
    ax1 = plt.subplot(gs[gs_inds[1]])
    plot_hist(ax1,h,color_hist, cdict)
    
    f.tight_layout()
    return f

def add_cluster_labels(X2d,y, ax, label_dict=None, cutoff=0.):
    
    clusters = sorted(list(set(y)))
    for cl in clusters:
        points = X2d[y==cl]
        
        if label_dict:
            cl_str = str(label_dict[cl])
        else:
            cl_str = str(cl)
        
        xloc, yloc = np.mean(points, axis=0)
        
        if sum(y==cl)>=cutoff:
            ax.text(xloc, yloc, cl_str, fontsize=7)
    return None
        
def get_embedded(X, perplexity,  seed=1, metric='euclidean'):
    X_embedded_tsne = TSNE(n_components=2, verbose=0, perplexity=perplexity, random_state=seed, metric=metric).fit_transform(X)
    return X_embedded_tsne


def plot_tsne_from_Xembed(fax = None , y_pred = None, point_colors = None, X_embedded_tsne=None, cmap=None, alpha=1., cdict=None):
    
    f, ax = fax
    
    if cdict is None:
        cdict = get_color(len(list(set(y_pred))))

    if point_colors is not None:
        c = point_colors
    else:
        c = [cdict[y] for y in y_pred]
    
    if cmap:
        vmin, vmax = np.min(c), np.max(c)
        scatter = ax.scatter(X_embedded_tsne[:, 0], X_embedded_tsne[:, 1], c=c,  edgecolor='none', cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.65)
        colorbar = f.colorbar(scatter, ax=ax)
    else:
        cmap = colors.ListedColormap(list(cdict.values()))
        bounds=list(range(len(cdict)+1))
        norm = colors.BoundaryNorm(bounds, cmap.N)
        scatter = ax.scatter(X_embedded_tsne[:, 0], X_embedded_tsne[:, 1], c=y_pred,  cmap=cmap, norm=norm,  edgecolor='none', alpha=alpha)
        colorbar = f.colorbar(scatter) 

    colorbar.ax.tick_params(labelsize=20)
    return X_embedded_tsne,colorbar 

    
def cluster(X,K,seed, compute_vars_dsts = False):
    kmean = KMeans(n_clusters=K, random_state=seed)
    y_pred = kmean.fit_predict(X)
    C = kmean.cluster_centers_
    
    if compute_vars_dsts:
        variances = get_variances(X, C, y_pred)
        distances = get_single_distances(X,C,y_pred)
        return y_pred, C, variances, distances, 
    else:
        return y_pred, C    

    
def plot_tsne(X, K, perplexity,  seed=1, fax = None , y_pred = None, standardize=False, point_colors = None, metric=None, X_embedded_tsne=None, cmap=None, alpha=1., cdict=None):
    
    if standardize:
        print('Standardizing')
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        print('No Standardizing ')
    
    if y_pred is None:
        y_pred, centers = cluster(X,K, seed=seed)
    else:         
        if cdict is None:
            cdict = get_color(len(list(set(y_pred))))
            
    if fax:
        f,ax = fax
    else:
        f,ax = plt.subplots(1,1, figsize=(20,10), dpi=200)

    if point_colors is not None:
        c = point_colors
    else:
        c = [cdict[y] for y in y_pred]
    
    if X_embedded_tsne is None:
        X_embedded_tsne = TSNE(n_components=2, verbose=0, perplexity=perplexity, random_state=seed, metric=metric).fit_transform(X)
        
    if cmap:
        vmin, vmax = np.min(c), np.max(c)
        scatter = ax.scatter(X_embedded_tsne[:, 0], X_embedded_tsne[:, 1], c=c,  edgecolor='none', cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.5)
        colorbar = f.colorbar(scatter, ax=ax)

    elif point_colors is not None:
        scatter = ax.scatter(X_embedded_tsne[:, 0], X_embedded_tsne[:, 1], c=c,  edgecolor='none', alpha=0.5)
        colorbar = f.colorbar(scatter, ax=ax)
    else:
        cmap = colors.ListedColormap(list(cdict.values()))
        bounds=list(range(len(cdict)+1))
        norm = colors.BoundaryNorm(bounds, cmap.N)
        scatter = ax.scatter(X_embedded_tsne[:, 0], X_embedded_tsne[:, 1], c=y_pred,  cmap=cmap, norm=norm,  edgecolor='none', alpha=alpha)
        colorbar = f.colorbar(scatter) 

    ax.set_title(perplexity)
    return X_embedded_tsne,colorbar, y_pred


def plot_layers(ref_page, res_dict, plot_dir=None):
    
    img, hist, pdigit = res_dict['img'], res_dict['hist'], res_dict['pdigit']
    H_single, H_grams = res_dict['H_single'], res_dict['H_grams']

    if plot_dir:
        pageid = ref_page.split('/')[-1].replace('.jpg','')
        plot_dir_page = os.path.join(plot_dir,'{}__'.format(h.sum())+ pageid )
        set_up_dir(plot_dir_page)

    x_gray = load_image(ref_page, gray=True)
    x_color = load_image(ref_page.replace('processed', 'raw'), gray=False)

    f,ax = plt.subplots(1,1 , dpi=200)
    ax.imshow(x_gray)
    ax.axis('off')
    if plot_dir:
        f.savefig(os.path.join(plot_dir_page, 'gray.png'), dpi=300, transparent=True)
        plt.close()
    else:
        plt.show()

    f,ax = plt.subplots(1,1 , dpi=200)
    ax.imshow(x_color)
    ax.axis('off')
    if plot_dir:
        f.savefig(os.path.join(plot_dir_page, 'color.png'), dpi=300, transparent=True)
        plt.close()
    else:
        plt.show()
        
    f,ax = plt.subplots(1,1 , dpi=200)
    ax.imshow(img, cmap='Greys', alpha=0.2)
    scatter_digits(ax, pdigit, 4)
    ax.axis('off')
    
    if plot_dir:
        f.savefig(os.path.join(plot_dir_page, 'bigrams.png'), dpi=300, transparent=True)
        plt.close()
    else:
        plt.show()
        
    f,ax = plt.subplots(figsize=(6,3))
    plot_hist(ax, hist,color_hist = False, cdict = {j:'black' for j in range(110)})
    if plot_dir:
        f.savefig(os.path.join(plot_dir_page, 'pd_hist.png'), dpi=200, transparent=True)
        plt.close()
    else:
        plt.show()
        
    f, ax = plt.subplots(1,1, dpi=200)
    ax.imshow(img,alpha=0.0)
    X = H_grams.sum(0)
    ax.imshow(X, cmap='RdGy_r', vmin=-200, vmax=200)
    ax.imshow(img, cmap='Greys', alpha=0.2)
    ax.axis('off')
    
    if plot_dir:
        f.savefig(os.path.join(plot_dir_page, 'bigram_activity.png'), dpi=300, transparent=True)
        plt.close()
    else:
        plt.show()
    
    f, ax = plt.subplots(1,1, dpi=200)
    X = H_single.sum(0)
    h0= ax.imshow(X, cmap='RdGy_r', vmin=-np.max(X), vmax=np.max(X))
    ax.imshow(img, cmap='Greys', alpha=0.2)#, vmin=-200, vmax=200)
    ax.axis('off')
    
    if plot_dir:
        f.savefig(os.path.join(plot_dir_page, 'single_activity.png'), dpi=300, transparent=True)
        plt.close()
    else:
        plt.show()


def plot_2D(X, y_cl, case = 'pca', perplexity=100, seed=1, colors='black',
            markers='o', attractors=[], fax=None, mask=None, K=None, s=20,
           dataset_2D = None):

    if K is None:
        K=len(list(set(y_cl)))
        
    cdict = get_color(K)
    colors = np.array([cdict[y_] for y_ in  y_cl])

    if dataset_2D is None:
        if case=='pca':
            mapping = PCA(n_components = 2)
        elif case=='tsne':
            mapping = TSNE(n_components=2, verbose=0, perplexity=perplexity, random_state=seed)
        dataset_2D = mapping.fit_transform(X)
    
    if fax is None:
        f,ax = plt.subplots(1,1, figsize=(8,6), dpi=50)
    else:
        f,ax = fax
                      
    if mask is None:
        mask = np.ones(len(X))

    if len(attractors)>0:
        ax.scatter(dataset_2D[:-2, 0], dataset_2D[:-2, 1], c=np.array(colors), s=s, marker =markers)
        ax.scatter(dataset_2D[-2:, 0], dataset_2D[-2:, 1], c=['m', 'c'], s=80)
    else:        
        ax.scatter(dataset_2D[:,0], dataset_2D[:,1], alpha=0.007, s=11, c='black', edgecolors='none')    
        hh = get_hist(y_cl[mask==1], K=len(list(set(y_cl))))
        hh_sort = np.argsort(hh)
        
        for j, h_ in enumerate(hh_sort):
            
            y_mask = y_cl==h_
            joined_mask = np.logical_and(y_mask, mask==1)
        
            if h_>0:
                ax.scatter(dataset_2D[joined_mask==1, 0], dataset_2D[joined_mask==1, 1], alpha=0.3, 
                           c=colors[joined_mask==1], s=s, marker =markers)    

    if fax is None:
        plt.show()
        
    return dataset_2D