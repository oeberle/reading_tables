import os
import numpy as np
import matplotlib
import palettable
import matplotlib.pyplot as plt
from matplotlib import colors
import scipy
import itertools
from plotting.plot_utils import shorten, get_color
from plotting.plot import plot_tsne, add_cluster_labels, plot_tsne_from_Xembed
from table_utils import *

        
def plot_by_book(M, H=None, X_embed=None, K=None, seed=1, metric=None, fax=None, focus_book=None, titlestr='',**kwargs):
    # Plot 2D projection with points colored by book 

    if fax:
        f,ax = fax
    else:
        f,ax = plt.subplots(1,1, figsize=(20,10), dpi=200)

    colmap = matplotlib.cm.get_cmap('rainbow')
    colmap = matplotlib.colors.ListedColormap(np.random.rand(256,3))
    books = [get_book_page(m)[0]for m in M]
    book_dict = {y:i for i,y in enumerate(sorted(list(set(books))))}
    book_dict_inv = {v:k for k,v in book_dict.items()}
    boks = [book_dict[y] for y in books]

    cdict = get_color(len(list(set(boks))))

    if focus_book is not None:        
        idx_book = [book_dict[fb] for fb in focus_book] 
        cdict = {k:v if k in idx_book else '#D3D3D326' for k,v in cdict.items()}
        cdict2 = {k:'#ffffff' if k in idx_book else '#D3D3D326' for k,v in cdict.items()}

    plot_tsne_from_Xembed(fax = (f,ax) , y_pred = np.array(boks), point_colors = None, X_embedded_tsne=X_embed, cmap=None, alpha=.5, cdict=cdict)
    
    if focus_book is not None:     
        colmap_blues = matplotlib.cm.get_cmap('ocean_r')    
        
        cols = ['#031A6B', '#77ede5', '#6DAEDB', '#519686'] #56A3A6'] ##055878' Now #05665e
        scats = []
        
        ii = 0
        for fb in focus_book:
            mask = np.array(books) == fb
            _=[scats.append((x_,y_,cols[ii], fb)) for x_,y_ in zip(X_embed[mask,0], X_embed[mask,1])]
            ii+=1    
            
        inds_ = np.array(list(range(len(scats))))
        np.random.shuffle(inds_)
        
        used_labs = []
        l=0
        for j in inds_:
            xx,yy,c,lab = scats[j]
            if lab not in used_labs:
                ax.scatter(xx,yy, color = c, s=130, label=lab,edgecolor='#4c4c4c', alpha=0.9)
                used_labs.append(lab)
            else:
                ax.scatter(xx,yy, color = c, s=130,edgecolor='#4c4c4c',alpha=0.9)
            l+=1

        plt.legend()
        
    ax.set_title( label=titlestr)
    if not fax:
        plt.show()      

def plot_by_year(M, H=None, y_pred=None, fax=None,  X_embed = None, titlestr='', cmap='cool', **kwargs):
    # Plot 2D projection with points colored by publishing year 
    
    if fax:
        f,ax = fax
    else:
        f,ax = plt.subplots(1,1, figsize=(20,10), dpi=200)
    
    _, years, _, _,  _ = get_meta(M)

    if isinstance(cmap, str):
        colmap = matplotlib.cm.get_cmap(cmap)
    else:
        colmap = cmap
        
    plot_tsne_from_Xembed(fax = (f,ax) , y_pred = y_pred, point_colors = years, X_embedded_tsne=X_embed, cmap=colmap) #, alpha=.5, cdict=cdict)

    ax.set_title( label=titlestr)
    if not fax:
        plt.show()

    
        
def plot_by_digit_density(H, X_embed=None, y_pred=None, fax=None,  titlestr='',**kwargs):
    # Plot 2D projection with points colored by digit density 
    
    if fax:
        f,ax = fax
    else:
        f,ax = plt.subplots(1,1, figsize=(20,10), dpi=200)
    
    colmap = matplotlib.cm.get_cmap('viridis')
    digits = [np.sum(m) for m in H]

    plot_tsne_from_Xembed(fax = (f,ax) , y_pred = y_pred, point_colors = digits, X_embedded_tsne=X_embed, cmap=colmap)    
    ax.set_title( label=titlestr)
    
    if not fax:
        plt.show()
        
def plot_by_cluster_size(y_pred=None, X_embed=None, fax=None,  titlestr='',**kwargs):
    # Plot 2D projection with points colored by size of the cluster 

    if fax:
        f,ax = fax
    else:
        f,ax = plt.subplots(1,1, figsize=(20,10), dpi=200)
    
    colmap = matplotlib.cm.get_cmap('viridis')
    cmap = palettable.scientific.sequential.Imola_7.mpl_colormap
    clustersizes = [sum(y_pred==y) for y in y_pred]
    n=len(list(set(clustersizes)))
    colmap = cmap.from_list('colmap',[(1,0,0,1)] + list(map(cmap,range(n))), N=n+1)
    plot_tsne_from_Xembed(fax = (f,ax) , y_pred = y_pred, point_colors = clustersizes, X_embedded_tsne=X_embed, cmap=colmap) 
    
    ax.set_title( label=titlestr)
    
    if not fax:
        plt.show()
        
        
def plot_by_city(M, H=None, X_embed=None, K=None, seed=1, metric=None,  fax=None,  titlestr='', focus_city=None, s=None, ref_scatter=True,plot_cbar=True, scatter_pad=0, **kwargs):
    # Plot 2D projection with points colored by print location
    
    if fax:
        f,ax = fax
    else:
        f,ax = plt.subplots(1,1, figsize=(20,10), dpi=200)
    
    df = pd.read_csv('data/corpus/tables_wo_artifactsDuplicates.csv')
    page_id = [os.path.split(m)[1] for m in M]
    book_id = [int(p_.split('_')[0]) for p_ in page_id]

    cities = []

    for m in M:
        pid = os.path.split(m)[1] 
        bid = int(pid.split('_')[0])
        cs = df[df['bookId']==bid].city.unique()

        assert len(cs) == 1
        cities.append(cs[0])

    city_dict = {y:i for i,y in enumerate(sorted(list(set(cities))))}
    city_dict_inv = {v:k for k,v in city_dict.items()}
    cs = np.array([city_dict[y] for y in cities])
    cdict = get_color(len(list(set(cs))))

    if focus_city is not None:
        idx_city = city_dict[focus_city]
        cdict = {k:v if k==idx_city else '#D3D3D326' for k,v in cdict.items()}
        #fully transparent
        city_mask = cs==idx_city

        cmap = colors.ListedColormap(list(cdict.values()))
        bounds=list(range(len(cdict)+1))
        norm = colors.BoundaryNorm(bounds, cmap.N)
        scatter = ax.scatter(X_embed[city_mask, 0], X_embed[city_mask, 1], s=80 if s is None else s, c=np.array(cs)[city_mask],  cmap=cmap, norm=norm, edgecolor='#4c4c4c', alpha=1.0)
        
        # add reference scatters:
        xmin,xmax = np.min(X_embed[:,0])-scatter_pad,  np.max(X_embed[:,0])+scatter_pad
        ymin,ymax = np.min(X_embed[:,1])-scatter_pad,  np.max(X_embed[:,1])+scatter_pad
        
        _ = ax.scatter([xmin,xmax],[ymax, ymin], c='black' if ref_scatter else 'white', s=100,  edgecolor='none', marker='+')
            
        if plot_cbar:
            cbar = f.colorbar(scatter) #, ax=ax) #, cax=ax)

    else:
       # as background
        scatter = ax.scatter(X_embed[:,0], X_embed[:,1],  c='#D3D3D326', edgecolor='none', alpha=1.0)
        xmin,xmax = np.min(X_embed[:,0])-scatter_pad,  np.max(X_embed[:,0])+scatter_pad
        ymin,ymax = np.min(X_embed[:,1])-scatter_pad,  np.max(X_embed[:,1])+scatter_pad
        
        _ = ax.scatter([xmin,xmax],[ymax, ymin], c='black' if ref_scatter else 'white', s=100,  edgecolor='none',marker='+')

        if plot_cbar:
            cbar = f.colorbar(scatter)
        
    if plot_cbar:

        cbar.ax.get_yaxis().set_ticks([])
        for j, lab in enumerate(sorted(list(set(cs)))):
            text = city_dict_inv[lab]
            cbar.ax.text(35., j+0.5, text, ha='left', va='center')

    ax.set_title( label=titlestr)
    if not fax:
        plt.show()
        
        

def plot_background(M, H=None, X_embed=None, K=None, seed=1, metric=None,  fax=None,  titlestr='', focus_city=None, s=None, **kwargs):
    
    if fax:
        f,ax = fax
    else:
        f,ax = plt.subplots(1,1, figsize=(20,10), dpi=200)
    
    scatter = ax.scatter(X_embed[:,0], X_embed[:,1],  c='black', edgecolor='none', alpha=0.1)
    xmin,xmax = np.min(X_embed[:,0]),  np.max(X_embed[:,0])
    ymin,ymax = np.min(X_embed[:,1]),  np.max(X_embed[:,1])
    _ = ax.scatter([xmin,xmax],[ymax, ymin], c='black', s=100,  edgecolor='none',marker='+')
    cbar = f.colorbar(scatter)
    cbar.ax.get_yaxis().set_ticks([])
    ax.set_title( label=titlestr)
    
    if not fax:
        plt.show()
            

def plot_evolution_in_one(all_data, fax=None, kwargs={}, keys = None, cdict={}, legs=[], color=None, ls='-'):
    # Plot temporal evolution of entropy scores
    
    if fax:
        f,axs = fax
    else:
        raise

    i=0
    if keys is None:
        keys = list(all_data.keys())
        
    plot_legs = True if len(legs)>0 else False
        
    entropies = {}
    for k in keys:
        v = all_data[k]
        thresh = k[0]
        temps = v[0]
        e = np.nanmean(v[1],axis=0)
        counts = np.nanmean(v[2],axis=0)
        c_ = color if color is not None else cdict[k[0]]
        l, = axs.plot(temps, e, c=c_, label=str(k[1]), linestyle=ls)

        entropies[k[0]] = e 
        legs.append(l)

    if fax is None:
        plt.show()

    return legs, entropies


def plot_bar(x, labels, n_tables, fax=None, sort='alphabetical'):
    # Plot geographical entropy scores

    if fax:
        f,ax = fax
    else:
        f, ax = plt.subplots(1,1, figsize=(6,3))
        
    labels= shorten(labels)
    
    if sort=='alphabetical':
        inds_ = np.argsort(labels)
        
    elif sort == 'value':
        inds_ = np.argsort(x)
    else:
        raise
    
    labels_ = np.array(labels)[inds_]
    x_ = x[inds_]  
    n_tables_ = n_tables[inds_]

    c_bars =  np.copy(np.array(['#0a0a0d']*(len(labels))))
    c_bars[n_tables_<=100]= '#d5d5d5'
        
    ax.bar(labels_, x_, align='center', color=c_bars , width=0.75)

    ax.tick_params(axis='x', which='major',  rotation=90)
    ax.tick_params(axis='x', which='minor', rotation=90)
    
    ax.tick_params(axis='y', which='major',  rotation=180,  labelsize=13)
    ax.tick_params(axis='y', which='minor', rotation=180,  labelsize=13)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels = labels_, rotation=90,  ha="center")

    ax.set_yticks(np.arange(-3.0, 0.5, 1.))
    ax.set_yticklabels(labels = ['{:0.1f}'.format(x) for x in np.arange(-3.0, 0.5, 1.)]) 

    ax.set_ylabel(r'$H(p)-H(p_{\max})$', fontsize=17) 
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    ax.xaxis.tick_top() 
    ax.xaxis.set_label_position('top')
    
    if fax is None:
        plt.show()
        
        
def plot_correlation_bar(x, labels, fax=None):
    # Plot correlation bars to compare different embedding approaches

    def proc_label(x):
        x = x.replace('x_sqrt','')
        return x
        
    if fax:
        f,ax = fax
    else:
        f, ax = plt.subplots(1,1, figsize=(6,3))
    
    if False:
        x, labels = x[::-1], labels[::-1]

        if  'Random' in names:
            ind_= labels.index('Random')
            ax.vlines(np.mean(x[ind_]), ymin=0-0.5, ymax=len(names)-1.5, linestyle='--')

            X_ = x if ind_==0 else x[:-1]
            ax.barh(labels[1:] if ind_==0 else labels[:-1], [np.mean(x_) for x_ in X_],xerr=  [np.std(x_) for x_ in X_], align='center', color='#808080')
        else:
            ax.barh(labels, [np.mean(x_) for x_ in x], xerr= [np.std(x_) for x_ in x], align='center', color='#808080')

    x_ = np.array([np.mean(x_) for x_ in x])
    err_  = np.array([np.std(x_) for x_ in x])
    inds_ = np.argsort(x_)
        
    width = 0.6
    labels =  [proc_label(l) for l in labels]
    
    ax.barh(np.array(labels)[inds_], x_[inds_], xerr=err_[inds_], align='center', color= ['#d1d1d1']*(len(inds_)-1) + ['#6495ed'], height=width)
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=20)
    
    i=-0.3
    for k,v in zip(np.array(labels)[inds_], x_[inds_]):
        ax.text(v+0.085, i, '{:0.2f}'.format(v), fontsize=18)    
        i+=1
        
    if fax is None:
        plt.show()
        

def plot_corrs(X, X2, plot_dir = None,    cmap = 'Reds'):
    # Plot correlation bars of fully annotated data 

    def count_uni(x):
        count = 0
        for i,x_ in enumerate(x):
            if i < 10:
                count+=(1*x_)
            else:
                count+=(2*x_)
        return count
    
    X_flat = np.array(list(itertools.chain(*X)))
    X2_flat = np.array(list(itertools.chain(*X2)))
    
    n_bigram = np.sum(X_flat)
    n_singles = np.sum([count_uni(x) for x in X])
    p_corr = np.mean([scipy.stats.pearsonr(x1, x2)[0] for x1,x2 in zip(X, X2)])
    p_corr_flat = scipy.stats.pearsonr(X_flat, X2_flat)[0]
    print('N_digits', np.sum(X_flat))
    print('Pearson_r', p_corr)
    print('Pearson_r flat', p_corr_flat)

    f, axs = plt.subplots(2,1, figsize=(6,1.5))

    axs[0].imshow(X, cmap=cmap)
    cbar = axs[1].imshow(X2, cmap=cmap)
    [ax.axis('off') for ax in axs.flatten()]
    f.subplots_adjust(hspace=0.2, wspace=0.1)
    f.suptitle(r'$\rho$: {}, $\rho_c$: {}, $N_{{bigr}}$: {}'.format('{:0.2f}'.format(p_corr),'{:0.2f}'.format(p_corr_flat),
                                                                          n_bigram), y=1.0)
    axs[0].text(-6, 8, 'true', rotation=90)    
    X_diff = abs(np.array(X)-np.array(X2))

    if plot_dir:
        f.savefig(os.path.join(plot_dir, 'hist_comp.png'), dpi=300)
    
    plt.show()

    f, axs = plt.subplots(1,1)
    plt.colorbar(cbar)
    
    axs.axis('off')
    if plot_dir:
        f.savefig(os.path.join(plot_dir, 'hist_cbar.png'), dpi=250)
    plt.show()
    
    data = []
    for xt, xp in zip(X, X2):
        data.append([scipy.stats.pearsonr(xt, xp)[0], np.sum(xt), np.sum(count_uni(xt)) ] )
        
    return data

        