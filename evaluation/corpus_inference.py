import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import sklearn.metrics
import collections
import os
import scipy
import scipy.stats as stats
from evaluation.clustering import cluster

def collect_year_distribution(H_all, years, thresholds):
    # Get distribution of years over the corpus
    
    ydict = {}
    for thresh in thresholds:
        subset =  H_all.sum(1)>thresh
        years_filt= years[subset]
        ydict[thresh] = years_filt
    return ydict


def get_data_gausswindow(years, H_all, M_all, orig_inds, books, plot=False, t_sigma=5, n_samples =100, tstart = 1494, tend=1647):
    # Get gaussian sampling window
    
    bin_centers = np.arange(tstart, tend+1)
    
    if plot:
        f,ax = plt.subplots(1,1)
    data = {}
    scats = []
    sampled = []
    for t_center in bin_centers: #[:-3]:

        t_idx = int(np.where(bin_centers==t_center)[0])        
        p_=stats.norm.pdf(bin_centers, t_center, t_sigma)
        p_all = {k:v for k,v in  zip(bin_centers,p_)}
        p_all = np.array([p_all[t_] for t_ in years])
        p_all[np.logical_and(years>=t_center+t_sigma, years<t_center-t_sigma)] = 0.
        p_all = p_all/np.sum(p_all)
        

        out = np.random.choice(list(range(len(years))), size=n_samples, replace=False, p=p_all)
            
        t_max = collections.Counter(years[out]).most_common()[0][0]
        t_min_sample = np.min(years[out])
        t_max_sample = np.max(years[out])
        if plot:
            ax.scatter(t_center, t_max, c='grey', alpha=0.5)
        
        scats.append([t_center, t_max])
        sampled.append(orig_inds[out])
        
        mask = out
        Ht = H_all[mask]
        Bt = books[mask]
        Mt = M_all[mask]
        data[(t_center,t_center)] = (Ht, Bt, Mt)

    if plot:
        ax.set_xlim([1494, 1647+1])
        ax.set_ylim([1494, 1647+1])
        ax.set_xlabel('center t')
        ax.set_ylabel('max sampled t')
        ax.plot(bin_centers, bin_centers, c='r')
    return data, scats, sampled



def get_evolve_historic_data_threshed(M, data, years, y_pred, thresh, K, plot=False ):
    # apply different digits density thresholds
    
    cl_pages = {k:v for k,v in zip(M, y_pred)}
    times = sorted(data.keys(), key= lambda x: x[0])

    for nearest in [None]:
        evolve_data = {}
        for metric in ['euclidean']:
            
            if plot:
                f,axs = plt.subplots(2,1, figsize=(4,3), gridspec_kw={'height_ratios':[0.7,0.3]})

            sims = []
            counts = []

            entropy = []
            entropy2 = []
            entropy_cl = []

            e_vec = np.zeros(K)
            CLS_t = []
            P_t = []
            for t in times[:-1]:

                e_vec2 = np.zeros(K)

                Ht_ = data[t][0]
                Pt =  data[t][2]
               # import pdb;pdb.set_trace()
                cls_t =  [cl_pages[x] for x in Pt]
                p_t = [p_ for p_ in Pt]
                cls_years = [years[y_pred==c] for c in cls_t]

                for c_  in cls_t:
                    e_vec[c_]+=1
                    e_vec2[c_]+=1

                entropy.append(scipy.stats.entropy(e_vec))
                entropy2.append(scipy.stats.entropy(e_vec2))

                CLS_t.append(cls_t)
                P_t.append(p_t)
                counts.append(len(Ht_))

            evolve_data[(thresh, metric)] = (times, entropy, entropy2, CLS_t, P_t, counts)

            if plot:

                axs[0].plot([t[0] for t in times[:-1]], entropy)
                axs[0].plot([t[0] for t in times[:-1]], entropy2, color='r')

                axs[1].plot([t[0] for t in times[:-1]], counts)
                axs[0].set_title(thresh)
                axs[0].set_title(thresh)
                axs[0].set_xticklabels(['']*len(list(axs[0].get_xticklabels())))
                
            if plot:
                plt.show()
        
        
    return evolve_data


def entropy_exp(H_all,M_all, books, years_input, n_sims = 1, n_samples = 70, cl_dict = None,
                Ks  = [500, 1000, 1500, 2000], 
                thresholds = [0,100,200,250,300],
                t_sigma = 5, H_cluster=None, compute_clusters = True, extra_filter=None,  seed=1):
    # Compute entropy scores over time

    if isinstance(cl_dict, str):
        cl_dict = pickle.load(open(cl_dict, 'rb'))
    elif compute_clusters ==True:
        print('Computing clusters')
        assert H_cluster is not None
        cl_dict = {}
        for K in Ks:
            y_pred2, centers,  variances, distances = cluster(H_cluster, K=K, seed=seed, compute_vars_dsts=True)
            cl_dict[K] = (y_pred2, centers,  variances, distances )
    else:
        raise

    colors = ['red', 'blue', 'black', 'cyan']
    all_inds = np.array(range(len(H_all)))
    handles = []
    all_sampled = {}
    
    ydict = collect_year_distribution(H_all, years_input, thresholds)

    all_data_cl = {}
    if extra_filter is not None:
        print('Applying extra filter, samples filtered:', extra_filter.sum())

    for l,K in enumerate(Ks):
        y_pred2, centers,  variances, distances  = cl_dict[K]
        cl_centers = {k:v for k,v in enumerate(centers)}
        all_samples = {thresh:[] for thresh in thresholds}
        for j in range(n_sims):
            data_hist_evolution = {}
            for ii,thresh in enumerate(thresholds):

                subset =  H_all.sum(1)>thresh
                if extra_filter is not None:
                    subset = np.logical_and(subset, extra_filter==False)
                
                H_filt = H_all[subset]
                M_filt = M_all[subset]
                years_filt, books_filt = years_input[subset], books[subset]
                y_pred_filt = y_pred2[subset]
                orig_inds = all_inds[subset]

                data, _, sampled_ = get_data_gausswindow(years_filt, H_filt, M_filt, orig_inds, books_filt, 
                                                         t_sigma= t_sigma, n_samples=n_samples, plot=False)
                hilf =  get_evolve_historic_data_threshed(M_filt, data, years_filt, y_pred_filt, thresh, K, plot=False)

                data_hist_evolution.update(hilf)
                all_samples[thresh].append(sampled_)

            # collecting data        
            if j ==0:
                all_data = {k:[None,[], [], [], []] for k in data_hist_evolution.keys()}

            for k in all_data.keys():
                hilf = data_hist_evolution[k]
                times2 = [t[0] for t in hilf[0][:-1]]
                entropy2 = hilf[2]
                counts = hilf[5]
                clusters_t = hilf[3]
                pages_t = hilf[4]

                all_data[k][0]=times2
                all_data[k][1].append(entropy2)
                all_data[k][2].append(counts)

                if len(all_data[k][3])==0:
                    all_data[k][3] = clusters_t
                else:
                    for i_, cls_t_ in enumerate(clusters_t):
                        all_data[k][3][i_].extend(cls_t_)
                        
                if  len(all_data[k][4])==0: #
                    all_data[k][4] = pages_t
                    
                else:
                    for i_, p_t_ in enumerate(pages_t):
                        try:
                            all_data[k][4][i_].extend(p_t_)
                        except:
                            import pdb;pdb.set_trace()

        all_sampled[K] = all_samples
        all_data_cl[K] = all_data
        
    plt.legend(handles, [str(k_) for k_ in Ks] )
    plt.show()
    
    return all_data_cl
