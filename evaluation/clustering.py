import numpy as np
import os
from sklearn.cluster import KMeans
import random
import pandas as pd
from table_utils import *


def get_cluster_by_ref_page(ref_page, clusters): 
    # Get the cluster a reference page is contained in
    
    contained_in = []
    for k,cluster_pages in clusters.items():
        if isinstance(k, int):
            if ref_page in cluster_pages:
                contained_in.append(k)
    if len(contained_in)==0:
        print('Reference page not found')
    return contained_in


def get_gt_clusters(pages,  y_pred, all_pages):
    # Compute the cluster dictionary

    n_clusters = len(list(set(y_pred)))
    clusters = {k:[] for k in range(n_clusters)}
    _ = {clusters[k].append(v[0]) for k,v in zip(y_pred, all_pages)}
        
    gt_clusters = {}
    for j,(a,b) in enumerate(pages):

        if '2266' not in a:
            splits = a.split('_')
            book, page = '_'.join(splits[:-1]), splits[-1].replace('p','')
            clust = get_cluster_by_ref_page(a, clusters)

            assert len(clust)==1
            clust = clust[0]
            if clust in gt_clusters:
                gt_clusters[clust].append((a,b))
            else:
                gt_clusters[clust] = [(a,b)]
    return gt_clusters


def cluster(X, K, seed, compute_vars_dsts = False):
    # Helper to cluster data X into K clusters
    
    kmean = KMeans(n_clusters=K, random_state=seed)
    y_pred = kmean.fit_predict(X)
    
    if compute_vars_dsts:
        C = kmean.cluster_centers_
        variances = get_variances(X, C, y_pred)
        distances = get_single_distances(X, C, y_pred)
    
        return y_pred, C, variances, distances, 
    else:
        return y_pred, C


def compute_within_cluster_variance(X, c):
    var = np.sum((X-c)**2)/X.shape[0]
    return var


def get_variances(X, C, y_pred):
    variances = []
    for i,c in enumerate(C):
        vari = compute_within_cluster_variance(X[y_pred==i], c)
        variances.append((i,vari))
    return variances


def get_single_distances(X, C, y_pred):
    distances = []
    for i,x in enumerate(X):
        c = C[y_pred[i]]
        dist = np.linalg.norm(x-c)
        distances.append(dist)
    return np.array(distances)

