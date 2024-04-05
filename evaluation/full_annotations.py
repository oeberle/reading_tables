import collections
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
import pickle
import scipy
import scipy.stats
from scipy.ndimage.measurements import center_of_mass, label
from scipy import ndimage
import matplotlib.pyplot as plt
from dataloaders.annotated_sphaera import annotated_sphaera_tables
from utils import set_up_dir
import os
import sklearn.metrics
from table_utils import *
import pandas as pd
import itertools

class PeakDigitsEachChannel(object):
    """
    Detect peaks from digit activation maps.
    Arguments:
        scale_isolated (float): Parameter to normalize activity of isolated single digits.
        max_d (int): Maximum distance to detect activity clusters.
        peak_cutoff_rel (float): Defines cutoff value relative to the maximum.
        clip_cutoff (float): Defines the minimal cutoff value.
    """

    def __init__(self,
                 scale_isolated=6,
                 max_d = 15,
                 peak_cutoff_rel = None,
                 verbose=False,
                 clip_cutoff=0,
                ):
        
        self.scale_isolated = scale_isolated
        self.max_d = max_d  
        self.peak_cutoff_rel = peak_cutoff_rel
        self.clip_cutoff = clip_cutoff
        
    def label_peaks_channel(self, x):

        labeled_peaks, num_peaks = ndimage.label(x) 

        # get centers and pooled activations for these peaks
        centers = np.array(ndimage.center_of_mass(labeled_peaks, labels=labeled_peaks, index=range(1,num_peaks+1))
                        ).astype(int)
        act = np.array(ndimage.sum(x, labels=labeled_peaks, index=range(1,num_peaks+1)))
        return labeled_peaks, num_peaks, centers, act

    def proc(self, input_data):
        
        if isinstance(input_data, str):
            with open(input_data, 'rb') as f:
                data = pickle.load(f) 
        elif isinstance(input_data, dict):
            data = input_data
        else:
            raise
        
        img = data['X_scale'][0] #image
        b_data = data['H_all_comps']
                
        assert img.shape == b_data.shape[-2:]

        if data['rot_chosen'] != [0]:
            # undo rotation
            assert len(b_data.shape)==3 and len(img.shape)==2
            rot_chosen = data['rot_chosen'][0]
            # rotate back to standard (human-readable) orientation 
            img = np.rot90(img, rot_chosen, [0,1])
            b_data = np.rot90(b_data, rot_chosen, [1,2])
            
        b_data[:10] = b_data[:10]/self.scale_isolated

        # flexible cutoff
        self.max = b_data.max()
        cutoff = int(self.peak_cutoff_rel*self.max) 
        
        # Limit minimal values of cutoff if there is little activity of map (max peak is small)
        cutoff = max([cutoff, self.clip_cutoff])        
        b_data =  np.clip(b_data-cutoff, a_min=0, a_max=None) #ReLu, bigram maps 
        
        Cs = []
        A = []
        max_patch_size = []
        for i in range(110):

            labeled_peaks, num_peaks, centers, act  = self.label_peaks_channel(b_data[i])
            # digit_idx, center_y, center_x, act
            csa = [(i, d[0], d[1], a) for d, a in zip(centers, act)]
            A.append(labeled_peaks)
            Cs.extend(csa)
            
            if len(csa)>0:
                max_patch_size.append(int(np.sqrt(sorted(np.bincount(labeled_peaks.flatten()))[-2])))
                        
        if len(Cs)==0:
            print('No bigrams detected')
            bidist, bigrams_final =  [], []
            return bidist, bigrams_final,  b_data, img 
        
        elif len(Cs)==1:            
            Cs_ =  list(map(lambda x: [int(x[0]),x[1], x[2], x[3]], Cs))
            bidist =  [cs[0] for cs in Cs_]
            bigrams_final =  [cs[:3] for cs in Cs_]
            return bidist, bigrams_final,  b_data, img 
        else:
            if self.max_d is None:
                patch_size = max(max_patch_size)
                print('Using computing patch_size', patch_size)
            else: 
                patch_size = None

        X_clust = [[cs[1], cs[2]] for cs in Cs]
        Act = [cs[3] for cs in Cs]
        Digits = [int(cs[0]) for cs in Cs]

        Z = linkage(X_clust,
                    method='complete',  
                    metric='euclidean') 

        # retrieve clusters with `max_d`
        if self.max_d is None:
            assert patch_size is not None
            self.max_d = round(patch_size*2) 
        else:
            assert patch_size is None

        print('use max_d for clustering',  self.max_d)
        clusters_out = fcluster(Z, self.max_d, criterion='distance')

        final_inds = [int(np.argwhere(clusters_out==i)[np.argmax(np.array(Act)[np.argwhere(clusters_out==i)])]) for i in  np.unique(clusters_out)]

        bigrams_final = list(np.array(Cs)[final_inds]) #, np.array(Digits)[final_inds]))
        bigrams_final =  list(map(lambda x: [int(x[0]),x[1], x[2], x[3]], bigrams_final))
        
        bidist = [x[0] for x in bigrams_final]
        bigrams_final = [b_[:3] for b_ in bigrams_final]
        
        return bidist, bigrams_final, b_data, img        


def get_gt_veterum_nostro(H_all, M_all, gt_csv = 'data/corpus/sun_zodiac.csv'):
    # Get ground truth labels for the different types of veterum and nostro tables

    df_gt = pd.read_csv(gt_csv, delimiter=',')
    label_map = {1:1, 2:2, 8:8,  9:9}

    df_gt = df_gt[df_gt.page != '1931_beyer_quaestiones_1573_p204']
    gt_pages, gt_labels = [x+'.jpg' for x in list(df_gt.page)], [label_map[i] for i in list(df_gt['layout_type'])]
    all_pages = [(os.path.split(d)[1], d) for d in M_all]

    H_gt = []
    gt_idx = []
    y_true = np.zeros(len(H_all))
    y_true_multi = np.zeros(len(H_all))

    not_found = 0
    for page, multilabel in zip(gt_pages,gt_labels):
        idx = [i for i,x in enumerate(all_pages) if x[0]==page]
        try:
            assert len(idx)==1
        except:
            print('****No page', page)
            not_found+=1
            continue
        idx = idx[0]
        h = H_all[idx]
        H_gt.append(h)
        y_true[idx] = 1.
        y_true_multi[idx] = multilabel 
        
    return H_gt, y_true, y_true_multi
    
def get_bigrams(digits_list):
    # Get occuring unigrams and bigrams from a list of numbers
    
    bigrams = []
    numbers = []
    unigrams = []
    for digits in digits_list:
        number_string = ''.join([d[1] for d in digits])
        numbers.append(number_string)
        number_string_new = number_string.replace('^','').replace('$','')
        if len(number_string_new)==1:
            unigrams.append((int(number_string_new),))
        else:
            bis = [(int(n1),int(n2)) for n1,n2 in list(zip(number_string_new[:-1], number_string_new[1:]))]
            bigrams.extend(bis)
            
    return bigrams, unigrams, numbers


def get_digits(x, plot=False, fax = None, fax2=None):
    # Get detected digits from the full page annotations

    bboxes, numbers = x[1], x[3]
    all_data =  list(zip(bboxes, numbers))

    data_all_center = [(get_center(x_[0]), x_[1]) for x_ in all_data]
    data_all_center = list(sorted(data_all_center, key=lambda x: x[0][1]))
    digits = []
    
    for m,start in  enumerate(data_all_center):

        bbox, number = start
        X = x[0].squeeze()

        if number.startswith('^'):
           # print(m, number)
            new_digit = []
            data = (bbox,number)
            new_digit.append(data)

            candidates, dists_sort,dat_filterd =  compute_dists(data, data_all_center)#[::-1]
            candidates1= list(zip(candidates, dists_sort))

            next_number = (None,'')
            tries = 0
            while not next_number[1].endswith('$'):
                if len(candidates)==0:
                    break
                else:
                    next_number = candidates[0]
                    dist = dists_sort[0]
                    
                if dist <=25: 
                    new_digit.append(next_number)
                else:
                    break

                data_all_center_filt= [d for d in data_all_center if d!=next_number]
                candidates, dists_sort,dat_filterd =  compute_dists(next_number, data_all_center_filt)#[::-1]
                candidates2= list(zip(candidates, dists_sort))
                
                tries+=1
                if tries>=20:
                    break
            digits.append(new_digit)        
                
        else:continue

    if plot:
        mask = np.zeros_like(X)    
        new_label=['NaN']

        c = 0
        for digit in digits:
            new_label=[]
            for x_ in digit:
                bbox, digit = x_
                x0,y0,w,h =  bbox
                mask[int(y0-h/2):int(y0+h/2), int(x0-w/2):int(x0+w/2)] = 10 + c
                new_label.append(digit)
            c+=1
            
        if fax is None:
            f,ax = plt.subplots(1,1, dpi=400)
        else:
            f,ax = fax
            
        xx = X.squeeze().detach().cpu().numpy()
        
        if fax2 is not None:
            f2,ax2 = fax2

            mask = np.zeros_like(X)
            c=0
            for m,start in enumerate(data_all_center):
                bbox, number = start
                x0,y0,w,h =  bbox
                mask[int(y0-h/2):int(y0+h/2), int(x0-w/2):int(x0+w/2)] = 1
                c+=1

            ax2.imshow((xx-xx.min())/(xx.min()-xx.max()), cmap='Greys_r')
            ax2.imshow(mask, alpha=0.5,  cmap='Reds' ) #plt.get_cmap('hsv'))
            ax2.set_title('Annotations', fontsize=8)

    return digits


def get_center(xywh):
    # Get bounding box center coordinates

    x,y,w,h = xywh
    x_center = x+w/2.
    y_center = y+h/2.
    return (x_center, y_center,w,h)


def compute_dists(data, data_all):
    # Compute distances of digit to all to other digits

    dists = [x[0][0]-data[0][0] for x in data_all]
    inds = np.argsort(dists)
    
    # all Digits to the right
    ind_pos = [i for i in inds if dists[i]>=-0.1 and not data_all[i][1].startswith('^')]
    data_all_right = [data_all[i] for i in ind_pos]
    dists_right = [dists[i] for i in ind_pos]
    dat_filterd = [data_all[i] for i in inds if i not in ind_pos]
    
    # Now select based on y
    a,b =1.0,1.0
    disty = [((a*(x[0][1]-data[0][1]))**2 + (b*(x[0][0]-data[0][0]))**2)**0.5  for x in data_all_right]

    ind_pos = np.argsort(disty)
    ind_pos_y = [i for i in ind_pos]
    dists_sort = [disty[i] for i in ind_pos_y]
    data_all_sort = [data_all_right[i] for i in ind_pos_y]

    assert len(data_all) == len(data_all_sort) +  len(dat_filterd)
    return data_all_sort, dists_sort,dat_filterd


def get_corr_data(X, X2, plot = False, cmap = 'Reds'):
    # Compute and plot feature histogram correlations

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

    if plot:
        f, axs = plt.subplots(2,1, figsize=(6,1.5))

        axs[0].imshow(X, cmap=cmap)
        cbar = axs[1].imshow(X2, cmap=cmap)
        [ax.axis('off') for ax in axs.flatten()]
        f.subplots_adjust(hspace=0.2, wspace=0.1)
        f.suptitle(r'$\rho$: {}, $\rho_c$: {}, $N_{{bigr}}$: {}'.format('{:0.2f}'.format(p_corr),'{:0.2f}'.format(p_corr_flat),
                                                                              n_bigram), y=1.0)
        axs[0].text(-6, 8, 'true', rotation=90)    
        plt.show()

        f, axs = plt.subplots(1,1)
        plt.colorbar(cbar)
        axs.axis('off')
        plt.show()
    
    data = []
    for xt, xp in zip(X, X2):
        data.append([scipy.stats.pearsonr(xt, xp)[0], np.sum(xt), np.sum(count_uni(xt)) ] )
        
    return data

def correlate_(res_dir_case, 
               eval_root_dir,
               plot_dir_case=None, 
               save = False,
               plot = False,
               ignore_inds = [],
               hist_key = 'h_pool_plus',
               return_hists=False,
               peak_params = {'scale_isolated': 6,
                             'max_d': 15,
                             'peak_cutoff_rel' :15,
                             'verbose':False}):

    # Compute, plot and compare feature histograms

    eval_loader =  annotated_sphaera_tables(root_dir=eval_root_dir, binarize=False, refsize=1200,  removemean=True)

    if plot_dir_case:
        set_up_dir(plot_dir_case)
    
    C = []
    C2 = []
    C3 = []

    all_singles = []
    all_bigrams = []
    raw_singles = []    

    print(plot_dir_case)
    pdir = os.path.join(res_dir_case, '1_{}.p')

    k=0

    res_dict = {}
    for x in eval_loader:

        if k in ignore_inds:
            k+=1
            continue

        bboxes = x[1]
        numbers = x[3]
                
        if plot:
            gs = gridspec.GridSpec(4,3, width_ratios=[1,1,1], height_ratios=[0.7,0.1,0.1,0.1]) 
            f = plt.figure(dpi=300) 
            ax1 = plt.subplot(gs[2])
            ax2 = plt.subplot(gs[1])
            
            fax1 = (f,ax1)
            fax2 = (f,ax2)
        else:
            fax1=fax2 = None
            
        digits = get_digits(x, plot=plot, fax= fax1, fax2 = fax2)
        bigrams, unigrams, numbers = get_bigrams(digits)

        raw_singles+=[int(x_.replace('^','').replace('$','') ) for x_ in x[3]] 
        all_singles+=unigrams
        all_bigrams+=bigrams

        A = pickle.load(open(pdir.format(k), 'rb'))
        X = A['H'].squeeze().sum(0)
        page = A['X_scale'][0]
        
        if A['rot_chosen'] != [0]:
            # undo rotation
            HH = A['H'].squeeze()
            print(HH.shape)
            HH_rot = np.rot90(HH, A['rot_chosen'][0], [1,2])
            page_rot = np.rot90(page, A['rot_chosen'][0], [0,1])
            X = HH_rot.sum(0)
            page = page_rot
        
        hist1 = A[hist_key].squeeze()

        bigrams_count = collections.Counter(bigrams)
        unigrams_count = collections.Counter(unigrams)
        hist_true = [bigrams_count[num] if len(num)==2 else unigrams_count[num] for num in A['numbers']]
        
        PD = PeakDigitsEachChannel(**peak_params)
        hist_out_peaks, bigrams_final, bigram_map, img = PD.proc(pdir.format(k))
        peaks_counts = collections.Counter(hist_out_peaks)
        pd_hist = [peaks_counts[j] if j in peaks_counts else 0 for j in range(110)]
        
        pool_hist = np.clip(bigram_map, a_min=0, a_max=200).sum(2).sum(1)
                
        corr = np.corrcoef(pool_hist,hist_true)[0][1]    
        corr2 = np.corrcoef(pd_hist,hist_true)[0][1]
        corr3 = np.corrcoef(pool_hist,pd_hist)[0][1]

        if plot:    
            # Overwrite with peaks
            ax1.imshow(img, alpha=0.35, cmap='gray')
            for item in bigrams_final:
                 ax1.text(item[2], item[1], nr_from_index(item[0]), fontdict=None, fontsize=2)
        
            ax0 = plt.subplot(gs[0])

            percentile=100. #99.90
            abs_max = np.percentile(X, percentile)
            if abs_max <= 0.:
                percentile=99.99
                abs_max = np.percentile(X, percentile)

            print(bigram_map.shape, img.shape,  A['rot_chosen'])
            hh1 = ax0.imshow(bigram_map.sum(0), vmin=0., vmax=abs_max) #, cmap='Greys', vmin=0, vmax=1.)
            ax0.imshow(img, alpha=0.25, cmap='Greys_r') #, cmap='Greys', vmin=0, vmax=1.)

            #hists
            ax0 = plt.subplot(gs[3:6])
            ax0.hist(range(len(pool_hist)), weights = pool_hist, bins=len(hist_true))
            ax0.set_title('bigram histograms', fontsize=8)
            
            #hists
            ax0 = plt.subplot(gs[6:9])
            ax0.hist(range(len(pd_hist)), weights = pd_hist, bins=len(hist_true))
            ax0.set_title('peak detected histograms', fontsize=8)

            #hists
            ax0 = plt.subplot(gs[9:12])
            ax0.hist(range(len(hist_true)), weights = hist_true, bins=len(hist_true))
            ax0.set_title('GT histograms', fontsize=8)

            plt.suptitle('corr: pool {:0.3f} /  pd {:0.3f} / pool-pd {:0.3f}'.format(corr, corr2, corr3) , y=1.0, fontsize=9)
            f.tight_layout()

            if save:
                f.savefig(os.path.join(plot_dir_case, '{}.png'.format(k)), dpi=590)
            plt.show()

        C.append(corr)
        C2.append(corr2)
        C3.append(corr3)

        res_dict[k] = (hist_true, pool_hist, pd_hist)        
        k+=1
    
    if return_hists:
        return C, C2, C3, res_dict
    else:
        return C, C2, C3


def single_digits(h, digits):
    # Compute single digit histogram count
    
    h_single = {i:0 for i in range(10)}
    
    for i,h in enumerate(h):
        digit = digits[i]
        if len(digit)==2:
            h_single[digit[0]]+=h
            h_single[digit[1]]+=h

        elif len(digit)==1:
            h_single[digit[0]]+=h
            
    return np.array([h_single[i] for i in range(10)])
    
    
def eval_table_embeddings(H_train, y_train, H_test, y_test, case = 'min'):
    # How well can a smililartiy model predict y_test from H_test using training data (H_train, y_test)

    y_pred = []
    y_true_ = []
    
    for xte, yte in zip(H_test, y_test):

        D_true = sklearn.metrics.pairwise.euclidean_distances(xte[np.newaxis,:], H_train[y_train==1]).squeeze()
        D_false = sklearn.metrics.pairwise.euclidean_distances(xte[np.newaxis,:], H_train[y_train==0]).squeeze()
            
        # Test if test points are correct
        if case=='min':
            # for test point to be correctly classified its distance to the nearest train sample must be smaller than to any other
            if np.min(D_true) <= np.min(D_false):
                y_pred.append(1)
            else:
                y_pred.append(0)
        elif case == 'mean':
            kn = 5
            if np.mean(sorted(D_true)[:kn]) <= np.mean(sorted(D_false)[:kn]): 
                y_pred.append(1)
            else:
                y_pred.append(0)

        y_true_.append(yte)

    # fpr: TP/(TP+FN)
    # tpr: FP/(TN+FP)

    TP = len([1 for  yp, yt in zip(y_pred, y_true_ ) if (yp==yt) and (yt==1.)])
    FN = len([1 for  yp, yt in zip(y_pred, y_true_ ) if (yp!=yt) and (yt==1.)])

    TN = len([1 for  yp, yt in zip(y_pred, y_true_ ) if (yp==yt) and (yt==0.)])
    FP = len([1 for  yp, yt in zip(y_pred, y_true_ ) if (yp!=yt) and (yt==0.)])

    # Sensitivity, hit rate, recall, or true positive rate
    tpr = TP/(TP+FN) 
    
    # Fall out or false positive rate
    fpr = FP/(TN+FP) 
    
    # False negative rate
    fnr = FN/(TP+FN)
    
    # False discovery rate
    fdr = FP/(TP+FP)

    return (y_pred, y_true_, TP, FN, TN, FP, tpr, fpr, fnr , fdr)




    