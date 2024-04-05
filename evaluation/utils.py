import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import scipy
import itertools
from matplotlib import gridspec


def compute_in_out(mask,H):
    # Compute fraction of activity inside vs outside bounding boy
    
    mask = mask[np.newaxis,:,:] 
    H = np.maximum(0., H)
    mask_out = 1. - mask
    
    assert mask.sum()+mask_out.sum() == np.prod(mask.shape)
    inside = (H*mask).sum()
    outside = (H*mask_out).sum()
    joint = H.sum()
    return inside/joint, outside/joint
    
    
def evaluate_single_digit(model, dataloader, removemean=True):
    # Evaluate the model's detection accuracy of single digit patches
    
    numbers = [(i,) for i in range(10)]
    shift = [0]
    method = 'min'
        
    Out = []
    for x in dataloader:

        page, bbox, number, element_label = x
        mask_dict = get_masks(page, element_label, bbox)
        ndict = get_number_masks(mask_dict) #contains list of [(mask, xyhw, label) ,..] for each number [0..9]

        mask_all = np.minimum(1.,np.stack([v[0] for k,v in mask_dict.items()]).sum(0))

        page = torch.autograd.Variable(page.unsqueeze(0), requires_grad=False)
        H = model(page.cuda())
        y = H.detach().cpu().numpy().squeeze()
        in_out = compute_in_out(mask_all, y)
        X = page.detach().cpu().numpy().squeeze()
        del  H
        torch.cuda.empty_cache()
        out  = evaluate_bbox_accuracy(y, ndict)
        out = list(out)+ [X] + [in_out] + [mask_all]
        
        wrong_patches = get_misclassified_patches(out, page)
        out = out + [wrong_patches]
        
        Out.append(out)

    return Out

def get_masks(page, labels, fix):
    # Compute dictionary of ground truth digit masks

    mask_dict = {}
    i = 0
    for label, fi in zip(labels,fix):
        tensor =  np.zeros(np.shape(page))
        tensor[:,int(fi[1]):int(fi[1])+int(fi[3]), int(fi[0]):int(fi[0])+int(fi[2])] = 1.
        mask_dict[i] = (tensor.squeeze(), tuple(fi), label)
        i+=1
    return mask_dict

def get_misclassified_patches(eval_data,x):
    # Retrieve misclassified patches 

    masks_ = [x_.reshape(x[0].squeeze().shape) for x_ in eval_data[3]]
    y1  = eval_data[1]
    y2  = eval_data[2]

    patches_wrong  = []
    for m_, y1_, y2_ in zip(masks_, y1, y2):

        if y1_!=y2_:

            fac=.007
            x0=np.argmax(m_.sum(0))
            x1 = len(m_.sum(0)) - np.argmax(m_.sum(0)[::-1])
            x0= int((1-fac)*x0)
            x1 = int((1+fac)*x1)

            h0=np.argmax(m_.sum(1))
            h1 = len(m_.sum(1)) - np.argmax(m_.sum(1)[::-1])
            h0=int((1-fac)*h0)
            h1 = int((1+fac)*h1)

            plot_data = (x[0].squeeze()[h0:h1,x0:x1], y2_,y1_) # patch, ytrue, yfalse
            patches_wrong.append(plot_data)
     
    return patches_wrong


def get_number_masks(mask_dict):
    # Helper function to create ground truth digit masks
    
    number_dict = {i:[] for i in range(10)}
    for k,v in mask_dict.items():
        number = int(v[2].replace('^','').replace('$',''))
        number_dict[number].append(v)
    return number_dict
    
def evaluate_bbox_accuracy(y, number_dict):
    # Evaluate if maximum activity is assigned to the correct digit map
    
    numbers = sorted(number_dict.keys())
    Y_true = []
    Y_pred = []
    Masks = []
    
    for n in numbers:
        
        # Skip non-occuring ground truth
        if len(number_dict[n])==0:
            continue
        
        masks = np.array([x[0].flatten() for x  in number_dict[n]])
        yflat = np.array([y_.flatten() for y_ in y])
        Y_pred += np.argmax(np.dot(yflat, masks.T), axis=0).tolist()
        Y_true += [n]*len(masks)
        Masks.extend(list(masks))
       
    Y_pred, Y_true = np.array(Y_pred), np.array(Y_true)
    acc = np.sum(Y_pred==Y_true)/len(Y_pred)
    return acc, Y_pred, Y_true, Masks, y