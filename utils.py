import numpy as np
import itertools,time
import sys, os
from torch.utils.data.dataset import Dataset
import torch
import torch.nn as nn
from torch.autograd import Variable
from collections import OrderedDict
import pickle
from models.e2cnn_models import DigitsModel


def set_up_dir(path):
    # Set up a path directory
    
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise
            
def load_model(jobdir, weightfile, cuda=False):
    # Load model and initialize weights for inference

    jobdict = pickle.load(open(jobdir, 'rb'))
    modelcase = jobdict[6]
    model_spec = jobdict[7]
    model = get_model(modelcase, model_spec)
    model.load_state_dict(torch.load(weightfile))
    if cuda==True:
        model.cuda()
    model.eval()
    return model

    
def get_model(modelcase, model_spec, train_specs={}):
    # Helper function to get correct model class

    if len(train_specs) >0:
        print('Update model specs', train_specs)
        model_spec.update(train_specs)
    
    if modelcase == 'rotinvconv':
        model = DigitsModel(**model_spec)
    else:
        raise
    return model

