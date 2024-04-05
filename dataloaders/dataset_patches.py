import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PolyCollection
import os
from PIL import Image
import abc
from skimage import io
from torch import tensor
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import random
import torchvision


class HistoricNumbersDataset(Dataset):
    """
    Args:
        data_directory (string): Path of the dataset, should contain raw/processed folders.
        csv_file (string): Path to the csv file with annotations.
        train (bool): If train/test splts should be considered.
        return_fixation (bool): if true returns each digit location  (x,y).
        return_single (bool): if true returns each digit as a single target, else returns a list of digits and boxes.
        binarize (bool): if True the images will be binarized
        test_split (float): specify test split.
        page_transform (callable, optional): Optional transform to be applied on a sample.
        local_data (bool): Set to true if local images are to be processed (instead of a csv file specified 
            in csv_content, see get_img_path() function).
        seed (int): Random seed.
        
    """
    def __init__(self, 
        data_directory,
        csv_file="numbers.csv", 
        train=True, 
        return_fixation=None,
        return_single=False, 
        binarize=True, 
        delimiter = ',', 
        test_split=0.1,
        page_transform=None, 
        local_data = False,
        seed=1 ):
        
        super(HistoricNumbersDataset, self).__init__()
        self.csv_content = pd.read_csv(csv_file, delimiter=delimiter)
        self.train = train
        self.return_fixation =return_fixation
        self.return_single = return_single
        self.binarize = binarize
        self.data_directory = data_directory
        self.page_transform = page_transform
        self.test_split = test_split
        self.local_data = local_data
        self.seed = seed
        self.annotations_frame = self.get_dataset()
        

    def get_dataset(self):

        # Split each bounding box into the digits
        dataset = []
        for idx, row in self.csv_content.iterrows():
            if not np.isnan(row.number):
                adversarial = False
                try:
                    row.number = int(row.number)
                    nof_digits = len(str(row.number))
                except AttributeError:
                    nof_digits = 1
                    row.number = 0
                    print('AttributeError')
            else:
                nof_digits = 0
                row.number = np.nan
                adversarial = True

                
            if 'xywh'  in self.csv_content:
                bbox = [int(float(x)) for x in row.xywh.split(",")]

                if nof_digits == 1:
                    bbox_digit_width = int(bbox[2]/nof_digits)
                        
                else:
                    bbox_digit_width = None

            else:
                bbox = []
                

            data = {
                'num_page_id': row.num_page_id,
                'num_patch_id': row.num_patch_id,
                }
            
            if 'element_label' in self.csv_content:
                data['element_label'] = str(row.element_label)
            else:
              #  import pdb;pdb.set_trace()
                if np.isnan(row.number):
                    data['element_label'] =  ''
                else:
                    data['element_label'] =  str(row.number)

           # import pdb;pdb.set_trace()
            if 'book' in self.csv_content:
                try:
                    bookstr = row.book
                except:
                    bookstr = ''
            else:
                bookstr = ''
             
                
            data['bookstr'] = bookstr
                
            if 'star_attr' in self.csv_content:
                star_attr = int(row.star_attr)
            else:
                star_attr = 0
            
            data['star_attr'] = star_attr # [bookstr, star_attr]
            data['page_id'] = row.page_id

                
            #import pdb;pdb.set_trace()
            if not adversarial: 
                if self.return_single: # Create one entry for every digit
                    for i in range(nof_digits):
                        data['number'] = [int(str(row.number)[i])],
                        
                        if bbox_digit_width is None:
                            ratio = bbox[2]/bbox[3] #width/height

                            bbox_digit_width = int(bbox[2]/nof_digits)
                            assert bbox_digit_width >0
                            data['xywh'] = [[bbox[0]+bbox_digit_width*i,bbox[1], bbox_digit_width, bbox[3]]]

                        else:     
                            data['xywh'] = [[bbox[0]+bbox_digit_width*i,bbox[1], bbox_digit_width, bbox[3]]]
                            assert (np.array(data['xywh'])>0).all() ==True

                            
                        data['num_digit_id'] = i
                        

                        # Check for star_attr
                        if data['star_attr'] == 0:
                            dataset.append(data.copy()) # inser a copy, otherwise in the next loop the previous dict will be changed!!!
                        else:
                            print('Skipping horizontal', data['element_label'])


                else: # Create one entry for the whole number
                    print('Only support single digits for patch training')
                    data['xywh'] = [bbox]
                    assert (np.array(data['xywh'])>0).all() ==True
                    data['num_digit_id'] = 0
                    data['number'] = int(str(row.number))
                    dataset.append(data.copy())
                   
                    
            else: # adversarial
                data['number'] = []
                data['element_label'] = ''
                data['xywh']=  [bbox]
                data['num_digit_id'] = 0
                dataset.append(data)


        dataframe = pd.DataFrame(dataset)
        
        dataframe['id'] = list(range(len(dataframe)))
        index_columns = ['id']
        dataframe = dataframe.set_index(index_columns,verify_integrity=True,drop=False)
        
        # Split in train and test set by dividing the annotations frame and overwriting it.
        if self.test_split > 0.:

            np.random.seed(self.seed)

            train_inds = np.random.choice(
                np.arange(len(dataframe)), 
                size=int(len(dataframe)*(1.-self.test_split)), 
                replace=False)
            if self.train:
                dataframe = dataframe.iloc[train_inds]
            else:
                # if not training, then test indices are all remaining inds
                test_inds = set(np.arange(len(dataframe))) - set(train_inds)
                test_inds = np.array(list(test_inds))
                dataframe = dataframe.iloc[test_inds]
                
        return dataframe


    def __len__(self):
        return len(self.annotations_frame)


    @staticmethod
    def rescale_bbox(bbox, bbox_factors):
        return [int(bbox_factors[j]*x) for j,x in enumerate(bbox)]


    @property
    def raw_folder(self):
        return os.path.join(self.data_directory, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.data_directory, 'processed')
    
    def get_img_path(self, page_id, case='processed'):
        """
        Args:
            page_id (int): unique id of page
        """
        
        if self.local_data:
            splits = page_id.split('_')
            book = '_'.join(splits[:-1])
            page = int(splits[-1].replace('.jpg','').replace('p',''))
            file_name = book + '_p'+str(page-1)+".jpg"
            file_path = os.path.join(self.data_directory, book, case , str(page-1)+".jpg")
        else:
            folder= self.raw_folder if case=='raw' else self.processed_folder
            file_name = str(page_id)+".jpg"  
            file_path = os.path.join(folder, file_name)

        return file_path

    def load_image(self, path, gray=True):
        """
        Convenience method to load a page as PIL image. 
        
        Page is converted to three channel RGB image.
        """
        page = io.imread(path)
        page = Image.fromarray(page).convert('RGB') # some pages are scanned in greyscale, i.e. single channel, therefore we convert all images to RGB
        if gray:
            page = transforms.Grayscale(num_output_channels=3)(page)
        return page


    def show_page(self, page_id, ax=None):
        if not ax:
            fig, ax = plt.subplots(1,1,figsize=(10,10))
        ax.imshow(self.load_image(self.get_img_path(page_id)))

    @staticmethod
    def add_bboxes(flattend_boxes, ax, cut=None):
        X = []
        Y = []
        for xywh in flattend_boxes:
            X.append([xywh[0],xywh[0],xywh[0]+xywh[2],xywh[0]+xywh[2]])
            Y.append([xywh[1],xywh[1]+xywh[3],xywh[1]+xywh[3],xywh[1]])
            
            if cut:
                assert len(flattend_boxes)==1
                ax.set_xlim([xywh[0]-cut[0], xywh[0]+xywh[2] + cut[0]])
                ax.set_ylim([xywh[1]+xywh[3] + cut[1], xywh[1]-cut[1]])
            
        XY = np.array([X,Y])
        XY = np.swapaxes(XY, 0,2)
        XY = np.swapaxes(XY, 0,1)
        coll = PolyCollection(XY, facecolors=(0.9, 0., 0., 0.0), edgecolors=(0.9, 0., 0., 0.7))
        ax.add_collection(coll)
        return XY

    def show_page_with_single_box(self, page_id, ax=None, cut=None,  case='processed'):
        """
        Args:
            page_id (int): unique id of this page
        """
        if not ax:
            fig, ax = plt.subplots(1,1,figsize=(2,2))
                        
        if isinstance(page_id, int):
            try:
                book_id = self.annotations_frame.loc[page_id].page_id.replace('.jpg','').strip()
            except:
                import pdb;pdb.set_trace()
        else:
            
            book_id = page_id

            
        path_ = self.get_img_path(book_id, case=case)
        ax.imshow(self.load_image(path_, gray=False))
        page_data = self.annotations_frame.loc[page_id]
        flattend_boxes = page_data.xywh
        
        XY = self.add_bboxes(flattend_boxes, ax=ax, cut=cut)
        ax.set_title( [page_data.number,page_data.element_label] )

        return XY, page_data
        
        
    def show_page_with_all_boxes(self, page_id, ax=None, cut=None):
        """
        Args:
            page_id (int): unique id of this page
        """
        if not ax:
            fig, ax = plt.subplots(1,1,figsize=(20,20))

            
        assert isinstance(page_id, str)
        
        book_id = page_id.replace('.jpg','').strip()
        page_data = self.annotations_frame[self.annotations_frame.page_id==page_id]
        
        ax.imshow(self.load_image(self.get_img_path(book_id, case='raw')))

        #flattend_boxes = [item for sublist in page_data.xywh.values for item in sublist]
        flattend_boxes = [x[0] for x in list(page_data.xywh)]
      
        XY = self.add_bboxes(flattend_boxes, ax=ax)
        
        if cut:
            ax.set_xlim([xywh[0]-cut[0], xywh[0]+xywh[2] + cut[0]])
            ax.set_ylim([xywh[1]+xywh[3] + cut[1], xywh[1]-cut[1]])
        
        return XY, page_data
    
    
        
    def get_full_page(self, idx):
        """
        Returns:
            page: PIL image (if no further transforms specified), showing the whole page of the book
            region: PIL image (if no further transforms specified), showing the cropped region of the number
            bbox: 1d tensor containing the bounding box as [x, y, w, h]
            number: integer, target number displayed in the bbox
        """
        factor_h, factor_w = 1., 1.

        
        book_id = self.annotations_frame.iloc[idx].page_id.replace('.jpg','')
        page = self.load_image(self.get_img_path(book_id, case='processed'))

        if self.page_transform:
            h0, w0, _ = np.shape(page)
            page = self.page_transform(page)
            _, h1, w1 = np.shape(page)
            
            factor_h = float(h1)/h0
            factor_w = float(w1)/w0

        bbox_factors = [factor_w, factor_h, factor_w, factor_h]

        bbox = self.annotations_frame.iloc[idx].xywh

        bbox = [self.rescale_bbox(box, bbox_factors) for box in bbox] 
        bbox_before = bbox
        
        if self.return_fixation:
            bbox = [[box[0] + np.ceil(box[2]/2.) , box[1] + np.ceil(box[3]/2.)] for box in bbox]
            
        assert (np.array(bbox)>0).all() == True
            
        bbox = tensor(bbox)
        number = tensor(self.annotations_frame.iloc[idx].number)
        
        if 'element_label' in self.annotations_frame:
            element_label =  self.annotations_frame.iloc[idx].element_label
        else:
            element_label = ''
        
        
        return page, bbox, number, element_label
        
                
        
class HistoricFeaturesDataset(HistoricNumbersDataset):
    """
    Args:
        pad (int): How many pixels to add outside of each patch.
        scale (float): Fraction to define if the mask should be scaled (1. is no scaling)
        beta (float): Defines how strongly to consider activations outside the ground truth mask.
        patch_transform (callable, optional): Optional transform to be applied on the patch.
        mask_transform (callable, optional): Optional transform to be applied on the mask patch.
        extra_rotations (bool): Adds additional samples by augmenting rotations. 
    """

    def __init__(self, 
        *args, 
        pad = 0,
        scale = 1.,
        beta=0.5,
        patch_transform = None,
        mask_transform = None,
        extra_rotations = True, 
        **kwargs):

        self.pad = pad
        self.scale = scale
        self.beta = beta
        self.patch_transform = patch_transform
        self.mask_transform = mask_transform
        self.extra_rotations = extra_rotations
        super(HistoricFeaturesDataset, self).__init__(*args, **kwargs)

        
    def __getitem__(self, idx):
        # Call HistoricPatchesDataset getitem to get page, bbox, number, element_label

        page, bbox, number, element_label = self.get_full_page(idx)
        
        try:
            patch, Y, mask = HistoricFeaturesDataset.get_historic_map(page, number, bbox, extract_patch=True, pad=self.pad, scale=self.scale, beta=self.beta)

            if self.patch_transform:
                patch = self.patch_transform[0](patch)
                try:

                    Y =  torch.stack( [self.patch_transform[1](y) for y in Y],0)
                except:
                    print( bbox, number, element_label)
                    for y in Y:
                        print(y.shape)

                mask = self.mask_transform(mask)
                if(mask.min() < 0): print('WARNING: Mask has been scaled and now contains values below 0. May needs renormalisation.')

            if number.numel() == 0:
                number = ''
            else:
                number = ''.join([str(x.item()) for x in number])

            number = (number, element_label)
            
            if  self.extra_rotations:
                rot = int(np.random.choice([-1,0,1]))
                if rot !=0:
                    patch =  torch.rot90(patch, rot, [1,2])
                    mask= torch.rot90(mask, rot, [1,2])
                    Y =  torch.rot90(Y, rot, [1,2])

        except IndexError:
            return None
        
        return patch, Y, mask, number

    @staticmethod
    def get_historic_map(X, y, fix, patch=False, extract_patch=False, pad = None, scale=1, beta=0.5):
        ''' Get patches from input pages, pad context and extract corresponding mask '''
        
        kernel_size = int(13*scale)
        sigma = int(2*scale)
        gaussian_filter = HistoricFeaturesDataset.get_gaussian_kernel(kernel_size,sigma,1)

        h0,w0 = X.shape[1], X.shape[2]

        if extract_patch:
            if len(X.shape) ==3:
                X = X.unsqueeze(0)
            bbox = fix
            scale_pad = 1.1 # add some extra padding for affine transformations
            Xsh = X.shape
            Xfull = X.clone()
            X = X[:,:,bbox[0][1]-pad:bbox[0][1]+bbox[0][3]+pad, bbox[0][0]-pad:bbox[-1][0]+bbox[-1][2]+pad].squeeze(0)

            if len(X.squeeze().shape) <2:
                raise

        h,w = X.shape[1], X.shape[2]

        if y.numel() == 0:
            y = []

        labels = [int(y_.cpu().detach().numpy()) for y_ in y]
        grid_dict = {i:torch.zeros((1,h,w)) for i in range(10)}
            
        k = 0
        if fix.shape[1] > 1:
            fix = fix

        
        for fi in fix:
            fi = fi.squeeze()


            if labels:
                tensor = grid_dict[labels[k]]
            else: # if we don't have a label, i.e. this is an adversarial sample
                tensor =  torch.zeros((1,h,w))
            if len(tensor.shape) ==3:
                tensor = tensor      
            else:
                raise
                
            if patch and extract_patch==False:
                tensor[:,int(fi[1]):int(fi[1])+int(fi[3]), int(fi[0]):int(fi[0])+int(fi[2])] = 1.
                
            elif extract_patch==True:
                
                fix_ = [np.floor(k*fi[2] + fi[2]/2.)+ pad , np.floor(fi[3]/2.) + pad] 
                tensor[:,int(fix_[1]), int(fix_[0])] = 1.
                mask = torch.zeros_like(tensor)
                mask[:,int(pad):int(h-pad), int(pad):int(w-pad)] = 1.
                # use this to set mask at half frame, e.g. pad = 20 -> mask periphery 10 border
                # pad_center = int(pad/2.)
                # use this to set mask at qarter, e.g. pad = 20  -> mask periphery 5 border
                # use: larger context but only direct bbox gets into loss function 
                pad_center = int(pad/2.) + int(pad/4.) 
                mask_transition = torch.zeros_like(mask)
                mask_transition[:,pad_center:-pad_center,pad_center:-pad_center] = 1.
                mask_periphery = mask_transition - mask

                # Final mask
                mask = mask + beta*mask_periphery
            
            else:
                tensor[:,int(fi[1]), int(fi[0])] = 1.

            if labels: # if this is not an adversarial sample
                grid_dict[labels[k]] = tensor
            
            k+=1
            
        grid_dict = {k:gaussian_filter(v.unsqueeze(0)).squeeze(0) for k,v in grid_dict.items()}    

        grid_dict = {k:1000.*v/torch.max(v) if torch.max(v)>0. else v for k,v in grid_dict.items()}

        out = torch.stack([grid_dict[i] for i in range(10)]).squeeze()
        
        if extract_patch:
            return X, out, mask
        else:
            return out       

    @staticmethod        
    def get_gaussian_kernel(kernel_size, sigma, channels=3):
        import math
        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        mean = (kernel_size - 1)/2.
        variance = sigma**2.

        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        gaussian_kernel = (1./(2.*math.pi*variance)) *\
                          torch.exp(
                              -torch.sum((xy_grid - mean)**2., dim=-1) /\
                              (2*variance)
                          )

        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

        gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                    kernel_size=kernel_size, groups=channels, bias=False, padding= int(kernel_size/2))

        gaussian_filter.weight.data = gaussian_kernel
        gaussian_filter.weight.requires_grad = False
        
        return gaussian_filter
    
    
    
    
    