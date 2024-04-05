import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from PIL import Image
import abc
from skimage import io
from torch import tensor
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torchvision
import os
        
class PageDataset(Dataset):
    """
    Dataset loader to process book pages. 
    Args:
        table_source (string): Path to the dataset csv file.
        root_dir (string): Directory containing the images (raw/processed).
        transform (callable, optional): Optional transform to be applied on a sample.
    """
    
    def __init__(self, table_source, root_dir, transform=None):        
        super(PageDataset, self).__init__()

        self.table_source = table_source
        if isinstance(table_source, list):
            self.tables = table_source
            self.loadcase = 'individual'
        
        elif table_source.endswith('csv'):
            df = pd.read_csv(table_source)
            self.tables = list(df.page_id)
            self.loadcase = 'book'

        else:
            print('No table_source provided.')
            raise
        
        self.root_dir = root_dir
        self.transform = transform
        
    
    def load_image(self, path):
        """
        Convenience method to load a page as PIL image. 
        
        Page is converted to three channel RGB image.
        """
        page = io.imread(path)
        page = Image.fromarray(page).convert('RGB') 
        page = transforms.Grayscale(num_output_channels=3)(page)
        return page

        
    def get_image(self,page_id):
        return os.path.join(self.root_dir, page_id)
        
    def __len__(self):
        return len(self.tables)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        if self.loadcase =='book':
            table = self.tables[idx]
            img_name = self.get_image(table)
        elif self.loadcase == 'individual':
            img_name = self.tables[idx]
            
        page = self.load_image(img_name)

        if self.transform:
            page = self.transform(page)

        return page, img_name
    
    
    
class FullyAnnotatedPages(PageDataset):
    """
    Dataset loader to load and  process book pages alongside their annotations. 
    Args:
        return_bookstr (bool, optional): Returns the page location if set to True.
        delimiter (string, optional): Sets the delimiter string to load the csv.
    """
    
    def __init__(self, 
        *args,
        return_bookstr = False,
        delimiter = ',', 
        **kwargs
        ):
        
        super(FullyAnnotatedPages, self).__init__(*args, **kwargs)

        self.csv_content = pd.read_csv(self.table_source, delimiter=delimiter)
        self.annotations_frame = self.get_dataset()
        self.page_ids = sorted(list(set(list(self.annotations_frame.num_page_id))))
        self.return_bookstr = return_bookstr
        
        
    @staticmethod
    def rescale_bbox(bbox, bbox_factors):
        return [int(bbox_factors[j]*x) for j,x in enumerate(bbox)]

        
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
                raise

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
                data['element_label'] = row.element_label
            else:
                data['element_label'] =  str(row.number)

                
            if 'page_url' in self.csv_content:
                try:
                    bookstr = '!'.join(row.page_url.split('!')[2:])
                except:
                    bookstr = ''

            data['bookstr'] = bookstr
                
            if 'star_attr' in self.csv_content:
                star_attr = int(row.star_attr)
            else:
                star_attr = 0
            
            data['star_attr'] = star_attr 
            data['number'] = [int(str(row.number)[i]) for i in range(nof_digits)]

            bboxes = []
            L =0
            for i in range(nof_digits):
                if bbox_digit_width is None:
                    ratio = bbox[2]/bbox[3] #width/height

                    if ratio >=0.7:
                        bbwidth = int(bbox[2]/nof_digits)
                        bb_ = [bbox[0]+bbwidth*i,bbox[1], bbwidth, bbox[3]]
                    else:
                        bbwidth = int(bbox[3]/nof_digits)
                        bb_ = [bbox[0],bbox[1]-bbwidth*i, bbox[2], -bbwidth]
                        L = 1
                else:     
                    bb_ = [bbox[0]+bbox_digit_width*i,bbox[1], bbox_digit_width, bbox[3]]

                bboxes.append(bb_)

            data['xywh'] = bboxes
            data['num_digit_id'] = 0

            if True:
                if data['star_attr'] == 0:
                    dataset.append(data.copy()) 
                else:
                    print('Skipping horizontal', data['element_label'])
            else: 
                print('Keeping horizontals') #only the ones with a  star (*) attr
                dataset.append(data.copy())


        dataframe = pd.DataFrame(dataset)
        index_columns = ['num_page_id','num_patch_id', 'num_digit_id']
        dataframe = dataframe.set_index(index_columns,verify_integrity=True,drop=False)
        return dataframe

        
    def get_img_path(self, page_id):
        """
        Args:
            page_id (int): unique id of page
        """
        file_name = str(page_id)+".jpg"
        file_path = os.path.join( self.root_dir , file_name)
        return file_path
        
    def __getitem__(self, idx):
        """
        Returns:
            page: PIL image (if no further transforms specified), showing the whole page of the book
            region: PIL image (if no further transforms specified), showing the cropped region of the number
            bbox: 1d tensor containing the bounding box as [x, y, w, h]
            number: integer, target number displayed in the bbox
        """
        factor_h, factor_w = 1., 1.

        num_page_id =  self.page_ids[idx]

        img_path = self.get_img_path(num_page_id)
        
        page = self.load_image(img_path)
        if self.transform:
            h0, w0, _ = np.shape(page)
            page = self.transform(page)
            _, h1, w1 = np.shape(page)
            factor_h = float(h1)/h0
            factor_w = float(w1)/w0
        bbox_factors = [factor_w, factor_h, factor_w, factor_h]

        page_data = self.annotations_frame.loc[num_page_id]

        bbox = [item for sublist in page_data.xywh.values for item in sublist]
        bbox = [self.rescale_bbox(box, bbox_factors) for box in bbox] 
        number = list(self.annotations_frame.loc[num_page_id].number)

        if 'element_label' in self.annotations_frame:
            element_label =  list(self.annotations_frame.loc[num_page_id].element_label)
        else:
            element_label = ''
            
        if 'bookstr' in self.annotations_frame:
            bookstr = list(set(self.annotations_frame.loc[num_page_id].bookstr)) + [img_path]
        else:
            bookstr = [img_path]

        if self.return_bookstr:
            return page, bbox, number, element_label, bookstr
        else:
            return page, bbox, number, element_label
    
    def __len__(self):
        # overwrite otherwise returns n_page_ids*n_bboxes
        return len(self.page_ids)

 
 