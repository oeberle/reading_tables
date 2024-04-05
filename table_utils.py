import os
from skimage import io
from PIL import Image
import torchvision.transforms as transforms
import pickle
import numpy as np

def get_data(res_pfile, file_map = None):
    # Get function to retrieve histograms (H) and page_ids (M)
    
    res_ = pickle.load(open(res_pfile, 'rb'))
    
    if file_map:
        M = [x.replace('/home/space/sphaera/sphaera_tables', file_map) for x in res_['M']]
    else:
        M = res_['M']
    
    H, M = np.array(res_['H']), np.array(M)
    return H, M

def get_meta(M_all):
    # Decode meta information from page_id
    
    all_pages = [(os.path.split(d)[1].replace('.jpg',''), d) for d in M_all]
    years = [int(p[0].split('_')[-2]) for p in all_pages]
    years = np.array(years)
    books = np.array(['_'.join((p[0].split('_')[:-1])) for p in all_pages])

    books_unique= sorted(list(set(books)))
    book_dict = {b:i for i,b in enumerate(books_unique)}

    return all_pages, years, books, books_unique,  book_dict


def nr_from_index(ind):
    # get number feature from index
    
    if ind < 10:
        nr = str(ind)
    elif ind < 20:
        nr = '0' + str(ind-10)
    else:
        nr = str(ind-10)
    return nr

def get_book_page(m):
    # get book and page from page_id

    page_id = os.path.split(m)[1]
    splits = page_id.split('_')
    book = '_'.join(splits[:-1])
    page = int(splits[-1].replace('.jpg','').replace('p',''))    
    return book, page


def load_image(path, gray=True):
    # load PIL image and convert to three channel RGB image.
    
    page = io.imread(path)
    page = Image.fromarray(page).convert('RGB') # some pages are scanned in greyscale, i.e. single channel, therefore we convert all images to RGB
    if gray:
        page = transforms.Grayscale(num_output_channels=3)(page)
    return page


def get_hist(y, K):
    # get histogram of occurences y in from K classes
    
    h = np.zeros(K)
    for y_ in y:
        h[y_]+=1
    return h
