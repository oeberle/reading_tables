from utils import set_up_dir
import sys
import os
from tqdm import tqdm
import glob
from dataloaders.proc_utils import read_image_gray, binarize, write_image_binary
from configs import SACROBOSCO_DATA_ROOT

RAW_DATA_DIR = os.path.join(SACROBOSCO_DATA_ROOT, 'raw') # Assuming raw image files are in subfolder 'raw'

PROC_DIR = RAW_DATA_DIR.replace('raw', 'processed')
set_up_dir(PROC_DIR)

all_files = glob.glob(os.path.join(RAW_DATA_DIR,'*.jpg'))

for file_path in tqdm(all_files):
    file_path_bin = file_path.replace('raw', 'processed')
    if not os.path.exists(file_path_bin):
        image = read_image_gray(file_path)
        image = binarize(image)
        _ = write_image_binary(file_path_bin, image)

        
        