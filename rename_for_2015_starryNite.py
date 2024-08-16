import os.path
import shutil
from glob import glob

threed_image_path=r'C:\Users\zelinli6\Downloads\Galaxy26-[WT-uncompressed\uncompressed-EMB01'
saving_path=r'C:\Users\zelinli6\Downloads\Galaxy26-[WT-uncompressed\EMB01'
embryo_name='uncompressed-EMB01'

for single_tp_image_path in glob(os.path.join(threed_image_path,'*.tif')):
    tp=os.path.basename(single_tp_image_path).split('.')[0].split('_')[-1][1:]
    saving_image_path=os.path.join(saving_path,embryo_name+tp+'.tif')
    shutil.copy(single_tp_image_path,saving_image_path)