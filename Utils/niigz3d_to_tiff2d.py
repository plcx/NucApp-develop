from skimage.transform import resize
import numpy as np
import os
# from PIL import Image
import tifffile
from Utils.data_io import nib_load, check_folder


def seperate_3dniigz_to_2dtif(para):
    source_niigz_file_path, target_tif_root_path, raw_tiff_root, raw_img_shape =para
    embryo_name,tp= os.path.basename(source_niigz_file_path).split('.')[0].split('_')[:2]
    raw_nuc_seg3d_volume=nib_load(source_niigz_file_path)
    # reshaped_3d_volume=resize(image=raw_nuc_seg3d_volume,output_shape=target_shape,preserve_range=True,order=0).astype(np.uint8)
    for page_num in range(1, raw_img_shape[-1] + 1):

        raw_index= raw_img_shape[-1] + 1 - page_num
        index_in_3d_seg_nuc=int(page_num * raw_nuc_seg3d_volume.shape[-1] / raw_img_shape[-1])
        # tif_file_name='{}_L1-t{}-p{}.tif'.format(embryo_name,str(tp).zfill(3),str(page_num).zfill(2))

        raw_tif_file_name='{}_L1-t{}-p{}.tif'.format(embryo_name,str(tp).zfill(3),str(raw_index).zfill(2))
        raw_tif_file_path=os.path.join(raw_tiff_root, embryo_name,'tif', raw_tif_file_name)
        saving_tif_path=os.path.join(target_tif_root_path, raw_tif_file_name)


        raw_tif_image_this_p=tifffile.imread(raw_tif_file_path)
        biggest_grey =np.max(raw_tif_image_this_p)
        tif_image_this_p=resize(image=raw_nuc_seg3d_volume[:,:,index_in_3d_seg_nuc-1], output_shape=raw_img_shape[:2], preserve_range=True, order=0)
        raw_tif_image_this_p[tif_image_this_p>0]=biggest_grey
        image_array_this_p=raw_tif_image_this_p.astype(np.uint8)
        check_folder(saving_tif_path)

        # tif_image=Image.fromarray(image_array_this_p, mode="L")
        tifffile.imwrite(saving_tif_path, image_array_this_p,

                         byteorder='>',
                         # bigtiff=True,
                         imagej=True,
                         # ome=True,
                         resolution=(10.981529, 10.981529),
                         # 'info':{'compression':'raw'},
                         metadata={'size': (raw_img_shape[1], raw_img_shape[0]),
                                   'height': raw_img_shape[0],
                                   'width': raw_img_shape[1],
                                   'use_load_libtiff': False,
                                   'tile': [('raw', (0, 0, raw_img_shape[1], raw_img_shape[0]), 396, ('L', 0, 1))]})

        # tif_image.save(saving_tif_path,format='TIFF')


# def nuc_enhance(raw_tiff_path, pred_niigz_path, saving_enhanced_tiff_path, embryo_name, max_time, raw_tif_shape):
#     # embryo_names = [embryo_name]
#     # max_times = [max_time]
#     # niigz_source_path = os.path.join(saving_enhanced_tiff_path, 'RawStack', embryo_name, 'SegNuc')
#     # saving_target_tif_root = os.path.join(saving_enhanced_tiff_path, 'RawStack')
#     # for embryo_i, embryo_name in enumerate(embryo_names):
#

