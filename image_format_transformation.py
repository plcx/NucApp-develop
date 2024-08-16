import os
import numpy as np
import tifffile
from glob import glob
from skimage.exposure import rescale_intensity
import scipy.ndimage
import shutil

from Utils.data_io import check_folder


def convert_3D_tif_to_2D_tif_for_lineage():
    raw_img_shape=(512,512,30)
    embryo_names = {
        'ALR-1_RW10757': 'ALR1RW10757',
        # 'AHA-1_SYS135': 'AHA1SYS135'
    }
    xy_cropped=[[206,416],[126,432]]
    the_root_path_3D = r'C:\Users\zelinli6\Downloads\another3embryos'


    # raw_img_shape=(512,512,30)
    # embryo_names = {
    #     # 'RW10434_uncompressed_20170608_2_s1': 'uncompressedEMB0102',
    #     #             'RW10434_uncompressed_20170608_2_s2': 'uncompressedEMB0304',
    #                 'RW10434_uncompressed_20170608_3_s1': 'uncompressedEMB0607'
    #                 }
    #
    # the_root_path_3D = r'E:\BaiduNetdiskDownload\RW10434_uncompressed_20170608_3'

    # raw_img_shape = (184, 323, 30)
    # embryo_names = {
    #     'uncompressed-EMB05': 'uncompressedEMB05'
    # }
    # the_root_path_3D = r'C:\Users\zelinli6\Downloads\Galaxy26-[WT-uncompressed'

    the_saving_root_path = r'E:\ProjectData\MembraneProject\AllRawData'

    for original_embryo_name, new_embryo_name in embryo_names.items():
        # tif3D_files = glob(os.path.join(the_root_path_3D, original_embryo_name, '*.tif'))
        tif3D_files = glob(os.path.join(the_root_path_3D, original_embryo_name,'{}*.tif'.format(original_embryo_name)))

        for tif3D_file in tif3D_files:
            tp_str = os.path.basename(tif3D_file).split('.')[0].split('_')[-1][1:]
            # CAREFUL! DIFFERENT IMAGE HAVE DIFFERENT USEFUL RANGE
            raw_nuc_arr = np.array(tifffile.imread(tif3D_file))[:,xy_cropped[0][0]:xy_cropped[0][1],xy_cropped[1][0]:xy_cropped[1][1]]
            # raw_nuc_arr = np.array(tifffile.imread(tif3D_file))
            v_min, v_max = np.percentile(raw_nuc_arr, (0.2, 99.9))  # erase the outrange grayscale
            raw_nuc_arr = rescale_intensity(raw_nuc_arr, in_range=(v_min, v_max), out_range=(0, 255.0))
            # raw_nuc_arr[raw_nuc_arr<30]=0
            # # raw_nuc_arr=scipy.ndimage.gaussian_filter(raw_nuc_arr,sigma=3,radius=3)
            # raw_nuc_arr=scipy.ndimage.median_filter(raw_nuc_arr,size=8)
            # raw_nuc_arr[raw_nuc_arr<30]=0
            # raw_nuc_arr=scipy.ndimage.median_filter(raw_nuc_arr,size=8)

            # print(np.unique(raw_nuc_arr,return_counts=True))
            # print(np.min(raw_nuc_arr),np.max(raw_nuc_arr))
            z_shape,y_shape,x_shape = raw_nuc_arr.shape
            assert raw_img_shape[-1]==z_shape
            for z_idx in range(z_shape):
                this_2d_saving_path = os.path.join(the_saving_root_path, new_embryo_name, 'tif',
                                                   '{}_L1-t{}-p{}.tif'.format(new_embryo_name, tp_str.zfill(3),
                                                                              str(z_idx + 1).zfill(2)))
                raw_tif_image_this_p=raw_nuc_arr[z_idx]
                image_array_this_p = raw_tif_image_this_p.astype(np.uint8)
                check_folder(this_2d_saving_path)

                # tif_image=Image.fromarray(image_array_this_p, mode="L")
                tifffile.imwrite(this_2d_saving_path, image_array_this_p,

                                 byteorder='>',
                                 # bigtiff=True,
                                 imagej=True,
                                 # ome=True,
                                 resolution=(10.981529, 10.981529),
                                 # 'info':{'compression':'raw'},
                                 metadata={'size': (y_shape, x_shape),
                                           'height': x_shape,
                                           'width': y_shape,
                                           'use_load_libtiff': False,
                                           'tile': [
                                               ('raw', (0, 0, y_shape, y_shape), 396, ('L', 0, 1))]})

def deconvolution_to_normal_name():

    deconvoluted_path=r'E:\BaiduNetdiskDownload'
    the_saving_root_path = r'E:\ProjectData\MembraneProject\AllRawData'

    embryo_names = {'ALR1RW10757'}
    max_times=[270]
    z_number=[30]
    for emb_idx,embryo_name in enumerate(embryo_names):
        # tif3D_files = glob(os.path.join(the_root_path_3D, original_embryo_name, '*.tif'))
        for tp_this in range(1,max_times[emb_idx]+1):
            for z_number_this in range(1,z_number[emb_idx]):
                original_file_path=os.path.join(deconvoluted_path,embryo_name,'{}_L1-t{}-p{}_cmle_z000.tif'.format(embryo_name,str(tp_this).zfill(3),str(z_number_this).zfill(2)))
                saving_file_path=os.path.join(the_saving_root_path,embryo_name,'tif','{}_L1-t{}-p{}.tif'.format(embryo_name,str(tp_this).zfill(3),str(z_number_this).zfill(2)))
                check_folder(saving_file_path)
                shutil.copy(original_file_path,saving_file_path)
        for tp_this in range(1,max_times[emb_idx]+1):
            original_file_path=os.path.join(deconvoluted_path,embryo_name,'{}_L1-t{}-p{}_cmle.tif'.format(embryo_name,str(tp_this).zfill(3),str(30).zfill(2)))
            saving_file_path=os.path.join(the_saving_root_path,embryo_name,'tif','{}_L1-t{}-p{}.tif'.format(embryo_name,str(tp_this).zfill(3),str(30).zfill(2)))
            check_folder(saving_file_path)
            shutil.copy(original_file_path,saving_file_path)


if __name__ == '__main__':
    convert_3D_tif_to_2D_tif_for_lineage()
