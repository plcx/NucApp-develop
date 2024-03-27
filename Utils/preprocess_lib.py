import os
import glob
import warnings
import shutil

import nibabel

warnings.filterwarnings("ignore")
import numpy as np
from PIL import Image
import nibabel as nib
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
from skimage.transform import resize
from Utils.utils import save_indexed_tif
import traceback


def stack_nuc_slices2(para):
    try:
        [origin_files, save_folder, embryo_name, tp, out_size, num_slice, res] = para

        first_img_idx = (tp - 1) * num_slice
        bottom_img = np.asanyarray(Image.open(origin_files[first_img_idx]))
        bottom_img = np.expand_dims(bottom_img, axis=0)
        save_file_name = "{}_{}_rawNuc.nii.gz".format(embryo_name, str(tp).zfill(3))

        for idx in range(first_img_idx + 1, tp * num_slice):
            raw_file_name = origin_files[idx]
            img = np.asanyarray(Image.open(raw_file_name))
            img = np.expand_dims(img, axis=0)
            bottom_img = np.concatenate((img, bottom_img), axis=0)

        img_stack = np.transpose(bottom_img, axes=(1, 2, 0))
        img_stack = resize(image=img_stack, output_shape=out_size, preserve_range=True, order=1).astype(np.uint16)
        nib_stack = nib.Nifti1Image(img_stack, np.eye(4))
        nib_stack.set_qform(np.eye(4), code='aligned')
        nib_stack.header["pixdim"] = [1.0, res[0], res[1], res[2], 0., 0., 0., 0.]
        nib_stack.header.set_xyzt_units(xyz=3, t=8)
        nib_stack.header["xyzt_units"] = 10
    except:
        pass

    return nib_stack


def stack_nuc_slices(para):
    """
    作用是将每个时间点的所有阶段的照片堆叠在一起,形成一个三维立体的图片
    :param para:
    :return:
    """
    try:
        [origin_files, save_folder, embryo_name, tp, out_size, num_slice, res] = para

        out_stack = []
        save_file_name = "{}_{}_rawNuc.nii.gz".format(embryo_name, str(tp).zfill(3))
        save_tif_name = "{}_{}_rawNuc.tif".format(embryo_name, str(tp).zfill(3))
        for idx in range((tp - 1) * num_slice, tp * num_slice):
            raw_file_name = origin_files[idx]
            img = np.asanyarray(Image.open(raw_file_name))
            out_stack.insert(0, img)

        img_stack = np.transpose(np.stack(out_stack), axes=(1, 2, 0))
        img_stack = resize(image=img_stack, output_shape=out_size, preserve_range=True, order=1).astype(np.uint16)
        nib_stack = nib.Nifti1Image(img_stack, np.eye(4))
        nib_stack.set_qform(np.eye(4), code='aligned')
        nib_stack.header["pixdim"] = [1.0, res[0], res[1], res[2], 0., 0., 0., 0.]
        nib_stack.header["xyzt_units"] = 11
        #nib.save(nib_stack, os.path.join(save_folder, save_file_name))
        tif_save(nib_stack.get_fdata(), os.path.join(save_folder, save_tif_name), False)

    except Exception as e:
        return "Threadpool return exception: {}".format(e)


def stack_memb_slices(para):
    try:
        [origin_files, save_folder, embryo_name, tp, out_size, num_slice, res] = para

        out_stack = []
        save_file_name = "{}_{}_rawMemb.nii.gz".format(embryo_name, str(tp).zfill(3))
        for idx in range((tp - 1) * num_slice, tp * num_slice):
            raw_file_name = origin_files[idx]
            img = np.asanyarray(Image.open(raw_file_name))
            out_stack.insert(0, img)
        img_stack = np.transpose(np.stack(out_stack), axes=(1, 2, 0))
        img_stack = resize(image=img_stack, output_shape=out_size, preserve_range=True, order=1).astype(np.uint8)
        nib_stack = nib.Nifti1Image(img_stack, np.eye(4))
        nib_stack.set_qform(np.eye(4), code='aligned')
        nib_stack.header["pixdim"] = [1.0, res[0], res[1], res[2], 0., 0., 0., 0.]
        nib_stack.header["xyzt_units"] = 11
        nib.save(nib_stack, os.path.join(save_folder, save_file_name))
    except Exception as e:
        return "Threadpool return exception: {}".format(e)
    """
    pixdim体素维度：每个体素维度信息都保存在pixdim中，各自对应dim，
    但pixdim[0]有特殊意义，其值只能是 - 1或1。
    pixdim[1]对应x轴，pixdim[2]对应y轴, pixdim[3]对应z轴
    sform的前三个维度将在srow_x, srow_y, srow_z字段中指定。
    """

def tif_save(nifti_file_arr, ouput_tif, segmented):
    obj_selection_index_list = []
    if segmented is False and np.max(nifti_file_arr) < 255:
        nifti_file_arr = (nifti_file_arr * 255 / np.max(nifti_file_arr)).astype(np.uint)
    save_indexed_tif(ouput_tif, nifti_file_arr, segmented=segmented,
                        obj_selection_index_list=obj_selection_index_list,is_seperate=False)


if __name__ == '__main__':
    # origin_files =  glob.glob(os.path.join("C:\CellAltas\MembRaw\MembRaw", "181210plc1p1", "tif", "*.tif"))
    #origin_files2 = glob.glob(os.path.join("C:\CellAltas\MembRaw\MembRaw", "181210plc1p1", "tifR", "*.tif"))
    #para = (origin_files2, "./", "test", 1, [512, 712, 28], 68, [0.999, 0.99, 0.99])
    #test = stack_nuc_slices2(para)
    #print(test.get_sform(coded=True))
    #print(test.get_qform(coded=True))
    #print(test.header)
    input_dir = r'C:\Users\User\Desktop\181210plc1p1_012_rawMemb.nii.gz'
    output_dir = r'C:\Users\User\Desktop\181210plc1p1_012_rawMemb.tif'
    #nifti2tiff(input_dir, output_dir, False)

