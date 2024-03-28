import nibabel
from skimage import io as ski_io
import glob
import numpy as np
import nibabel as nib
import os
import shutil
from PIL import Image
import math
from skimage.transform import resize
from tqdm import tqdm
from glob import glob
import glob
import pandas as pd
from scipy.ndimage import binary_closing
from skimage.transform import resize, rescale


# generate 768 tiff figure drawing put palette
P = [252, 233, 79, 114, 159, 207, 239, 41, 41, 173, 127, 168, 138, 226, 52,
     233, 185, 110, 252, 175, 62, 211, 215, 207, 196, 160, 0, 32, 74, 135, 164, 0, 0,
     92, 53, 102, 78, 154, 6, 143, 89, 2, 206, 92, 0, 136, 138, 133, 237, 212, 0, 52,
     101, 164, 204, 0, 0, 117, 80, 123, 115, 210, 22, 193, 125, 17, 245, 121, 0, 186,
     189, 182, 85, 87, 83, 46, 52, 54, 238, 238, 236, 0, 0, 10, 252, 233, 89, 114, 159,
     217, 239, 41, 51, 173, 127, 178, 138, 226, 62, 233, 185, 120, 252, 175, 72, 211, 215,
     217, 196, 160, 10, 32, 74, 145, 164, 0, 10, 92, 53, 112, 78, 154, 16, 143, 89, 12,
     206, 92, 10, 136, 138, 143, 237, 212, 10, 52, 101, 174, 204, 0, 10, 117, 80, 133, 115,
     210, 32, 193, 125, 27, 245, 121, 10, 186, 189, 192, 85, 87, 93, 46, 52, 64, 238, 238, 246]

P = P * math.floor(255*3/len(P))
l = int(255 - len(P)/3)
P = P + P[3:(l+1)*3]
P = [0,0,0] + P



def read_indexed_png(fname):
    im = Image.open(fname)
    palette = im.getpalette()
    im = np.array(im)
    return im, palette


def save_indexed_png(fname, label_map, palette=P):
    if label_map.max() > 255:
        label_map = np.remainder(label_map, 255)
    label_map = np.squeeze(label_map.astype(np.uint8))
    im = Image.fromarray(label_map, 'P')
    im.putpalette(palette)
    im.save(fname, 'PNG')

def save_indexed_tif(file_name, volume_arr):
    """Save matrix data as indexed images which can be rendered by ImageJ"""
    # label_
    check_folder(file_name)
    # name,tp=os.path.basename(file_name).split('.')[0].split('_')[:2]
    # embryo_name=os.path.basename(file_name).split('.')[0].split('_')[0]
    num_slices = volume_arr.shape[-1]

    tif_imgs = []
    for i_slice in range(num_slices):
        label_map = volume_arr[..., i_slice]  # avoid 256 become 0
        label_map_out = np.squeeze((label_map + 1).astype(np.uint8))
        label_map_out[label_map == 0] = 0
        tif_img = Image.fromarray(label_map_out, mode="P")
        tif_img.putpalette(P)
        tif_imgs.append(tif_img)
    tif_saving=file_name.split('.')[0]+'.tif'
    if os.path.isfile(tif_saving):
        os.remove(tif_saving)
    # save the 1th slice image, treat others slices as appending
    tif_imgs[0].save(tif_saving, save_all=True, append_images=tif_imgs[1:])

def scale2index(seg0):
    """Rescale all labels into range [0, 255]"""
    seg = seg0 % 255
    reduce_mask = np.logical_and(seg0!=0, seg==0)
    seg[reduce_mask] = 255  # Because only 255 colors are available, all cells should be numbered within [0, 255].
    seg = seg.astype(np.uint8)

    return seg

def isotropic_resolution(src_folder, dst_folder=None, target_res=None):
    src_files = glob(os.path.join(src_folder, "*.nii.gz"))
    dst_folder = os.path.dirname(src_files[0]) if dst_folder is None else dst_folder
    target_res = 0.09 if target_res is None else target_res

    for src_file in tqdm(src_files, desc="Processing {}".format(src_folder)):
        img = nib.load(src_file)
        data = img.get_fdata()
        header = img.header
        origin_shape = list(data.shape)
        origin_res = header["pixdim"][1:4].tolist()

        header["pixdim"] = [1.0, target_res, target_res, target_res, 0., 0., 0., 0.]

        target_shape = [int(a / target_res * y) for a, y in zip(origin_res, origin_shape)]
        data = resize(image=data, output_shape=target_shape, preserve_range=True, order=1).astype(np.uint8)

        img = nib.Nifti1Image(data, np.eye(4), header)
        nib.save(img, os.path.join(dst_folder, os.path.basename(src_file)))

def check_folder(file_folder, overwrite=False):
    if "." in os.path.basename(file_folder):
        file_folder = os.path.dirname(file_folder)
    if os.path.isdir(file_folder) and overwrite:
        shutil.rmtree(file_folder)
    elif not os.path.isdir(file_folder):
        os.makedirs(file_folder)

def nib_save(file_name, data, overwrite=False):
     check_folder(file_name, overwrite)
     img = nib.Nifti1Image(data, np.eye(4))
     nib.save(img, file_name)

def nib_load(file_name):
     assert os.path.isfile(file_name), "File {} not exist".format(file_name)

     return nib.load(file_name).get_fdata()

def get_boundary(seg, b_width=1):
    """
    Get boundary of instance segmentation as white front pixels
    """
    padded = np.pad(seg, b_width, mode='edge')

    border_pixels = np.logical_and(
        np.logical_and(seg == padded[:-2*b_width, b_width:-b_width, b_width:-b_width], seg == padded[2*b_width:, b_width:-b_width, b_width:-b_width]),
        np.logical_and(seg == padded[b_width:-b_width, :-2*b_width, b_width:-b_width], seg == padded[b_width:-b_width, 2*b_width:, b_width:-b_width])
    )

    border_pixels = np.logical_and(
        border_pixels,
        np.logical_and(seg == padded[b_width:-b_width, b_width:-b_width, :-2 * b_width],seg == padded[b_width:-b_width, b_width:-b_width, 2 * b_width:])
    )


    # border_pixels = np.logical_not(border_pixels).astype(np.uint8)
    border_pixels = (border_pixels == 0).astype(np.uint8)

    return border_pixels * 1


def tiff2nifti(root, target):
    tiff_file_paths = glob.glob(os.path.join(root, '*.tif'))
    for tiff_file_path in tiff_file_paths:
        tiff_file_arr = ski_io.imread(tiff_file_path)
        print(tiff_file_path, np.max(tiff_file_arr), np.min(tiff_file_arr))
        namelist = os.path.basename(tiff_file_path).split('.')[0].split('_')
        save_name = namelist[1] + '_' + namelist[2]
        # nib_save(tiff_file_arr,os.path.join(target,os.path.basename(tiff_file_path)))
        nib_save(os.path.join(target, save_name + '_segCell.nii.gz'), np.transpose(tiff_file_arr, axes=(1, 2, 0)))


def nifti2tiff_seperated(root, target, segmented):
    nifti_file_paths = glob.glob(os.path.join(root, '*.nii.gz'))
    obj_selection_index_list = []
    saving_obj_selection_index_list = os.path.join(os.path.dirname(target), "{}_render_indexed.txt".format(
        os.path.basename(target).split('.')[0]))

    print(saving_obj_selection_index_list)
    for nifti_file_path in nifti_file_paths:
        nifti_file_arr = nibabel.load(nifti_file_path).get_fdata()
        # print(np.unique(nifti_file_arr,return_counts=True))
        # target_shape = [int(x / 2) for x in nifti_file_arr.shape]
        # nifti_file_arr = resize(image=nifti_file_arr, output_shape=target_shape, preserve_range=True, order=0).astype(np.uint8)

        if segmented is False and np.max(nifti_file_arr) < 255:
            nifti_file_arr = (nifti_file_arr * 255 / np.max(nifti_file_arr)).astype(np.uint)
        # print(nifti_file_arr.shape)
        # print(np.unique(nifti_file_arr,return_counts=True))
        # print(np.unique(nifti_file_arr))
        # nifti_file_arr=scale2index(nifti_file_arr)
        # print(np.unique(nifti_file_arr))
        # embryo_name = os.path.basename(nifti_file_path).split(".")[0].split('_')[0]
        # tp = os.path.basename(nifti_file_path).split(".")[0].split('_')[1]
        save_file_path = os.path.join(target, os.path.basename(nifti_file_path).split(".")[0] + ".tif")
        target_shape_scale=1
        resize_seg_array = rescale(nifti_file_arr, scale=target_shape_scale, preserve_range=True, mode='constant', order=0,anti_aliasing=False)
        save_indexed_tif(save_file_path, resize_seg_array, segmented=segmented,
                         obj_selection_index_list=obj_selection_index_list)
        # Open the file for writing
        saving_obj_selection_index_list = os.path.join(os.path.dirname(target), "{}_render_indexed.txt".format(
        os.path.basename(target).split('.')[0]))

    if segmented:
        # print(saving_obj_selection_index_list, len(obj_selection_index_list))
        assert len([name for name in os.listdir(target)]) == len(obj_selection_index_list)
        with open(saving_obj_selection_index_list, "w") as f:
            # Write each string to a new line in the file
            for string in obj_selection_index_list:
                f.write(string + "\n")

def nifti2tiff(root, target, segmented):
    nifti_file_paths = glob.glob(os.path.join(root, '*.nii.gz'))
    obj_selection_index_list = []
    for nifti_file_path in nifti_file_paths:
        nifti_file_arr = nibabel.load(nifti_file_path).get_fdata()
        if segmented is False and np.max(nifti_file_arr) < 255:
            nifti_file_arr = (nifti_file_arr * 255 / np.max(nifti_file_arr)).astype(np.uint)
        save_file_path = os.path.join(target, os.path.basename(nifti_file_path).split(".")[0] + ".tif")
        save_indexed_tif(save_file_path, nifti_file_arr, segmented=segmented,
                         obj_selection_index_list=obj_selection_index_list,is_seperate=False)



def nift2npy(root, target):
    nifti_file_paths = glob.glob(os.path.join(root, '*.nii.gz'))
    for nifti_file_path in nifti_file_paths:
        nifti_file_arr = nibabel.load(nifti_file_path).get_fdata()
        embryo_name = os.path.basename(nifti_file_path).split(".")[0].split('_')[0]
        tp = os.path.basename(nifti_file_path).split(".")[0].split('_')[1]
        save_file_path = os.path.join(target, embryo_name + '_' + tp + ".npy")
        np.save(save_file_path, nifti_file_arr)


def nift2npy_3type(root, target):
    nifti_file_paths = glob.glob(os.path.join(root, 'SegCell/*.nii.gz'))
    for nifti_file_path in nifti_file_paths:
        nifti_file_arr = nibabel.load(nifti_file_path).get_fdata()
        embryo_name = os.path.basename(nifti_file_path).split(".")[0].split('_')[0]
        tp = os.path.basename(nifti_file_path).split(".")[0].split('_')[1]
        # -------------
        save_file_path_1 = os.path.join(target, 'masks', embryo_name + '_' + tp + "_foreground.npy")
        foreground_arr = nifti_file_arr.copy()
        foreground_arr[nifti_file_arr != 0] = 1
        print('foreground', np.unique(foreground_arr, return_counts=True))
        np.save(save_file_path_1, foreground_arr)
        # -----------------
        save_file_path_2 = os.path.join(target, 'masks', embryo_name + '_' + tp + "_background.npy")
        background_arr = np.ones(nifti_file_arr.shape)
        binary_closing_arr_back = binary_closing(nifti_file_arr != 0, iterations=5)
        background_arr[binary_closing_arr_back] = 0
        np.save(save_file_path_2, background_arr)
        print('backgaround', np.unique(background_arr, return_counts=True))
        # ------------------
        save_file_path_3 = os.path.join(target, 'masks', embryo_name + '_' + tp + "_membrane.npy")
        nifti_file_path_membrane = os.path.join(root, 'SegMemb', embryo_name + '_' + tp + '_segMemb.nii.gz')
        nifti_file_arr_membrane = nibabel.load(nifti_file_path_membrane).get_fdata()
        np.save(save_file_path_3, nifti_file_arr_membrane)
        print('membrane', np.unique(nifti_file_arr_membrane, return_counts=True))


if __name__ == "__main__":
    root = r'E:\NucleiSegmentation\CellAppData\RawStack\181210plc1p1\RawNuc'
    target = r'E:\NucleiSegmentation\CellAppData\RawStack\181210plc1p1\RawNucTif'
    nifti2tiff(root, target, False)

