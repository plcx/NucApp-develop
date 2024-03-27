
import numpy as np
import nibabel as nib

import os
import shutil
from PIL import Image
import math
from skimage.transform import resize
from tqdm import tqdm
from glob import glob
import pandas as pd

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

def save_indexed_tif(file_name, data,segmented=True,obj_selection_index_list=[],is_seperate=True):
    """Save matrix data as indexed images which can be rendered by ImageJ"""
    # label_
    check_folder(file_name)
    # name,tp=os.path.basename(file_name).split('.')[0].split('_')[:2]
    if segmented:
        embryo_name=os.path.basename(file_name).split('.')[0].split('_')[0]
        num_slices = data.shape[-1]


        if not is_seperate:
            tif_imgs = []
            for i_slice in range(num_slices):
                label_map = data[..., i_slice]  # avoid 256 become 0
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
            return

        number_dictionary_path = r'necessary_files\name_dictionary_sample4.csv'
        label_name_dict = pd.read_csv(number_dictionary_path, index_col=0).to_dict()['0']

        mapping_dict={}
        for cell_label in list(np.unique(data))[1:]:
            mapping_dict[cell_label]=[int((cell_label+1)/256)+1,(cell_label+1)%256]
        tp=os.path.basename(file_name).split('.')[0].split('_')[1]
        map_txt_saving_path=os.path.join(os.path.dirname(os.path.dirname(file_name)),'tiffmaptxt',embryo_name,embryo_name+'_'+tp+'_map.txt')
        print(map_txt_saving_path)
        check_folder(map_txt_saving_path)
        tmp_label_0_cell_list=[]
        with open(map_txt_saving_path, 'w') as file:
            for key, value in mapping_dict.items():
                file.write(f'{label_name_dict[key]}\n')
                #file.write(f'{label_name_dict.get(key, "Unknown Label")}\n')
                if int(value[1]) == 0:
                    file.write(f'{int(key)}:{0}:{int(value[0])}\n')
                    tmp_label_0_cell_list.append([key,int(value[0])])
                else:
                    file.write(f'{int(key)}:{int(value[0])}:{int(value[1])}\n')

        # 0--------mask
        if len(tmp_label_0_cell_list)>0:
            data_this = np.zeros(data.shape)

            for cell_label,tmp_label in tmp_label_0_cell_list:
                data_this[data==cell_label]=tmp_label
            # no plus 1 because base on itself
            obj_selection_index_list.append(str(list(np.unique(data_this.astype(np.uint8)))[-1]))

            tif_imgs = []
            for i_slice in range(num_slices):
                label_map = data_this[..., i_slice]  # avoid 256 become 0
                label_map_out = np.squeeze((label_map).astype(np.uint8))
                tif_img = Image.fromarray(label_map_out, mode="P")
                tif_img.putpalette(P)
                tif_imgs.append(tif_img)
            tif_saving = file_name.split('.')[0] + '_{}.tif'.format(str(0).zfill(3))
            if os.path.isfile(tif_saving):
                os.remove(tif_saving)
            # save the 1th slice image, treat others slices as appending
            tif_imgs[0].save(tif_saving, save_all=True, append_images=tif_imgs[1:])

        seperate_mask_tmp=((data+1)/256).astype(np.uint8)+1
        seperate_mask_tmp[data==0]=0
        # seperate_mask_list={}
        for seperate_idx in list(np.unique(seperate_mask_tmp)[1:]):
            seperate_mask_this=(seperate_mask_tmp!=seperate_idx)
            # print(seperate_idx,np.unique(seperate_mask_this,return_counts=True),len(np.unique(seperate_mask_this)))
            data_this=data.copy()
            data_this[seperate_mask_this]=0
            # plus 1 beacuae it is based on data
            obj_selection_index_list.append(str(list(np.unique((data_this+1).astype(np.uint8)))[-1]))

            tif_imgs = []
            for i_slice in range(num_slices):
                label_map = data_this[..., i_slice] # avoid 256 become 0
                # print('seg data',np.unique(label_map)) # build match map
                # print('seg data',np.unique(label_map)%256+1) # build match map

                label_map_out = np.squeeze((label_map+1).astype(np.uint8))
                # print('seg data',np.unique(label_map_out)) # build match map

                label_map_out[label_map==0]=0

                tif_img = Image.fromarray(label_map_out, mode="P")
                tif_img.putpalette(P)
                # print('image data',np.unique(np.asarray(tif_img))[0:])
                # tif_img = tif_img.convert("P", palette=P, colors=256)

                # tif_img=tif_img.convert("RGB")
                # print('seg data',np.unique(label_map,return_counts=True))
                #
                # print('image data',np.unique(np.asarray(tif_img))[0:])
                tif_imgs.append(tif_img)
            tif_saving=file_name.split('.')[0]+'_{}.tif'.format(str(seperate_idx).zfill(3))
            if os.path.isfile(tif_saving):
                os.remove(tif_saving)

            # save the 1th slice image, treat others slices as appending
            tif_imgs[0].save(tif_saving, save_all=True, append_images=tif_imgs[1:])
    else:
        tif_imgs = []
        num_slices = data.shape[-1]
        for i_slice in range(num_slices):
            label_map=data[..., i_slice]
            # raw_map=label_map.copy()
            # if label_map.max() > 255:
            #     label_map = np.remainder(label_map, 255)
            label_map = np.squeeze(label_map.astype(np.uint8))
            # label_map[np.logical_and(raw_map, ~label_map)] = 255
            tif_img = Image.fromarray(label_map, mode="P")
            # tif_img.putpalette(P)
            tif_imgs.append(tif_img)
        if os.path.isfile(file_name):
            os.remove(file_name)

        # save the 1th slice image, treat others slices as appending
        tif_imgs[0].save(file_name, save_all=True, append_images=tif_imgs[1:])

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