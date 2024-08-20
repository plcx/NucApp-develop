import os
import glob
import pickle
import warnings
import shutil

from Utils.data_io import nib_save
# from Utils.generate_name_dictionary import add_number_dict

warnings.filterwarnings("ignore")
import numpy as np
from PIL import Image
# import nibabel as nib
import pandas as pd
# from skimage.exposure import rescale_intensity
from skimage.morphology import ball

import pandas
from tqdm import tqdm
import multiprocessing as mp
from skimage.transform import resize
from scipy import ndimage

# from Utils.utils import check_folder
from Utils.data_structure import read_cd_file, read_txt_cd_file


def combine_slices(config):
    """
    Combine slices into stack images
    :param config: parameters
    :return:
    """
    # signal.emit(True,'sss')
    num_slice = config["num_slice"]
    embryo_names = config["embryo_names"]
    max_times = config["max_times"]
    xy_res = config["xy_resolution"]
    z_res = config["z_resolution"]
    out_size = config["out_size"]
    raw_folder = config["raw_folder"]
    stack_folder = config["target_folder"]
    is_save_nuc_channel = config["save_nuc"]
    is_save_memb_channel = config["save_memb"]
    is_save_seg_cell_with_cd_file = config["is_ace_cd_file"]
    # is_reading_txt_cdfile=

    # get output size
    if is_save_nuc_channel:
        raw_image_files = glob.glob(os.path.join(raw_folder, embryo_names[0], "tif", "*.tif"))
    else:
        raw_image_files = glob.glob(os.path.join(raw_folder, embryo_names[0], "tifR", "*.tif"))
    # print(raw_memb_files)
    raw_size = list(np.asarray(Image.open(raw_image_files[0])).shape) + [int(num_slice * z_res / xy_res)]
    out_res = [res * in_scale / out_scale for res, in_scale, out_scale in
               zip([xy_res, xy_res, xy_res], raw_size, out_size)]
    # print(out_res)

    # multiprocessing
    mpPool = mp.Pool(mp.cpu_count() - 1)

    for idx_embryo, embryo_name in enumerate(embryo_names):

        # get lineage file
        if is_save_seg_cell_with_cd_file:
            csv_cd_file_path=os.path.join(config["raw_folder"], embryo_name, "aceNuc",
                                             "CD{}.csv".format(embryo_name))
            if os.path.exists(csv_cd_file_path):
                lineage_file_path = os.path.join(config["raw_folder"], embryo_name, "aceNuc",
                                             "CD{}.csv".format(embryo_name))
            else:
                lineage_file_path = os.path.join(config["raw_folder"], embryo_name, "aceNuc",
                                                 "CD{}.txt".format(embryo_name))
        else:
            lineage_file_path = None

        # ======================= || save nucleus
        if is_save_nuc_channel:
            origin_folder = os.path.join(raw_folder, embryo_name, "tif")
            target_folder = os.path.join(stack_folder, embryo_name, "RawNuc")
            if not os.path.isdir(target_folder):
                os.makedirs(target_folder)

            configs = []
            for tp in range(1, max_times[idx_embryo] + 1):
                configs.append((origin_folder, target_folder, embryo_name, tp, out_size, num_slice, out_res))

            for idx, _ in enumerate(tqdm(mpPool.imap_unordered(stack_nuc_slices, configs), total=len(configs),
                                         desc="1/3 Stack nucleus of {}".format(embryo_name))):
                pass

        # =============================
        if is_save_memb_channel:
            origin_folder = os.path.join(raw_folder, embryo_name, "tifR")
            target_folder = os.path.join(stack_folder, embryo_name, "RawMemb")
            if not os.path.isdir(target_folder):
                os.makedirs(target_folder)

            configs = []
            for tp in range(1, max_times[idx_embryo] + 1):
                configs.append((origin_folder, target_folder, embryo_name, tp, out_size, num_slice, out_res))
                # stack_memb_slices(configs[0])
            for idx, _ in enumerate(tqdm(mpPool.imap_unordered(stack_memb_slices, configs), total=len(configs),
                                         desc="2/3 Stack membrane of {}".format(embryo_name))):
                # TODO: Process Name: `2/3 Stack membrane`; Current status: `idx`; Final status: max_time
                pass

        # save nucleus
        if is_save_seg_cell_with_cd_file:
            assert lineage_file_path is not None
            target_folder = os.path.join(stack_folder, embryo_name, "AnnotatedNuc")
            if not os.path.isdir(target_folder):
                os.makedirs(target_folder)
            if os.path.basename(lineage_file_path).split('.')[-1]=='txt':
                pd_lineage = read_txt_cd_file(lineage_file_path)
            else:
                pd_lineage = read_cd_file(lineage_file_path)

            number_dictionary_path = config["name_dictionary"]
            label_name_dict = pd.read_csv(number_dictionary_path, index_col=0).to_dict()['0']
            name_label_dict = {value: key for key, value in label_name_dict.items()}

            configs = []
            for tp in range(1, max_times[idx_embryo] + 1):
                configs.append((embryo_name, name_label_dict, pd_lineage, tp, raw_size, out_size, out_res,
                                xy_res / z_res, target_folder))
                # save_nuc_seg(configs[0])
            for idx, _ in enumerate(tqdm(mpPool.imap_unordered(save_annotated_ace_nuc, configs), total=len(configs),
                                         desc="3/3 Construct nucleus location of {}".format(embryo_name))):
                # TODO: Process Name: `3/3 Construct nucleus location`; Current status: `idx`; Final status: max_time
                pass
            # for tp in range(1, max_time+1):
            #     save_nuc_seg(embryo_name=embryo_name,
            #                  name_dict=name_dict,
            #                  pd_lineage=pd_lineage,
            #                  tp=tp,
            #                  raw_size=raw_size,
            #                  out_size=out_size,
            #                  out_res=out_res,
            #                  dif_res=xy_res/z_res,
            #                  save_folder=target_folder)
            shutil.copy(lineage_file_path, os.path.join(stack_folder, embryo_name))


# ============================================
# save raw nucleus stack
# ============================================
def stack_nuc_slices(para):
    [raw_folder, save_folder, embryo_name, tp, out_size, num_slice, res] = para

    out_stack = []
    save_file_name = "{}_{}_rawNuc.nii.gz".format(embryo_name, str(tp).zfill(3))
    for i_slice in range(1, num_slice + 1):
        # raw_file_name = "{}deconp1_L1-t{}-p{}.tif".format(embryo_name[:-2], str(tp).zfill(3), str(i_slice).zfill(2))
        raw_file_name = "{}_L1-t{}-p{}.tif".format(embryo_name, str(tp).zfill(3), str(i_slice).zfill(2))

        img = np.asanyarray(Image.open(os.path.join(raw_folder, raw_file_name)))
        out_stack.insert(0, img)
    img_stack = np.transpose(np.stack(out_stack), axes=(1, 2, 0))
    img_stack = resize(image=img_stack, output_shape=out_size, preserve_range=True, order=1).astype(np.int16)
    # nib_stack = nib.Nifti1Image(img_stack, np.eye(4))
    # nib_stack.header.set_xyzt_units(xyz=3, t=8)
    # nib_stack.header["pixdim"] = [1.0, res[0], res[1], res[2], 0., 0., 0., 0.]
    save_file = os.path.join(save_folder, save_file_name)
    # check_folder(save_file)
    nib_save(img_stack, save_file)


# ============================================
# save raw membrane stack
# ============================================
def stack_memb_slices(para):
    [raw_folder, save_folder, embryo_name, tp, out_size, num_slice, res] = para

    out_stack = []
    save_file_name = "{}_{}_rawMemb.nii.gz".format(embryo_name, str(tp).zfill(3))
    for i_slice in range(1, num_slice + 1):
        # r"D:\TemDownload\201112plc1_late_Lng\tifR\c elegans 3.lif_Series001_Lng_001_t00_z08_ch01.tif"
        # raw_file_name = "{}deconp1_L1-t{}-p{}.tif".format(embryo_name[:-2], str(tp).zfill(3), str(i_slice).zfill(2))
        raw_file_name = "{}_L1-t{}-p{}.tif".format(embryo_name, str(tp).zfill(3), str(i_slice).zfill(2))

        # transform the image to array and short them in a list
        img = np.asanyarray(Image.open(os.path.join(raw_folder, raw_file_name)))
        if img.shape != (512, 712):
            print('ERRORRRRR', img.shape, raw_file_name)
        out_stack.insert(0, img)

    img_stack = np.transpose(np.stack(out_stack), axes=(1, 2, 0))  # trasnpose the image from zxy to xyz
    # v_min, v_max = np.percentile(img_stack, (0.2, 99.9))  # erase the outrange grayscale
    # img_stack = rescale_intensity(img_stack, in_range=(v_min, v_max), out_range=(0, 255.0))
    # cut xy, interpolate z
    img_stack = resize(image=img_stack, output_shape=out_size, preserve_range=True, order=1).astype(np.int16)
    # nib_stack = nib.Nifti1Image(img_stack, np.eye(4))
    # nib_stack.header.set_xyzt_units(xyz=3, t=8)
    # nib_stack.header["pixdim"] = [1.0, res[0], res[1], res[2], 0., 0., 0., 0.]
    save_file = os.path.join(save_folder, save_file_name)
    nib_save(img_stack, save_file)


# =============================================
# save nucleus segmentation
# =============================================
def save_annotated_ace_nuc(para):
    [embryo_name, name_dict, pd_lineage, tp, raw_size, out_size, out_res, dif_res, save_folder] = para

    zoom_ratio = [y / x for x, y in zip(raw_size, out_size)]
    tp_lineage = pd_lineage[pd_lineage["time"] == tp]
    tp_lineage.loc[:, "x"] = (tp_lineage["x"] * zoom_ratio[0]).astype(np.int16)
    tp_lineage.loc[:, "y"] = (np.floor(tp_lineage["y"] * zoom_ratio[1])).astype(np.int16)
    # tp_lineage.loc[:, "z"] = np.floor(tp_lineage["z"] * (zoom_ratio[2] / dif_res)).astype(np.int16)

    tp_lineage.loc[:, "z"] = (out_size[2] - np.floor(tp_lineage["z"] * (zoom_ratio[2] / dif_res))).astype(np.int16)
    # tp_lineage.loc[:, "size"] = np.floor(tp_lineage["size"] * 2/3).astype(np.int16)


    # !!!! x <--> y !!!!!!!
    nuc_dict = dict(
        zip(tp_lineage["cell"], zip(tp_lineage["y"].values, tp_lineage["x"].values, tp_lineage["z"].values)))
    labels = [name_dict[name] for name in list(nuc_dict.keys())]
    locs = list(nuc_dict.values())
    out_seg = np.zeros(out_size, dtype=np.int16)
    out_seg[tuple(zip(*locs))] = labels
    nucleus_marker_footprint = ball(7 - int(int(tp) / 100))
    out_seg = ndimage.morphology.grey_dilation(out_seg, footprint=nucleus_marker_footprint)
    # out_seg=ndimage.

    # out_seg=out_seg-1
    save_file_name = "_".join([embryo_name, str(tp).zfill(3), "annotatedNuc.nii.gz"])
    # nib_stack = nib.Nifti1Image(out_seg, np.eye(4))
    # nib_stack.header.set_xyzt_units(xyz=3, t=8)
    # nib_stack.header["pixdim"] = [1.0, out_res[0], out_res[1], out_res[2], 0., 0., 0., 0.]
    save_file = os.path.join(save_folder, save_file_name)
    # check_folder(save_file)
    nib_save(out_seg, save_file)


if __name__ == "__main__":
    IS_CD_FILES = True
    # if IS_CD_FILES:
    #     CD_folder = r"F:\packed membrane nucleus 3d niigz\CD FILES"
    #     nuc_files = sorted(glob.glob(os.path.join(CD_folder, "*.csv")))
    #
    #     for idx, nuc_file in enumerate(nuc_files):  # go through all cd files, one cd file for one embryo
    #         add_number_dict(nuc_file, max_time=1000)  # the max time for your data

    config = dict(

        # ====================================
        num_slice=94,
        embryo_names=['221017plc1p2RAWp1'],
        max_times=[240],
        z_resolution=0.42,
        out_size=[256, 356, 224],  # todo: need to be MANUALLY calculated with the vertical image amount
        # ======================================


        xy_resolution=0.18,
        # 94  *   0.43/0.09  *  356/712
        # out_size=[205, 285, 134],  # todo: need to be MANUALLY calculated with the vertical image amount

        raw_folder=r'E:\ProjectData\MembraneProject\AllRawDataBeforeDeconv',
        target_folder=r"C:\Users\zelinli6\Downloads\Packed",
        save_nuc=True,
        save_memb=False,
        is_ace_cd_file=IS_CD_FILES,
        name_dictionary=r"../static/name_dictionary.csv"
    )

    combine_slices(config)
