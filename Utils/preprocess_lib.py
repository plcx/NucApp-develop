import os
import glob
import warnings
# import shutil

# import nibabel

warnings.filterwarnings("ignore")
import numpy as np
from PIL import Image
# import pandas as pd
# from tqdm import tqdm
# import multiprocessing as mp
from skimage.transform import resize
from Utils.utils import save_indexed_tif


def stack_nuc_slices(para):
    """
    作用是将每个时间点的所有阶段的照片堆叠在一起,形成一个三维立体的图片
    :param para:
    :return:
    """
    try:
        [origin_tif_files_path_list, save_folder, embryo_name, tp, out_size] = para

        out_stack = []
        # save_file_name = "{}_{}_rawNuc.nii.gz".format(embryo_name, str(tp).zfill(3))
        save_tif_name = "{}_{}_rawNuc.tif".format(embryo_name, str(tp).zfill(3))

        for raw_tif_path in origin_tif_files_path_list:
            # raw_file_name = origin_tif_files_path[idx]
            img = np.asanyarray(Image.open(raw_tif_path))
            out_stack.insert(0, img)

        img_stack = np.transpose(np.stack(out_stack), axes=(1, 2, 0))
        img_stack = resize(image=img_stack, output_shape=out_size, preserve_range=True, order=1).astype(np.uint16)
        # nib_stack = nib.Nifti1Image(img_stack, np.eye(4))
        # nib_stack.set_qform(np.eye(4), code='aligned')
        # nib_stack.header["pixdim"] = [1.0, res[0], res[1], res[2], 0., 0., 0., 0.]
        # nib_stack.header["xyzt_units"] = 11
        #nib.save(nib_stack, os.path.join(save_folder, save_file_name))
        save_indexed_tif(os.path.join(save_folder, save_tif_name),img_stack)
        # tif_save(img_stack, os.path.join(save_folder, save_tif_name), False)

    except Exception as e:
        return "Threadpool return exception: {}".format(e)
