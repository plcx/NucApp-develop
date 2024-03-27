import os
import glob
import warnings
import shutil

warnings.filterwarnings("ignore")
import numpy as np
from PIL import Image
import pandas as pd
import multiprocessing as mp
from skimage.transform import resize
import traceback
import torch
import os
from scipy import ndimage, stats
from skimage.morphology import h_maxima, binary_opening
from skimage.segmentation import watershed
from scipy.ndimage import binary_closing
from skimage.feature import peak_local_max
import nibabel as nib
from scipy.optimize import linear_sum_assignment
import time
import copy
import math
from scipy.spatial import Delaunay
from threading import Thread
from concurrent.futures import ThreadPoolExecutor


def segmentation(configs):
    try:
        with torch.no_grad():
            [data, model, device, save_path] = configs
            model.eval()
            raw_memb = data[0]
            raw_memb_shape = data[1]
            embryo_name_tp = data[2][0]
            raw_memb_shape = (raw_memb_shape[0].item(), raw_memb_shape[1].item(), raw_memb_shape[2].item())
            pred_memb = model(raw_memb.to(device))
            pred_memb = pred_memb[0] if len(pred_memb) > 1 else pred_memb

            pred_memb = pred_memb[0, 0, :, :, :]
            pred_memb = pred_memb.cpu().numpy().transpose([1, 2, 0])
            pred_memb = resize(pred_memb,
                               raw_memb_shape,
                               mode='constant',
                               cval=0,
                               order=1,
                               anti_aliasing=True)

            save_name = os.path.join(save_path, embryo_name_tp + "_segMemb.nii.gz")
            nib_stack = nib.Nifti1Image((pred_memb * 256).astype(np.int16), np.eye(4))
            nib.save(nib_stack, save_name)

    except Exception as e:
        return "Threadpool return exception: {}".format(e)


def instance_segmentation_with_nucleus(para):
    try:
        [memb_files, nuc_files, save_path, tp, structure, size] = para

        embryo_name_tp = "_".join(os.path.basename(memb_files[tp]).split("_")[0:2])

        segNuc_file = nuc_files[tp]
        marker = nib.load(segNuc_file).get_fdata()
        marker = watershell_processing(marker)
        marker[marker > 0] = 1

        memb = nib.load(memb_files[tp]).get_fdata()

        if (len(np.unique(memb)) == 2):
            image = memb
        else:
            image = np.zeros_like(memb)
            image[memb > 200] = 1
        image = (image == 0).astype(np.uint16)

        if structure == "cube":
            struc = np.ones((size, size, size), dtype=bool)
            marker = ndimage.binary_dilation(marker, structure=struc)
            marker_final = ndimage.label(marker)[0]

        elif structure == "ball":
            marker_final = make_points_ball(marker, size // 2)

        memb_distance = ndimage.distance_transform_edt(image)
        memb_distance_reverse = memb_distance.max() - memb_distance

        watershed_seg = watershed(memb_distance_reverse, marker_final.astype(np.uint16), watershed_line=True)
        watershed_seg = set_boundary_zero(watershed_seg)

        save_path = os.path.join(save_path, embryo_name_tp + "_segcell.nii.gz")
        nib.save(nib.Nifti1Image(watershed_seg.astype(np.uint16), np.eye(4)), save_path)

    except Exception as e:
        return "Threadpool return exception: {}".format(e)


def instance_segmentation_without_nucleus(para):
    try:
        [memb_files, save_path, tp, structure, size] = para

        embryo_name_tp = "_".join(os.path.basename(memb_files[tp]).split("_")[0:2])

        memb = nib.load(memb_files[tp]).get_fdata()

        if (len(np.unique(memb)) == 2):
            image = memb
        else:
            image = np.zeros_like(memb)
            image[memb > 200] = 1
        image = (image == 0).astype(np.uint16)

        image1 = ndimage.binary_opening(image).astype(np.uint16)
        image2 = ndimage.distance_transform_edt(image1)
        marker1 = h_maxima(image2, 2)

        if structure == "cube":
            struc = np.ones((size, size, size), dtype=bool)
            marker2 = ndimage.binary_dilation(marker1, structure=struc)
            marker_final = ndimage.label(marker2)[0]

        elif structure == "ball":
            marker1 = marker1.astype(np.uint16)
            marker_final = make_points_ball(marker1, size // 2)

        memb_distance = ndimage.distance_transform_edt(image)
        memb_distance_reverse = memb_distance.max() - memb_distance

        watershed_seg = watershed(memb_distance_reverse, marker_final.astype(np.uint16), watershed_line=True)
        watershed_seg = set_boundary_zero(watershed_seg)

        save_path = os.path.join(save_path, embryo_name_tp + "_segcell.nii.gz")
        nib.save(nib.Nifti1Image(watershed_seg.astype(np.uint16), np.eye(4)), save_path)

    except Exception as e:
        return "Threadpool return exception: {}".format(e)


def watershell_processing(binary_nuc):
    binary_nuc = binary_nuc.astype(int)
    distance = ndimage.distance_transform_edt(binary_nuc)
    max_coords = peak_local_max(distance, labels=binary_nuc, footprint=np.ones((3, 3, 3)))  # best dice score
    local_maxima = np.zeros_like(binary_nuc, dtype=bool)
    local_maxima[tuple(max_coords.T)] = True
    markers = ndimage.label(local_maxima)[0]
    labels = watershed(-distance, markers, mask=binary_nuc)

    # find center points in watershell
    unique_values = np.unique(labels)
    center_coords = []

    for value in unique_values:
        coords = np.array(np.where(labels == value)).T
        center_coord = np.mean(coords, axis=0)
        center_coords.append((int(center_coord[0]), int(center_coord[1]), int(center_coord[2])))

    int_mask = np.zeros_like(labels, dtype=np.uint8)

    for coord in center_coords:
        int_mask[coord[0], coord[1], coord[2]] = labels[coord[0], coord[1], coord[2]]

    return int_mask


def make_points_ball(marker, radius):
    label = 1

    x = np.array(list(range(marker.shape[0]))).reshape([marker.shape[0], 1, 1])

    y = np.array(list(range(marker.shape[1]))).reshape([1, marker.shape[1], 1])

    z = np.array(list(range(marker.shape[2]))).reshape([1, 1, marker.shape[2]])

    [maxima_x, maxima_y, maxima_z] = np.nonzero(marker)
    points = np.stack((maxima_x, maxima_y, maxima_z), axis=1)

    new_points = []
    new_points = points

    for actual_point in new_points:
        marker[
            (x - actual_point[0]) ** 2 + (y - actual_point[1]) ** 2 + (z - actual_point[2]) ** 2 <= radius ** 2] = label
        label += 1

    return marker


def set_boundary_zero(pre_seg):
    '''
    SET_BOUNARY_ZERO is used to set all segmented regions attached to the boundary as zero background.
    :param pre_seg:
    :return:
    '''
    opened_mask = binary_opening(pre_seg)
    pre_seg[opened_mask == 0] = 0
    seg_shape = pre_seg.shape
    boundary_mask = np.zeros_like(pre_seg, dtype=np.uint8)
    boundary_mask[0:2, :, :] = 1;
    boundary_mask[:, 0:2, :] = 1;
    boundary_mask[:, :, 0:2] = 1
    boundary_mask[seg_shape[0] - 1:, :, :] = 1;
    boundary_mask[:, seg_shape[1] - 1:, :] = 1;
    boundary_mask[:, :, seg_shape[2] - 1:] = 1
    boundary_labels = np.unique(pre_seg[boundary_mask != 0])
    for boundary_label in boundary_labels:
        pre_seg[pre_seg == boundary_label] = 0

    return pre_seg
