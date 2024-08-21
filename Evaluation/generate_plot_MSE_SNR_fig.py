import os
import warnings
import shutil

import seaborn as sns
import matplotlib.pyplot as plt
from Utils.data_io import nib_save, nib_load
from PIL import Image

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from skimage.morphology import ball

from tqdm import tqdm
import multiprocessing as mp
from skimage.transform import resize
from scipy import ndimage

from Utils.data_io import check_folder
from Utils.data_structure import read_cd_file


def generate_nuc_gt_from_cd_file(para):
    [embryo_name, name_dict, pd_lineage, tp, raw_size, out_size, save_folder] = para

    zoom_ratio = [y / x for x, y in zip(raw_size, out_size)]
    tp_lineage = pd_lineage[pd_lineage["time"] == tp]
    cell_number = len(tp_lineage)
    tp_lineage.loc[:, "x"] = (tp_lineage["x"] * zoom_ratio[0]).astype(np.int16)
    tp_lineage.loc[:, "y"] = (np.floor(tp_lineage["y"] * zoom_ratio[1])).astype(np.int16)
    # tp_lineage.loc[:, "z"] = np.floor(tp_lineage["z"] * (zoom_ratio[2] / dif_res)).astype(np.int16)

    tp_lineage.loc[:, "z"] = (out_size[2] - np.floor(tp_lineage["z"] * (out_size[-1] / raw_size[-1]))).astype(np.int16)

    tp_lineage.loc[:, 'radius'] = (tp_lineage["size"] * zoom_ratio[0] / 2).astype(np.int16)

    # !!!! x <--> y !!!!!!!
    nuc_dict = dict(
        zip(tp_lineage["cell"],
            zip(tp_lineage["y"].values, tp_lineage["x"].values, tp_lineage["z"].values, tp_lineage["radius"].values)))
    # labels = [name_dict[name] for name in list(nuc_dict.keys())]
    locs = list(nuc_dict.values())
    out_seg = np.zeros(out_size, dtype=np.int16)
    for cell_name in nuc_dict.keys():
        # tem_arr = np.zeros(out_size)
        # tem_arr[nuc_dict[cell_name][:3]]=1
        center_x, center_y, center_z = nuc_dict[cell_name][:3]
        # radius_this = int(nuc_dict[cell_name][3]) - int(cell_number / 150)
        radius_this = int(nuc_dict[cell_name][3])

        nucleus_marker_structure = ball(radius_this, dtype=bool)
        tem_arr_loc = np.full(out_size, False)
        tem_arr_loc[center_x - radius_this:center_x + radius_this + 1,
        center_y - radius_this:center_y + radius_this + 1,
        center_z - radius_this:center_z + radius_this + 1] = nucleus_marker_structure
        # struct = ndimage.generate_binary_structure(3, 1)
        # tem_arr = ndimage.morphology.grey_dilation(tem_arr, footprint=nucleus_marker_structure)
        cell_label = name_dict[cell_name]
        out_seg[tem_arr_loc] = cell_label
    # out_seg[tuple(zip(*locs))] = labels
    # out_seg = ndimage.morphology.grey_dilation(out_seg, footprint=nucleus_marker_footprint)
    # out_seg=ndimage.

    # out_seg=out_seg-1
    save_file_name = "_".join([embryo_name, str(tp).zfill(3), "gtNuc.nii.gz"])
    # nib_stack = nib.Nifti1Image(out_seg, np.eye(4))
    # nib_stack.header.set_xyzt_units(xyz=3, t=8)
    # nib_stack.header["pixdim"] = [1.0, out_res[0], out_res[1], out_res[2], 0., 0., 0., 0.]
    save_file = os.path.join(save_folder, save_file_name)
    # check_folder(save_file)
    nib_save(out_seg, save_file)


if __name__ == "__main__":
    # ===========================generate nucleus labeling gt========================================
    # embryo_names=['191108plc1p1','200326plc1p4']
    # max_times=[205,195]
    #
    # # embryo_names = ['191108plc1p1']
    # # max_times = [205]
    # raw_size=[512,712,92]
    # out_size=[256,356,214]
    # # dif_z=2*0.09/0.42
    # raw_img_folder=r'E:\ProjectData\MembraneProject\AllRawData'
    # saving_path=r'F:\packed membrane nucleus 3d niigz'
    # name_dictionary_path=r'D:\ImagesPrepocessor\necessary_files\name_dictionary_TUNETr.csv'
    #
    # mpPool = mp.Pool(mp.cpu_count()-1)
    # # mpPool = mp.Pool(1)
    #
    # for idx_embryo, embryo_name in enumerate(embryo_names):
    #
    #     # get lineage file
    #     lineage_file_path = os.path.join(raw_img_folder, embryo_name, "aceNuc",
    #                                          "CD{}.csv".format(embryo_name))
    #     assert lineage_file_path is not None
    #     target_folder = os.path.join(saving_path, embryo_name, "GTNuc")
    #     if not os.path.isdir(target_folder):
    #         os.makedirs(target_folder)
    #     pd_lineage = read_cd_file(lineage_file_path)
    #
    #     number_dictionary_path = name_dictionary_path
    #     label_name_dict = pd.read_csv(number_dictionary_path, index_col=0).to_dict()['0']
    #     name_label_dict = {value: key for key, value in label_name_dict.items()}
    #
    #     configs = []
    #     for tp in range(1, max_times[idx_embryo] + 1):
    #         configs.append((embryo_name, name_label_dict, pd_lineage, tp, raw_size, out_size, target_folder))
    #         # save_nuc_seg(configs[0])
    #     for idx, _ in enumerate(tqdm(mpPool.imap_unordered(generate_nuc_gt_from_cd_file, configs), total=len(configs),
    #                                  desc="3/3 Construct nucleus location of {}".format(embryo_name))):
    #         # TODO: Process Name: `3/3 Construct nucleus location`; Current status: `idx`; Final status: max_time
    #         pass
    # ===========================generate nucleus labeling gt========================================

    # =========================calculate nucleus image SNR====================================
    # df_SNR_nuc_enhancement = pd.DataFrame(
    #     columns=['Embryo Name', 'Time Point', 'Raw Signal Power', 'Raw Noise Power', 'Enhance Signal Power',
    #              'Enhanced Noise Power', 'Raw SNR', 'Enhanced SNR'])
    # slice_num = 92
    # size_3d = [256, 356, 214]
    #
    # embryo_names = ['191108plc1p1', '200326plc1p4']
    # max_times = [205, 195]
    #
    # packed_and_gt_nuc_path = r'F:\packed membrane nucleus 3d niigz'
    # enhanced_folder = r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\10 nucleus tracing enhancement\NucEnhancementForEvaluation'
    # raw_folder = r'E:\ProjectData\MembraneProject\AllRawData'
    #
    # for idx_emb, embryo_name in enumerate(embryo_names):
    #     for tp in range(1,max_times[idx_emb]+1):
    #         print(embryo_name,tp)
    #         out_stack = []
    #         for i_slice in range(1, slice_num + 1):
    #             # raw_file_name = "{}deconp1_L1-t{}-p{}.tif".format(embryo_name[:-2], str(tp).zfill(3), str(i_slice).zfill(2))
    #             raw_file_name = "{}_L1-t{}-p{}.tif".format(embryo_name, str(tp).zfill(3), str(i_slice).zfill(2))
    #
    #             img = np.asanyarray(Image.open(os.path.join(enhanced_folder,embryo_name,'tif', raw_file_name)))
    #             out_stack.insert(0, img)
    #         img_stack = np.transpose(np.stack(out_stack), axes=(1, 2, 0))
    #         enhanced_nuc_volume = resize(image=img_stack, output_shape=size_3d, preserve_range=True, order=1).astype(
    #             np.int16)
    #
    #         raw_nuc_volume = nib_load(os.path.join(packed_and_gt_nuc_path, embryo_name, 'RawNuc',
    #                                                '{}_{}_rawNuc.nii.gz'.format(embryo_name, str(tp).zfill(3))))
    #         gt_nuc_volume = nib_load(os.path.join(packed_and_gt_nuc_path, embryo_name, 'GTNuc',
    #                                               '{}_{}_gtNuc.nii.gz'.format(embryo_name, str(tp).zfill(3))))
    #
    #         enhanced_signal_power = enhanced_nuc_volume[gt_nuc_volume > 0].sum()
    #         enhanced_total_power = enhanced_nuc_volume.sum()
    #         enhanced_noise_power = enhanced_total_power - enhanced_signal_power
    #
    #         raw_signal_power = raw_nuc_volume[gt_nuc_volume > 0].sum()
    #         raw_total_power = raw_nuc_volume.sum()
    #         raw_noise_power = raw_total_power - raw_signal_power
    #         df_SNR_nuc_enhancement.loc[len(df_SNR_nuc_enhancement)] = [embryo_name, tp, raw_signal_power,
    #                                                                    raw_noise_power, enhanced_signal_power,
    #                                                                    enhanced_noise_power,
    #                                                                    raw_signal_power / raw_noise_power,
    #                                                                    enhanced_signal_power / enhanced_noise_power]
    # df_SNR_nuc_enhancement.to_csv('SNR_enhanced_nuc.csv')

    # reshaped_3d_volume=resize(image=raw_nuc_seg3d_volume,output_shape=target_shape,preserve_range=True,order=0).astype(np.uint8)
    #
    # embryo_naming={'191108plc1p1':'WT_SampleB', '200326plc1p4':'WT_SampleA'}
    # ONLYONE='WT_SampleB'
    # df_SNR_nuc_enhancement=pd.read_csv(r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\10 nucleus tracing enhancement\Figure1\SNR_enhanced_nuc.csv')
    # df_SNR_nuc_enhancement_plotting=pd.DataFrame(columns=['Sample','TP','SNR','Imaging'])
    # for index_snr in df_SNR_nuc_enhancement.index:
    #     this_info_list=df_SNR_nuc_enhancement.loc[index_snr]
    #     embryo_name=this_info_list[1]
    #     tp=this_info_list[2]
    #     SNR_raw=this_info_list[7]
    #     SNR_enhanced=this_info_list[8]
    #     df_SNR_nuc_enhancement_plotting.loc[len(df_SNR_nuc_enhancement_plotting)]=[embryo_naming[embryo_name],tp,SNR_enhanced,'Enhanced']
    #     df_SNR_nuc_enhancement_plotting.loc[len(df_SNR_nuc_enhancement_plotting)]=[embryo_naming[embryo_name],tp,SNR_raw,'Raw']
    # df_SNR_nuc_enhancement_plotting=df_SNR_nuc_enhancement_plotting.loc[df_SNR_nuc_enhancement_plotting['Sample']==ONLYONE]
    #
    # # hue_order_list = [
    # #     'WT_SampleB',
    # #     'WT_SampleA'
    # # ]
    # # hue_palette = {
    # #     'WT_SampleB': '#74fff8',
    # #     'WT_SampleA': '#e8000b'
    # # }
    #
    # # sns.lineplot(data=df_SNR_nuc_enhancement_plotting, x="TP", y='SNR', hue='Sample', style='Imaging',
    # #                    errorbar=('ci', 99),
    # #                    hue_order=hue_order_list, palette=hue_palette)
    # # plt.show()
    # # out.savefig(f'time_lapse_cell_loss.pdf', dpi=300)
    #
    #
    # hue_order_list = [
    #         'Raw',
    #         'Enhanced'
    #     ]
    # hue_palette = {
    #     'Raw': '#74fff8',
    #     'Enhanced': '#e8000b'
    # }
    # sns.lineplot(data=df_SNR_nuc_enhancement_plotting, x="TP", y='SNR', hue='Imaging',
    #              errorbar=('ci', 99),
    #              hue_order=hue_order_list, palette=hue_palette)
    #
    # plt.xticks([0,50,100,150,200], fontsize=16)
    #
    # plt.yticks(fontsize=16)
    #
    # plt.xlabel("Time Point", size=20)
    # plt.ylabel('Signal to Noise Ratio', size=20)
    # plt.savefig(ONLYONE+" SNR.pdf", format="pdf", dpi=300)
    # plt.show()
    # =========================calculate nucleus image SNR====================================

    # =========================calculate nucleus MSE via traced GT============================
    # embryo_naming={'191108plc1p1':'WT_Sample5', '200326plc1p4':'WT_Sample6'}
    #
    # # df_SNR_nuc_enhancement=pd.read_csv(r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\10 nucleus tracing enhancement\SNR_enhanced_nuc_v2.csv')
    # df_SNR_nuc_enhancement_plotting=pd.DataFrame(columns=['Sample','TP','MSE','Imaging'])
    # # df_SNR_nuc_enhancement = pd.DataFrame(
    # #     columns=['Embryo Name', 'Time Point', 'Raw MSE', 'Enhanced SNR'])
    # slice_num = 92
    # size_3d = [256, 356, 214]

    # embryo_names = ['191108plc1p1', '200326plc1p4']
    # max_times = [205, 195]
    #
    # packed_and_gt_nuc_path = r'F:\packed membrane nucleus 3d niigz'
    # enhanced_folder = r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\10 nucleus tracing enhancement\NucEnhancementForEvaluation'
    # raw_folder = r'E:\ProjectData\MembraneProject\AllRawData'
    #
    # for idx_emb, embryo_name in enumerate(embryo_names):
    #     for tp in range(1,max_times[idx_emb]+1):
    #         print(embryo_name,tp)
    #         out_stack = []
    #         for i_slice in range(1, slice_num + 1):
    #             # raw_file_name = "{}deconp1_L1-t{}-p{}.tif".format(embryo_name[:-2], str(tp).zfill(3), str(i_slice).zfill(2))
    #             raw_file_name = "{}_L1-t{}-p{}.tif".format(embryo_name, str(tp).zfill(3), str(i_slice).zfill(2))
    #
    #             img = np.asanyarray(Image.open(os.path.join(enhanced_folder,embryo_name,'tif', raw_file_name)))
    #             out_stack.insert(0, img)
    #         img_stack = np.transpose(np.stack(out_stack), axes=(1, 2, 0))
    #         enhanced_nuc_volume = resize(image=img_stack, output_shape=size_3d, preserve_range=True, order=1).astype(
    #             np.int16)
    #
    #         raw_nuc_volume = nib_load(os.path.join(packed_and_gt_nuc_path, embryo_name, 'RawNuc',
    #                                                '{}_{}_rawNuc.nii.gz'.format(embryo_name, str(tp).zfill(3))))
    #         gt_nuc_volume = nib_load(os.path.join(packed_and_gt_nuc_path, embryo_name, 'GTNuc',
    #                                               '{}_{}_gtNuc.nii.gz'.format(embryo_name, str(tp).zfill(3))))
    #         gt_nuc_volume[gt_nuc_volume>0]=255
    #
    #         enhanced_mse = np.square(enhanced_nuc_volume-gt_nuc_volume).mean()
    #         raw_mse = np.square(raw_nuc_volume-gt_nuc_volume).mean()
    #
    #         df_SNR_nuc_enhancement_plotting.loc[len(df_SNR_nuc_enhancement_plotting)] = [embryo_naming[embryo_name], tp, enhanced_mse,'Enhanced']
    #         df_SNR_nuc_enhancement_plotting.loc[len(df_SNR_nuc_enhancement_plotting)] = [embryo_naming[embryo_name], tp, raw_mse,'Raw']
    # df_SNR_nuc_enhancement_plotting.to_csv('MSE_enhanced_nuc.csv')

    df_SNR_nuc_enhancement_plotting=pd.read_csv('MSE_enhanced_nuc.csv',index_col=[0])
    embryo_naming = {'WT_Sample5': 'WT_SampleA', 'WT_Sample6': 'WT_SampleB'}
    df_SNR_nuc_enhancement_plotting=df_SNR_nuc_enhancement_plotting.replace({'Sample':embryo_naming})
    ONLYONE='WT_SampleB'

    df_SNR_nuc_enhancement_plotting=df_SNR_nuc_enhancement_plotting.loc[df_SNR_nuc_enhancement_plotting['Sample']==ONLYONE]



    # hue_order_list = [
    #     'WT_Sample5',
    #     'WT_Sample6']
    # hue_palette = {
    #     'WT_Sample5': '#74fff8',
    #     'WT_Sample6': '#e8000b'}
    # sns.lineplot(data=df_SNR_nuc_enhancement_plotting, x="TP", y='MSE', hue='Sample', style='Imaging',
    #                    errorbar=('ci', 99),
    #                    hue_order=hue_order_list, palette=hue_palette)

    hue_order_list = [
            'Raw',
            'Enhanced'
        ]
    hue_palette = {
        'Raw': '#74fff8',
        'Enhanced': '#e8000b'
    }
    sns.lineplot(data=df_SNR_nuc_enhancement_plotting, x="TP", y='MSE', hue='Imaging',
                 errorbar=('ci', 99),
                 hue_order=hue_order_list, palette=hue_palette)

    # plt.show()
    # out.savefig(f'time_lapse_cell_loss.pdf', dpi=300)
    plt.xticks([0,50,100,150,200], fontsize=16)

    plt.yticks(fontsize=16)

    plt.xlabel("Time Point", size=20)
    plt.ylabel('MSE (0~255 Grayscale)', size=20)
    plt.savefig(ONLYONE+" MSE.pdf", format="pdf", dpi=300)
    plt.show()
    # =========================calculate nucleus MSE via traced GT============================


