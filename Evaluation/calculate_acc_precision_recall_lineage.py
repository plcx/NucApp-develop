import seaborn as sns
import pandas as pd
import os

import matplotlib.pyplot as plt

from Utils.data_structure import read_cd_file

cd_file_path=r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Documents\07 nucleus tracing enhancement\Figure and materials\3linegea_precision_recall_evaluation'

embryo_names=['191108plc1p1','200113plc1p2','200326plc1p4']
max_times=[205,255,195]

for emb_idx,embryo_name in enumerate(embryo_names):
    the_gt_cd_file=read_cd_file(os.path.join(cd_file_path,'CD{}_edited.csv'.format(embryo_name)))
    the_raw_image_cd_file=read_cd_file(os.path.join(cd_file_path,'CD{}_raw.csv'.format(embryo_name)))
    the_enhanced_image_cd_file=read_cd_file(os.path.join(cd_file_path,'CD{}_enhanced.csv'.format(embryo_name)))

    print(the_gt_cd_file)
    pd_acc_precision_recall=pd.DataFrame(columns=['TP','Result','Imaging','Metrics'])
    for timepoint in range(1,max_times[emb_idx]+1):
        GT_cell_set_this_tp=set(the_gt_cd_file.loc[the_gt_cd_file['time']==timepoint]['cell'])
        rawimg_cell_set_this_tp=set(the_raw_image_cd_file.loc[the_raw_image_cd_file['time']==timepoint]['cell'])
        enhancedimg_cell_set_this_tp=set(the_enhanced_image_cd_file.loc[the_enhanced_image_cd_file['time']==timepoint]['cell'])

        raw_true_positive=GT_cell_set_this_tp.intersection(rawimg_cell_set_this_tp)
        raw_true_negative=0
        raw_false_positive=rawimg_cell_set_this_tp-GT_cell_set_this_tp
        raw_false_negative=GT_cell_set_this_tp-raw_true_positive

        if len(raw_true_positive)==0:
            accuracy_this=0
            precision=0
            recall=0
        else:
            accuracy_this = len(raw_true_positive)/(len(raw_true_positive)+len(raw_false_positive)+len(raw_false_negative))
            precision = len(raw_true_positive)/(len(raw_true_positive)+len(raw_false_positive))
            recall = len(raw_true_positive)/(len(raw_true_positive)+len(raw_false_negative))

        # print(embryo_name,timepoint,accuracy_this,precision,re)
        pd_acc_precision_recall.loc[len(pd_acc_precision_recall.index)]=[timepoint,accuracy_this,'Raw','Accuracy']
        pd_acc_precision_recall.loc[len(pd_acc_precision_recall.index)]=[timepoint,precision,'Raw','Precision']
        pd_acc_precision_recall.loc[len(pd_acc_precision_recall.index)]=[timepoint,recall,'Raw','Recall']

        # ====================================================================================
        enhanced_true_positive = GT_cell_set_this_tp.intersection(enhancedimg_cell_set_this_tp)
        enhanced_true_negative = 0
        enhanced_false_positive = enhancedimg_cell_set_this_tp - GT_cell_set_this_tp
        enhanced_false_negative = GT_cell_set_this_tp - enhanced_true_positive

        if len(enhanced_true_positive) == 0:
            accuracy_this = 0
            precision = 0
            recall = 0
        else:
            accuracy_this = len(enhanced_true_positive) / (
                        len(enhanced_true_positive) + len(enhanced_false_positive) + len(enhanced_false_negative))
            precision = len(enhanced_true_positive) / (len(enhanced_true_positive) + len(enhanced_false_positive))
            recall = len(enhanced_true_positive) / (len(enhanced_true_positive) + len(enhanced_false_negative))

        pd_acc_precision_recall.loc[len(pd_acc_precision_recall.index)] = [timepoint, precision, 'Enhanced', 'Precision']
        pd_acc_precision_recall.loc[len(pd_acc_precision_recall.index)] = [timepoint, recall, 'Enhanced', 'Recall']
        pd_acc_precision_recall.loc[len(pd_acc_precision_recall.index)] = [timepoint, accuracy_this, 'Enhanced', 'Accuracy']


    hue_order_list = [
            'Raw',
            'Enhanced'
        ]
    hue_palette = {
        'Raw': '#74fff8',
        'Enhanced': '#e8000b'
    }
    sns.lineplot(data=pd_acc_precision_recall, x="TP", y='Result', hue='Imaging',style='Metrics',
                 errorbar=('ci', 99),
                 hue_order=hue_order_list, palette=hue_palette)

    # plt.xticks([0,50,100,150,200], fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.xlabel("Time Point", size=20)
    plt.ylabel('Precision/Recall/Accuracy', size=20)
    plt.savefig(embryo_name+".pdf", format="pdf", dpi=300)
    pd_acc_precision_recall.to_csv(embryo_name+'.csv')
    plt.show()