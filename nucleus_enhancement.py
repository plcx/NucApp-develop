from PyQt5.QtWidgets import (QWidget, QLabel, QLineEdit,
                             QTextEdit, QGridLayout, QApplication, QPushButton, QFileDialog, QMessageBox,
                             QComboBox, QVBoxLayout, QProgressBar, QHBoxLayout, QCheckBox)
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QMutex, QWaitCondition
from concurrent.futures import ThreadPoolExecutor
# from multiprocessing import cpu_count
from csbdeep.utils import normalize
from shutil import rmtree

import traceback
import os
import nibabel as nib
# from Utils.nuc_preprocess import nifti2tiff
# Other imports and code follow here
# import sys
import numpy as np
from glob import glob
from tqdm import tqdm
from tifffile import imread
import multiprocessing as mp
import logging
# from stardist import fill_label_holes, random_label_cmap, calculate_extents, gputools_available
from stardist import random_label_cmap

# from stardist import Rays_GoldenSpiral
# from stardist.matching import matching, matching_dataset
from stardist.models import StarDist3D
from Utils.niigz3d_to_tiff2d import seperate_3dniigz_to_2dtif
from Utils.preprocess_lib import stack_nuc_slices
# from Utils.segment_lib import segment_nuc

np.random.seed(42)
lbl_cmap = random_label_cmap()




class Nucleus_Enhancement(QWidget):

    def __init__(self):
        super().__init__()
        self.raw_tif = None
        self.mainlayout = QVBoxLayout()  # 组件的整体布局是垂直布局,第一层是一个栅格放参数,第二层是个水平布局放进度条, 最下层是一个反馈信息栏
        self.initUI()  # 设置参数相关的各种组件位置
        self.middlelayout = QGridLayout()
        self.runsegmentBtn = QPushButton("Run Enhancement")
        self.runsegmentBtn.clicked.connect(self.runEnhancement)
        self.cancelsegmentBtn = QPushButton("Cancel Enhancement")
        self.cancelsegmentBtn.setEnabled(False)
        self.cancelsegmentBtn.clicked.connect(self.cancelSegmentation)

        # self.pausesegmentBtn = QPushButton("Pause Enhancement")
        # self.pausesegmentBtn.setEnabled(False)
        # self.pausesegmentBtn.clicked.connect(self.pauseSegmentation)
        #
        # self.resumesegmentBtn = QPushButton("Resume Enhancement")
        # self.resumesegmentBtn.setEnabled(False)
        # self.resumesegmentBtn.clicked.connect(self.resumeSegmentation)

        self.segmentBar = QProgressBar()
        self.segmentBar.valueChanged.connect(self.completeSegmentation)
        self.middlelayout.addWidget(self.runsegmentBtn, 0, 1)
        self.middlelayout.addWidget(self.cancelsegmentBtn, 0, 2)
        # self.middlelayout.addWidget(self.pausesegmentBtn, 2, 1)
        # self.middlelayout.addWidget(self.resumesegmentBtn, 2, 2)
        self.middlelayout.addWidget(self.segmentBar, 0, 3)
        self.mainlayout.addStretch(1)
        self.mainlayout.addLayout(self.middlelayout)
        self.textEdit = QTextEdit()  # 初始化反馈信息栏
        self.textEdit.setFocusPolicy(Qt.NoFocus)  # 将反馈信息栏设置为无法主动编辑
        self.mainlayout.addStretch(1)  # 将反馈信息栏压到垂直布局的底层
        self.mainlayout.addWidget(self.textEdit)  # 将反馈信息栏添加到整体布局中
        self.setLayout(self.mainlayout)
        self.setGeometry(300, 300, 450, 500)
        self.show()

    def initUI(self):
        # 栅格布局第一列是参数名称
        raw_image_project_folder = QLabel('Raw Nucleus Image')

        embryoName = QLabel('Embryo Name')

        x_raw = QLabel('Raw X Size')
        y_raw = QLabel('Raw Y Size')
        z_raw = QLabel('Z Slices Num')

        x_resize = QLabel('3D X Size')
        y_resize = QLabel('3D Y Size')
        z_resize = QLabel('3D Z Size')

        saved_enhanced_image_project_folder = QLabel('Enhanced Nucleus Image')

        self.raw_image_root = QLineEdit()

        self.embryoNameEdit = QComboBox()
        raw_projectFolderBtn = QPushButton("Select")
        raw_projectFolderBtn.clicked.connect(self.chooseRawNucImageProjectFolder)

        self.x_rawEdit = QLineEdit()
        self.y_rawEdit = QLineEdit()
        self.z_rawEdit = QLineEdit()
        self.x_rawEdit.setText('512')
        self.y_rawEdit.setText('712')
        self.z_rawEdit.setText('92')

        self.x_resizeEdit = QLineEdit()
        self.y_resizeEdit = QLineEdit()
        self.z_resizeEdit = QLineEdit()
        self.x_resizeEdit.setText('256')
        self.y_resizeEdit.setText('356')
        self.z_resizeEdit.setText('214')

        self.saving_enhanced_image_root = QLineEdit()
        saving_projectFolderBtn = QPushButton("Select")
        saving_projectFolderBtn.clicked.connect(self.chooseSavingEnhancedProjectFolder)


        grid = QGridLayout()
        grid.setSpacing(30)
        grid.addWidget(raw_image_project_folder, 1, 0)
        grid.addWidget(self.raw_image_root, 1, 1)
        grid.addWidget(raw_projectFolderBtn, 1, 2)
        grid.addWidget(embryoName, 2, 0)
        grid.addWidget(self.embryoNameEdit, 2, 1)

        grid.addWidget(x_raw, 3, 0)
        grid.addWidget(self.x_rawEdit, 3, 1)
        grid.addWidget(y_raw, 4, 0)
        grid.addWidget(self.y_rawEdit, 4, 1)
        grid.addWidget(z_raw, 5, 0)
        grid.addWidget(self.z_rawEdit, 5, 1)

        grid.addWidget(x_resize, 6, 0)
        grid.addWidget(self.x_resizeEdit, 6, 1)
        grid.addWidget(y_resize, 7, 0)
        grid.addWidget(self.y_resizeEdit, 7, 1)
        grid.addWidget(z_resize, 8, 0)
        grid.addWidget(self.z_resizeEdit, 8, 1)

        grid.addWidget(saved_enhanced_image_project_folder, 9, 0)
        grid.addWidget(self.saving_enhanced_image_root, 9, 1)
        grid.addWidget(saving_projectFolderBtn, 9, 2)



        self.mainlayout.addLayout(grid)

    def chooseRawNucImageProjectFolder(self):
        dirName = QFileDialog.getExistingDirectory(self, 'Choose Raw Nucleus Image Folder', './')
        try:
            self.textEdit.clear()
            self.embryoNameEdit.clear()
            self.raw_image_root.setText(dirName)

            if dirName:
                listdir = [x for x in os.listdir(os.path.join(dirName)) if not x.startswith(".")]
                listdir.sort()
                self.embryoNameEdit.addItems(listdir)

        except Exception as e:
            self.textEdit.setText(traceback.format_exc())
            QMessageBox.warning(self, 'Warning!', 'Please Choose Right Folder!')

    def chooseSavingEnhancedProjectFolder(self):
        dirName = QFileDialog.getExistingDirectory(self, 'Choose Saving Image Folder', './')
        try:
            self.textEdit.clear()
            # self.embryoNameEdit.clear()
            self.saving_enhanced_image_root.setText(dirName)
            #
            # if dirName:
            #     listdir = [x for x in os.listdir(os.path.join(dirName)) if not x.startswith(".")]
            #     listdir.sort()
            #     self.embryoNameEdit.addItems(listdir)

        except Exception as e:
            self.textEdit.setText(traceback.format_exc())
            QMessageBox.warning(self, 'Warning!', 'Please Choose Right Folder!')



    def runEnhancement(self):
        para = {}
        try:
            para["raw_project_dir"] = self.raw_image_root.text()
            # para['raw_tif_dir'] = self.raw_tif
            para['embryo_name'] = self.embryoNameEdit.currentText()

            para['x_raw'] = int(self.x_rawEdit.text())
            para['y_raw'] = int(self.y_rawEdit.text())
            para['z_raw'] = int(self.z_rawEdit.text())

            para['x_resize'] = int(self.x_resizeEdit.text())
            para['y_resize'] = int(self.y_resizeEdit.text())
            para['z_resize'] = int(self.z_resizeEdit.text())

            para['save_project_dir'] = self.saving_enhanced_image_root.text()

            self.sthread = EnhancementThread(para)

        except:
            para.clear()
            self.textEdit.setText(traceback.format_exc())
            QMessageBox.warning(self, 'Error!', 'Initialization Failure!')

        if para:
            try:
                self.textEdit.clear()
                self.textEdit.append("Running Nucleus Enhancement!")
                self.textEdit.append(f"The embryo name is {para.get('embryo_name')}")
                self.textEdit.append(str(para))
                # self.textEdit.append(f"Raw tif path is {para.get('raw_tif_dir')}")
                # self.textEdit.append(f"X resize is {para.get('x_resize')}")
                # self.textEdit.append(f"Y resize is {para.get('y_resize')}")
                # self.textEdit.append(f"Z resize is {para.get('z_resize')}")
                self.runsegmentBtn.setEnabled(False)
                # self.resumesegmentBtn.setEnabled(False)
                self.cancelsegmentBtn.setEnabled(True)
                # self.pausesegmentBtn.setEnabled(False)
                self.segmentBar.reset()
                self.sthread.segmentbarSignal.connect(self.showsegmentbar)
                # self.sthread.segmentexcSignal.connect(self.segmentexc)
                self.sthread.start()

            except:
                self.textEdit.append(traceback.format_exc())
                QMessageBox.warning(self, 'Error!', 'Can not start Segmentation!')

    def cancelSegmentation(self):
        try:
            self.sthread.cancel()
            self.runsegmentBtn.setEnabled(True)
            # self.resumesegmentBtn.setEnabled(False)
            self.cancelsegmentBtn.setEnabled(False)
            # self.pausesegmentBtn.setEnabled(False)
            self.textEdit.setText("Segment Cancel!")
            self.segmentBar.setValue(0)
            QMessageBox.information(self, 'Tips', 'Segmentation has been terminated.')
        except Exception:
            self.textEdit.append(traceback.format_exc())
            QMessageBox.warning(self, 'Warning!', 'Segmentation cancel fail!.')

    # def pauseSegmentation(self):
    #     try:
    #         self.sthread.pause()
    #         self.runsegmentBtn.setEnabled(False)
    #         self.resumesegmentBtn.setEnabled(False)
    #         self.cancelsegmentBtn.setEnabled(True)
    #         self.pausesegmentBtn.setEnabled(False)
    #         self.textEdit.append("Segment Suspend!")
    #     except Exception:
    #         self.textEdit.append(traceback.format_exc())
    #         QMessageBox.warning(self, 'Warning!', 'Segment pause fail!.')

    # def resumeSegmentation(self):
    #     try:
    #         self.sthread.resume()
    #         self.runsegmentBtn.setEnabled(False)
    #         self.resumesegmentBtn.setEnabled(False)
    #         self.cancelsegmentBtn.setEnabled(True)
    #         self.pausesegmentBtn.setEnabled(False)
    #         self.textEdit.append("Segment Restart!")
    #     except Exception:
    #         self.textEdit.append(traceback.format_exc())
    #         QMessageBox.warning(self, 'Warning!', 'Segment resume fail!.')

    def completeSegmentation(self, value):
        if value == 100:
            self.textEdit.append("Enhancement Complete!")
            self.runsegmentBtn.setEnabled(True)
            # self.resumesegmentBtn.setEnabled(False)
            # self.pausesegmentBtn.setEnabled(False)
            self.cancelsegmentBtn.setEnabled(False)

    def showsegmentbar(self, current, total):
        self.segmentBar.setValue(int(current * 100 / total))

    def segmentexc(self, text):
        try:
            self.sthread.cancel()
            self.runsegmentBtn.setEnabled(True)
            # self.resumesegmentBtn.setEnabled(False)
            self.cancelsegmentBtn.setEnabled(False)
            # self.pausesegmentBtn.setEnabled(False)
            self.textEdit.setText(text)
            self.segmentBar.setValue(0)
            QMessageBox.warning(self, 'Error!', 'Errors with Segmentation!!.')
        except:
            QMessageBox.warning(self, 'Warning!', 'Segment cancel fail!.')





class EnhancementThread(QThread):
    segmentbarSignal = pyqtSignal(int, int)
    segmentexcSignal = pyqtSignal(str)


    def __init__(self, para):
        super().__init__()

        # para["raw_project_dir"] = self.raw_image_root.text()
        # # para['raw_tif_dir'] = self.raw_tif
        # para['embryo_name'] = self.embryoNameEdit.currentText()
        #
        # para['x_raw'] = int(self.x_resizeEdit.text())
        # para['y_raw'] = int(self.y_resizeEdit.text())
        # para['z_raw'] = int(self.z_resizeEdit.text())
        #
        # para['x_resize'] = int(self.x_resizeEdit.text())
        # para['y_resize'] = int(self.y_resizeEdit.text())
        # para['z_resize'] = int(self.z_resizeEdit.text())
        #
        # para['save_project_dir'] = self.embryoNameEdit.currentText()
        #project_dir, embryo_dir, flag
        self.raw_image_dir = para.get("raw_project_dir")
        # self.raw_tif_dir = para.get("raw_tif_dir")
        self.embryo_name = para.get("embryo_name")

        self.x_raw = para.get("x_raw")
        self.y_raw = para.get("y_raw")
        self.z_raw = para.get("z_raw")

        self.x_resize = para.get("x_resize")
        self.y_resize = para.get("y_resize")
        self.z_resize = para.get("z_resize")
        self.save_image_dir = para.get("save_project_dir")
        self.tem_3d_middle_folder=None

        self.isCancel = False
        # self.isPause = False
        # self.cond = QWaitCondition()
        # self.mutex = QMutex()
        #Load model



    def cancel(self):
        self.isCancel = True

    # def pause(self):
    #     self.isPause = True
    #
    # def resume(self):
    #     self.isPause = False
    #     self.cond.wakeAll()

    def run(self):
        try:
            self.combine_nucleus_slices()
            logging.debug('finished nucleus stack???')
            self.nuc_seg()

        except Exception:
            self.cancel()


    def combine_nucleus_slices(self):
        """
        Combine slices into stack images
        :param config: parameters
        :return:
        """
        # save nucleus
        self.segmentexcSignal.emit('Packing 3D for Enhancing')
        origin_files = sorted(glob(os.path.join(self.raw_image_dir, self.embryo_name, "tif", "*.tif")))
        self.max_time=int(len(origin_files)/self.z_raw)

        # origin_files.sort()
        self.tem_3d_middle_folder = os.path.join(self.save_image_dir, 'tem_middle',self.embryo_name, "RawNuc")
        if not os.path.isdir(self.tem_3d_middle_folder):
            os.makedirs(self.tem_3d_middle_folder)

        mpPool = mp.Pool(min(mp.cpu_count()//2,self.max_time))
        configs = []
        self.segmentbarSignal.emit(1, 10)

        for tp in range(1, self.max_time + 1):
            configs.append((origin_files[(tp - 1) * self.z_raw: tp * self.z_raw], self.tem_3d_middle_folder,
                    self.embryo_name, tp, [self.x_resize,self.y_resize,self.z_resize]))

        for idx, _ in enumerate(tqdm(mpPool.imap_unordered(stack_nuc_slices, configs), total=len(configs),
                                     desc="Stack 3D Nucleus of {}".format(self.embryo_name))):
            pass


    def nuc_seg(self):

        logging.debug('start running DL')

        if self.tem_3d_middle_folder == None:
            self.segmentexcSignal.emit('Please input RawFloder')
            return


        # Preprocessing
        input_dir = self.tem_3d_middle_folder

        if not os.path.exists(input_dir):
            self.segmentexcSignal.emit('Please generate the 3D tiff')
            return
        stack_dir_root=os.path.dirname(input_dir)

        enhance_dir = os.path.join(self.save_image_dir,  self.embryo_name, 'tif')

        # Stardist dataloader
        testset = sorted(glob(os.path.join(input_dir, '*.tif')))
        testset_copy = testset.copy()
        testset = list(map(imread, testset))
        self.segmentbarSignal.emit(2, 10)

        n_channel = 1 if testset[0].ndim == 3 else testset[0].shape[-1]
        axis_norm = (0, 1, 2)

        # Save prediction
        segNuc_dir_tem = os.path.join(stack_dir_root, 'SegNuc')
        if not os.path.exists(segNuc_dir_tem):
            os.makedirs(segNuc_dir_tem)

        logging.debug('start loading model')

        stardist_model = StarDist3D(None, name='stardist_nuc', basedir='static/models')
        logging.debug('finish loaded model')



        for i, path_this in enumerate(testset):
            # if self.isPause:
            #     self.cond.wait(self.mutex)
            logging.debug(str([self.isCancel,path_this,i,stardist_model]))
            if self.isCancel:
                break
            img = normalize(path_this, 1, 99.8, axis=axis_norm)
            labels, details = stardist_model.predict_instances(img)
            # Assuming testset and testset_copy have the same order and length
            embryo_tp = '_'.join(os.path.basename(testset_copy[i]).split('.')[0].split('_')[:2])
            pred_seg = labels.transpose([1, 2, 0])
            binary_pred = np.where(pred_seg > 0, 255, pred_seg)
            nib.save(nib.Nifti1Image(binary_pred, np.eye(4)),
                     os.path.join(segNuc_dir_tem, embryo_tp + '_predNuc.nii.gz'))
            self.segmentbarSignal.emit(i + 1, len(testset)*1.5)


        #Enhance nuc
        self.segmentbarSignal.emit(7, 10)

        # def nuc_enhance(raw_tiff_path, pred_niigz_path, saving_enhanced_tiff_path, embryo_name, max_time,
        #                 raw_tif_shape):

        mpPool = mp.Pool(min(mp.cpu_count()//2, self.max_time))
        configs = []
        for tp in range(1, self.max_time + 1):
            niigz3d_this_path = os.path.join(segNuc_dir_tem,
                                         '{}_{}_predNuc.nii.gz'.format(self.embryo_name, str(tp).zfill(3)))
            configs.append((niigz3d_this_path,enhance_dir,self.raw_image_dir,(self.x_raw, self.y_raw, self.z_raw)))

        for idx, _ in enumerate(tqdm(mpPool.imap_unordered(seperate_3dniigz_to_2dtif, configs), total=len(configs),
                                     desc="Enhancing Nucleus via DL of {}".format(self.embryo_name))):
            # self.segmentbarSignal.emit(9,10)
            pass

        # for tp in range(1, max_time + 1):
        #     niigz3d_this_path = os.path.join(pred_niigz_path,
        #                                      '{}_{}_predNuc.nii.gz'.format(embryo_name, str(tp).zfill(3)))
        #     seperate_3dniigz_to_2dtif(niigz3d_this_path, saving_enhanced_tiff_path, raw_tiff_path, raw_tif_shape)

        # nuc_enhance(self.raw_image_dir, segNuc_dir_tem, enhance_dir, self.embryo_name,len(testset), (self.x_resize, self.y_resize, self.z_resize))
        rmtree(stack_dir_root)
        self.segmentbarSignal.emit(10, 10)



