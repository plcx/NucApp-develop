import sys
import os
import traceback
import glob
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool, cpu_count
from PIL import Image
import numpy as np
import re
from PyQt5.QtWidgets import (QWidget, QLabel, QLineEdit,
                             QTextEdit, QGridLayout, QApplication, QPushButton, QFileDialog, QMessageBox,
                             QComboBox, QVBoxLayout, QProgressBar, QHBoxLayout)
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QMutex, QWaitCondition
from Utils.preprocess_lib import stack_nuc_slices


class OldNucPreprocessing(QWidget):

    def __init__(self):
        super().__init__()
        self.mainlayout = QVBoxLayout()  # 组件的整体布局是垂直布局,第一层是一个栅格放参数,第二层是个水平布局放进度条, 最下层是一个反馈信息栏
        self.initUI()  # 设置参数相关的各种组件位置
        self.middlelayout = QGridLayout()
        self.runpreprocessBtn = QPushButton("Run")
        self.runpreprocessBtn.clicked.connect(self.runPreprocess)
        self.cancelpreprocessBtn = QPushButton("Cancel")
        self.cancelpreprocessBtn.setEnabled(False)
        self.cancelpreprocessBtn.clicked.connect(self.cancelPreprocess)
        self.pausepreprocessBtn = QPushButton("Pause")
        self.pausepreprocessBtn.setEnabled(False)
        self.pausepreprocessBtn.clicked.connect(self.pausePreprocess)
        self.resumepreprocessBtn = QPushButton("Resume")
        self.resumepreprocessBtn.setEnabled(False)
        self.resumepreprocessBtn.clicked.connect(self.resumePreprocess)
        self.preprocessBar = QProgressBar()
        self.preprocessBar.valueChanged.connect(self.completePreprocess)

        self.middlelayout.addWidget(self.runpreprocessBtn, 0, 1)
        self.middlelayout.addWidget(self.cancelpreprocessBtn, 0, 2)
        self.middlelayout.addWidget(self.pausepreprocessBtn, 2, 1)
        self.middlelayout.addWidget(self.resumepreprocessBtn, 2, 2)
        self.middlelayout.addWidget(self.preprocessBar, 1, 3)
        self.mainlayout.addStretch(1)
        self.mainlayout.addLayout(self.middlelayout)

        self.textEdit = QTextEdit()  # 初始化反馈信息栏
        self.textEdit.setFocusPolicy(Qt.NoFocus)  # 将反馈信息栏设置为无法主动编辑
        self.mainlayout.addStretch(1)  # 将反馈信息栏压到垂直布局的底层
        self.mainlayout.addWidget(self.textEdit)  # 将反馈信息栏添加到整体布局中

        self.setLayout(self.mainlayout)  # 将Preprocess这个分组件应用上设置好的整体布局
        self.setGeometry(300, 300, 450, 500)
        self.show()

    def initUI(self):
        # 栅格布局第一列是参数名称
        rawFolder = QLabel('Raw Folder')
        embryoName = QLabel('Embryo Name')
        xLength = QLabel('X Size')
        yLength = QLabel('Y Size')
        zLength = QLabel('Z Size')
        reduceRation = QLabel('Reduce Ratio')
        sliceNum = QLabel('Slice Num')
        maxTime = QLabel('Max Time')
        projectFolder = QLabel('Project Folder')
        preprocessObject = QLabel('Preprocess Object')
        # 栅格布局第二列是参数输入框
        self.rawFolderEdit = QLineEdit()
        self.xLengthEdit = QLineEdit()
        self.yLengthEdit = QLineEdit()
        self.zLengthEdit = QLineEdit()
        self.reduceRationEdit = QLineEdit()
        self.sliceNumEdit = QLineEdit()
        self.maxTimeEdit = QLineEdit()
        self.projectFolderEdit = QLineEdit()
        self.projectFolderEdit.setPlaceholderText("Save Path")
        # 栅格布局第三列是参数选择按钮
        rawFolderBtn = QPushButton("Select")
        rawFolderBtn.clicked.connect(self.chooseRawFolder)  # 将这个按钮点击事件与函数chooseRawFolder绑定
        self.embryoNameBtn = QComboBox()
        self.embryoNameBtn.activated[str].connect(self.Autofillblank)
        projectFolderBtn = QPushButton("Select")
        projectFolderBtn.clicked.connect(self.chooseProjectFolder)
        self.preprocessObjectBtn = QComboBox()
        self.preprocessObjectBtn.setObjectName("preprocessObject")  # 这里给这个复选框设置名字是为了在主窗口不清除这里的数据
        self.preprocessObjectBtn.addItem("Both")
        self.preprocessObjectBtn.addItem("Nucleus")
        self.preprocessObjectBtn.addItem("Membrane")


        grid = QGridLayout()
        grid.setSpacing(15)

        grid.addWidget(rawFolder, 1, 0)
        grid.addWidget(self.rawFolderEdit, 1, 1)
        grid.addWidget(rawFolderBtn, 1, 2)

        grid.addWidget(embryoName, 2, 0)
        grid.addWidget(self.embryoNameBtn, 2, 1)

        grid.addWidget(projectFolder, 3, 0)
        grid.addWidget(self.projectFolderEdit, 3, 1)
        grid.addWidget(projectFolderBtn, 3, 2)

        grid.addWidget(preprocessObject, 4, 0)
        grid.addWidget(self.preprocessObjectBtn, 4, 1)

        grid.addWidget(xLength, 5, 0)
        grid.addWidget(self.xLengthEdit, 5, 1)
        
        grid.addWidget(yLength, 6, 0)
        grid.addWidget(self.yLengthEdit, 6, 1)

        grid.addWidget(zLength, 7, 0)
        grid.addWidget(self.zLengthEdit, 7, 1)

        grid.addWidget(reduceRation, 8, 0)
        grid.addWidget(self.reduceRationEdit, 8, 1)

        grid.addWidget(sliceNum, 9, 0)
        grid.addWidget(self.sliceNumEdit, 9, 1)

        grid.addWidget(maxTime, 10, 0)
        grid.addWidget(self.maxTimeEdit, 10, 1)

        self.mainlayout.addLayout(grid)  # 将栅格布局添加到整体垂直布局的第一层

    def chooseRawFolder(self):
        """
        弹出一个路径选择,将选择的路径名称放到rawFolder,里面的内容放入embryonames
        :return:
        """
        dirName = QFileDialog.getExistingDirectory(self, 'Choose Raw Folder', './')
        try:
            self.textEdit.clear()
            self.embryoNameBtn.clear()
            self.rawFolderEdit.setText(dirName)
            if dirName:
                listdir = [x for x in os.listdir(dirName) if not x.startswith(".")]
                listdir.sort()
                self.embryoNameBtn.addItems(listdir)

        except Exception as e:
            self.textEdit.setText(traceback.format_exc())
            QMessageBox.warning(self, 'Warning!', 'Please Choose Right Folder!')

    def chooseProjectFolder(self):
        dirName = QFileDialog.getExistingDirectory(self, 'Choose Stack Folder', './')
        try:
            self.textEdit.clear()
            self.projectFolderEdit.setText(dirName)


        except Exception as e:
            self.textEdit.setText(traceback.format_exc())
            QMessageBox.warning(self, 'Warning!', 'Please Choose Right Folder!')

    def Autofillblank(self, embryo_name):
        try:
            raw_memb_files = glob.glob(os.path.join(self.rawFolderEdit.text(), embryo_name, "tifR", "*.tif"))
            raw_memb_img = raw_memb_files[-1]
            max_time = re.findall(r"\d{2,3}", raw_memb_img)[-2]
            num_slice = re.findall(r"\d{2,3}", raw_memb_img)[-1]

            self.maxTimeEdit.setText(max_time)
            self.sliceNumEdit.setText(num_slice)
            self.xLengthEdit.setText("256")
            self.yLengthEdit.setText("356")
            self.zLengthEdit.setText("160")
            self.reduceRationEdit.setText("1.0")
        except:
            self.textEdit.setText(traceback.format_exc())
            QMessageBox.warning(self, 'Error!', 'Please check your paras!')

    def runPreprocess(self):
        config = {}
        try:
            self.textEdit.clear()
            config['num_slice'] = int(self.sliceNumEdit.text())
            config["embryo_name"] = self.embryoNameBtn.currentText()
            config["max_time"] = int(self.maxTimeEdit.text())
            config["x_size"] = float(self.xLengthEdit.text())
            config["y_size"] = float(self.yLengthEdit.text())
            config["z_size"] = float(self.zLengthEdit.text())
            config["reduce_ratio"] = float(self.reduceRationEdit.text())
            config["raw_folder"] = self.rawFolderEdit.text()
            config["project_folder"] = self.projectFolderEdit.text()
            config["preprocess_object"] = self.preprocessObjectBtn.currentText()

        except Exception:
            config.clear()
            self.textEdit.append(traceback.format_exc())
            QMessageBox.warning(self, 'Error!', 'Please check your paras!')

        if config:
            try:
                self.textEdit.append('Running Preprocess!')
                for key, value in config.items():
                    self.textEdit.append(f"The {key} is: {value}")

                self.runpreprocessBtn.setEnabled(False)
                self.resumepreprocessBtn.setEnabled(False)
                self.cancelpreprocessBtn.setEnabled(True)
                self.pausepreprocessBtn.setEnabled(True)
                self.preprocessBar.reset()

                self.pthread = PreprocessThread(config)

                self.pthread.preprocessbarSignal.connect(self.showpreprocessbar)
                self.pthread.preprocessexcSignal.connect(self.preprocessexc)

                self.pthread.start()
            except:
                self.textEdit.append(traceback.format_exc())
                QMessageBox.warning(self, 'Error!', 'Can not start Preprocess!')

    def cancelPreprocess(self):
        try:
            self.pthread.cancel()
            self.runpreprocessBtn.setEnabled(True)
            self.resumepreprocessBtn.setEnabled(False)
            self.cancelpreprocessBtn.setEnabled(False)
            self.pausepreprocessBtn.setEnabled(False)
            self.textEdit.setText("Preprocess Cancel!")
            self.preprocessBar.setValue(0)
            QMessageBox.information(self, 'Tips', 'Preprocess has been terminated.')
        except Exception:
            self.textEdit.append(traceback.format_exc())
            QMessageBox.warning(self, 'Warning!', 'Preprocess cancel fail!.')

    def pausePreprocess(self):
        try:
            self.pthread.pause()
            self.runpreprocessBtn.setEnabled(False)
            self.resumepreprocessBtn.setEnabled(True)
            self.cancelpreprocessBtn.setEnabled(True)
            self.pausepreprocessBtn.setEnabled(False)
            self.textEdit.append("Preprocess Suspend!")
        except Exception:
            self.textEdit.append(traceback.format_exc())
            QMessageBox.warning(self, 'Warning!', 'Preprocess pause fail!.')

    def resumePreprocess(self):
        try:
            self.pthread.resume()
            self.runpreprocessBtn.setEnabled(False)
            self.resumepreprocessBtn.setEnabled(False)
            self.cancelpreprocessBtn.setEnabled(True)
            self.pausepreprocessBtn.setEnabled(True)
            self.textEdit.append("Preprocess Restart!")
        except Exception:
            self.textEdit.append(traceback.format_exc())
            QMessageBox.warning(self, 'Warning!', 'Preprocess resume fail!.')

    def completePreprocess(self, value):
        if value == 100:
            self.textEdit.append("Preprocess Complete!")
            self.runpreprocessBtn.setEnabled(True)
            self.cancelpreprocessBtn.setEnabled(False)
            self.pausepreprocessBtn.setEnabled(False)
            self.resumepreprocessBtn.setEnabled(False)

    def showpreprocessbar(self, current, total):
        self.preprocessBar.setValue(int(current * 100 / total))

    def preprocessexc(self, text):
        try:
            self.pthread.cancel()
            self.runpreprocessBtn.setEnabled(True)
            self.resumepreprocessBtn.setEnabled(False)
            self.cancelpreprocessBtn.setEnabled(False)
            self.pausepreprocessBtn.setEnabled(False)
            self.textEdit.setText(text)
            self.preprocessBar.setValue(0)
            QMessageBox.warning(self, 'Error!', 'Errors with Preprocess!!.')
        except:
            QMessageBox.warning(self, 'Warning!', 'Preprocess cancel fail!.')


"""QT 中 QObject 作QT中类的最终父类，具有自定义信号与槽的能力，只要继承自这个类的类，也一样拥有自定义信号和槽的能力。
QT 中定义信号与槽是十分有用的，QT 下多线程类QThread 是继承自 QObject，同样具有有自定义信号和槽的能力"""


class PreprocessThread(QThread):
    preprocessbarSignal = pyqtSignal(int, int)
    preprocessexcSignal = pyqtSignal(str)

    def __init__(self, config={}):
        super().__init__()
        # 从配置中获取参数
        self.num_slice = config["num_slice"]
        self.embryo_name = config["embryo_name"]
        self.max_time = config["max_time"]
        self.x_size = config["x_size"]
        self.y_size = config["y_size"]
        self.z_size = config["z_size"]
        self.reduce_ratio = config["reduce_ratio"]
        self.raw_folder = config["raw_folder"]
        self.stack_folder = os.path.join(config["project_folder"], "RawStack")
        self.preprocess_object = config['preprocess_object']

        # 计算相关参数
        raw_memb_files = glob.glob(os.path.join(self.raw_folder, self.embryo_name, "tifR", "*.tif"))
        self.raw_size = list(np.asarray(Image.open(raw_memb_files[0])).shape) + [int(self.num_slice)]
        self.out_size = [int(i * self.reduce_ratio) for i in [self.x_size, self.y_size, self.z_size]]
        #self.out_res = [x / y for x, y in zip(self.out_size, self.raw_size)]
        self.out_res = [1, 1, 1]

        # 与线程运行有关
        self.isCancel = False
        self.isPause = False
        self.cond = QWaitCondition()
        self.mutex = QMutex()

    def cancel(self):
        self.isCancel = True

    def pause(self):
        self.isPause = True

    def resume(self):
        self.isPause = False
        self.cond.wakeAll()

    def run(self):
        try:
            if self.preprocess_object == "Nucleus":
                self.combine_nucleus_slices()
            elif self.preprocess_object == "Membrane":
                self.combine_memb_slices()
            else:
                self.combine_both_slices()

        except Exception:
            self.quit()

    def combine_nucleus_slices(self):
        """
        Combine slices into stack images
        :param config: parameters
        :return:
        """
        # save nucleus
        origin_files = glob.glob(os.path.join(self.raw_folder, self.embryo_name, "tif", "*.tif"))
        origin_files.sort()
        target_folder = os.path.join(self.stack_folder, self.embryo_name, "RawNuc")
        if not os.path.isdir(target_folder):
            os.makedirs(target_folder)

        with ThreadPoolExecutor(cpu_count() + 1) as t:
            for tp in range(1, self.max_time + 1):
                self.mutex.lock()
                if self.isPause:
                    self.cond.wait(self.mutex)
                if self.isCancel:
                    break
                configs = (
                    origin_files, target_folder, self.embryo_name, tp, self.out_size, self.num_slice, self.out_res)
                exception = t.submit(stack_nuc_slices, configs).result()
                self.preprocessbarSignal.emit(tp, self.max_time)
                if exception:
                    self.preprocessexcSignal.emit(exception)
                self.mutex.unlock()

    def combine_memb_slices(self):
        """
        Combine slices into stack images
        :param config: parameters
        :return:
        """
        # save nucleus
        origin_files = glob.glob(os.path.join(self.raw_folder, self.embryo_name, "tifR", "*.tif"))
        origin_files.sort()
        target_folder = os.path.join(self.stack_folder, self.embryo_name, "RawMemb")
        if not os.path.isdir(target_folder):
            os.makedirs(target_folder)

        with ThreadPoolExecutor(cpu_count() + 1) as t:
            for tp in range(1, self.max_time + 1):
                self.mutex.lock()
                if self.isPause:
                    self.cond.wait(self.mutex)
                if self.isCancel:
                    break
                configs = (
                    origin_files, target_folder, self.embryo_name, tp, self.out_size, self.num_slice, self.out_res)
                exception = t.submit(stack_memb_slices, configs).result()
                self.preprocessbarSignal.emit(tp, self.max_time)
                if exception:
                    self.preprocessexcSignal.emit(exception)
                self.mutex.unlock()

    def combine_both_slices(self):
        """
        Combine slices into stack images
        :param config: parameters
        :return:
        """
        origin_files1 = glob.glob(os.path.join(self.raw_folder, self.embryo_name, "tifR", "*.tif"))
        origin_files1.sort()
        target_folder1 = os.path.join(self.stack_folder, self.embryo_name, "RawMemb")

        origin_files2 = glob.glob(os.path.join(self.raw_folder, self.embryo_name, "tif", "*.tif"))
        origin_files2.sort()
        target_folder2 = os.path.join(self.stack_folder, self.embryo_name, "RawNuc")
        target_folder_tif = os.path.join(self.stack_folder, self.embryo_name, "RawNucTif")

        if not os.path.isdir(target_folder1):
            os.makedirs(target_folder1)

        if not os.path.isdir(target_folder2):
            os.makedirs(target_folder2)

        with ThreadPoolExecutor(cpu_count() + 1) as t:
            for tp in range(1, self.max_time + 1):
                self.mutex.lock()
                if self.isPause:
                    self.cond.wait(self.mutex)
                if self.isCancel:
                    break
                configs1 = (
                    origin_files1, target_folder1, self.embryo_name, tp, self.out_size, self.num_slice, self.out_res)
                configs2 = (
                    origin_files2, target_folder2, self.embryo_name, tp, self.out_size, self.num_slice, self.out_res)
                exception1 = t.submit(stack_memb_slices, configs1).result()
                self.preprocessbarSignal.emit(2 * tp - 1, self.max_time * 2)
                exception2 = t.submit(stack_nuc_slices, configs2).result()
                self.preprocessbarSignal.emit(2 * tp, self.max_time * 2)
                if exception1:
                    self.preprocessexcSignal.emit(exception1)
                elif exception2:
                    self.preprocessexcSignal.emit(exception2)
                self.mutex.unlock()

'''
if __name__ == '__main__':
    """单独看这个组件"""
    app = QApplication(sys.argv)
    ex = Preprocess()
    sys.exit(app.exec_())
'''