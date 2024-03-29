from __future__ import print_function, unicode_literals, absolute_import, division

import traceback

# from PyQt5.QtWidgets import (QApplication, QGridLayout, QGroupBox, QDialog, QTableView, QListWidgetItem,
#                              QLabel, QSlider, QVBoxLayout, QMainWindow, QLineEdit, QListWidget,
#                              QMessageBox, QComboBox, QTableWidgetItem, QAbstractItemView, QCheckBox, QWidget,
from PyQt5.QtWidgets import (QApplication, QGridLayout,
                              QMainWindow, QLineEdit,
                             QMessageBox, QComboBox, QWidget,
                               QDesktopWidget, QMenu, QTabWidget,  QFileDialog)
# from PyQt5.QtGui import QStandardItemModel, QStandardItem, QTextCursor, QIcon, QFont, QDesktopServices
from PyQt5.QtGui import  QIcon

# from PyQt5.QtCore import Qt, QSize, QRect, QMetaObject, QCoreApplication, QUrl

from PyQt5.QtCore import Qt
from multiprocessing import freeze_support

import sys
import os
# import glob
# import shutil
# from stack_2d23d_for_enhancement import OldNucPreprocessing
from nucleus_enhancement import Nucleus_Enhancement




class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # menubar = self.menuBar()  # 初始化菜单栏
        # fileMenu = menubar.addMenu("File")  # 添加File选项到菜单栏
        # filebutton0 = QAction("Input Raw Folder", self)  # 添加Input raw folder按钮到File选项
        # filebutton0.setStatusTip("Input Raw Folder")
        # filebutton0.triggered.connect(self.inputRawFloder)
        # fileMenu.addAction(filebutton0)
        # filebutton1 = QAction("New Project Folder", self)  # 添加new_project按钮到File选项
        # filebutton1.setStatusTip("Create a new project")
        # filebutton1.triggered.connect(self.newProjectFolder)
        # fileMenu.addAction(filebutton1)
        #
        # runMenu = menubar.addMenu("Run")  # 添加Run选项到菜单栏
        # runbutton1 = QAction("Run Preprocess", self)  # 添加run preprocess按钮到Run选项
        # runbutton1.setStatusTip("Preprocess the files")
        # runMenu.addAction(runbutton1)
        # runbutton2 = QAction("Run Segmentation", self)  # 添加run segmentation按钮到Run选项
        # runbutton2.setStatusTip("Segment the cell")
        # runMenu.addAction(runbutton2)
        #
        # runbutton3 = QAction("Run Analysis", self)  # 添加run analysis按钮到Run选项
        # runbutton3.setStatusTip("Analyse the result")
        # runMenu.addAction(runbutton3)
        #
        # runbutton4 = QAction("Run All", self)  # 添加run all按钮到Run选项
        # runbutton4.setStatusTip("Run all programs")
        # runMenu.addAction(runbutton4)
        # runbutton5 = QAction("Open Result Folder", self)  # 添加open result按钮到Run选项
        # runbutton5.setStatusTip("Open the result folder")
        # runMenu.addAction(runbutton5)
        # aboutMenu = menubar.addMenu("About")  # 添加About选项到菜单栏
        # aboutbutton1 = QAction("Online Document", self)  # 添加open document按钮到About选项
        # aboutbutton1.setStatusTip("Open the document")
        # aboutMenu.addAction(aboutbutton1)
        # aboutbutton2 = QAction("About CTransformerApp", self)  # 添加open introduction按钮到About选项
        # aboutbutton2.setStatusTip("Brief introduction of CTransformerApp")
        # aboutMenu.addAction(aboutbutton2)
        # aboutbutton3 = QAction("License", self)  # 添加open license按钮到About选项
        # aboutbutton3.setStatusTip("Open the license")
        # aboutMenu.addAction(aboutbutton3)


        mainwidget = QWidget()  # 初始化主窗口的主组件
        mainlayout = QGridLayout()  # 初始化主组件的栅格布局
        mainwidget.setLayout(mainlayout)  # 将栅格布局应用到主组件上,可以在初始化时传入父类来简化代码
        self.project_path = os.path.join(os.getcwd(), 'CellAppData')
        self.functionbar = QTabWidget()  # 初始化选项菜单栏
        self.functionbar.setLayoutDirection(Qt.LeftToRight)
        self.functionbar.setTabBarAutoHide(False)
        # self.preprocess = OldNucPreprocessing()
        self.segmentation = Nucleus_Enhancement()
        # self.functionbar.addTab(self.preprocess, "Preprocess")
        self.functionbar.addTab(self.segmentation, "Nucleus Enhancement")
        # self.functionbar.setCurrentIndex(0) #这个函数可以通过下标设置初始化打开那个功能, 默认是第一个
        # self.functionbar.currentChanged.connect(self.updateBlankInfo) #功能卡的变化会调用updateBlankInfo这个函数
        mainlayout.addWidget(self.functionbar, 1, 0, 1, 1)  # 将选项菜单栏放入主组件的栅格布局中,位置为第1行第0列
        self.setCentralWidget(mainwidget)  # 将主组件放入主窗口
        self.resize(600, 720)  # 设置窗口大小
        self.setWindowTitle("NucEnhancingApp")  # 设置窗口标题
        self.setWindowIcon(QIcon("./static/img/NucEnhancingLogo.png"))  # 设置窗口小图标
        self.show()  # 展示主窗口

    def closeEvent(self, event):
        """
        关闭窗口提醒
        :param event:
        :return:
        """
        reply = QMessageBox.question(self, "quit confirm",
                                     "Are you sure to quit?", QMessageBox.Yes |
                                     QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    # def center(self):
    #     """
    #     窗口居中
    #     :return:
    #     """
    #     qr = self.frameGeometry()  # 获取主窗口框架
    #     cp = QDesktopWidget().availableGeometry().center()  # 获取屏幕中心
    #     qr.moveCenter(cp)  # 将框架中心移动到屏幕中心
    #     self.move(qr.topLeft())  # 将主窗口左上角与框架左上角对齐

    #
    # def contextMenuEvent(self, event):
    #     """
    #     右键菜单
    #     使用 exec_() 方法显示菜单。从鼠标右键事件对象中获得当前坐标。
    #     mapToGlobal() 方法把当前组件的相对坐标转换为窗口 (window) 的绝对坐标
    #     :param event:
    #     :return:
    #     """
    #     cmenu = QMenu(self)
    #
    #     runAct = cmenu.addAction("Run")
    #     clearAct = cmenu.addAction("Clear")
    #     action = cmenu.exec_(self.mapToGlobal(event.pos()))
    #
    #     if action == clearAct:
    #         if self.functionbar.currentIndex() == 0:
    #             for item in self.preprocess.findChildren((QLineEdit, QComboBox)):
    #                 if not item is self.preprocess.findChild(QComboBox, "preprocessObject"):
    #                     item.clear()
    #         if self.functionbar.currentIndex() == 1:
    #             for item in self.segmentation.findChildren((QLineEdit, QComboBox)):
    #                 item.clear()

    # def updateBlankInfo(self):
    #     """
    #     用来同步更新参数
    #     :return:
    #     """
    #     if self.functionbar.currentIndex() == 1: #当按钮点到segmentation
    #         if self.preprocess.raw_image_root.text():
    #             self.segmentation.raw_image_root.setText(self.preprocess.raw_image_root.text())
    #             self.segmentation.embryoNameEdit.clear()
    #             if os.path.isdir(os.path.join(self.preprocess.raw_image_root.text(), "RawStack")):
    #                 listdir = [x for x in os.listdir(os.path.join(self.preprocess.raw_image_root.text(), "RawStack")) if not x.startswith(".")]
    #                 listdir.sort()
    #                 self.segmentation.embryoNameEdit.addItems(listdir)
    #             else:
    #                 os.makedirs(os.path.join(self.preprocess.raw_image_root.text(), "RawStack"))

    #
    # def inputRawFloder(self):
    #     dirName = QFileDialog.getExistingDirectory(self, 'Choose Raw Folder', './')
    #     try:
    #         self.preprocess.textEdit.clear()
    #         self.preprocess.embryoNameBtn.clear()
    #         self.preprocess.rawFolderEdit.setText(dirName)
    #         if dirName:
    #             listdir = [x for x in os.listdir(dirName) if not x.startswith(".")]
    #             listdir.sort()
    #             self.preprocess.embryoNameBtn.addItems(listdir)
    #             self.segmentation.raw_tif = dirName
    #
    #     except Exception as e:
    #         self.textEdit.setText(traceback.format_exc())
    #         QMessageBox.warning(self, 'Warning!', 'Please Choose Right Folder!')
    #
    # def newProjectFolder(self):
    #     dirName = QFileDialog.getExistingDirectory(self, 'Choose Folder', './')
    #     project_path = os.path.join(dirName, 'CellAppData')
    #     try:
    #         if not os.path.exists(project_path):
    #             os.makedirs(project_path)
    #         self.segmentation.raw_image_root.setText(project_path)
    #         self.preprocess.raw_image_root.setText(project_path)
    #
    #     except Exception as e:
    #         errorMessage = f"{e}"
    #         QMessageBox.warning(self, 'Please choose project folder', errorMessage)


if __name__ == '__main__':
    freeze_support()
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
