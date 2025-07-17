# # 功能：数据选择、模型训练、模型预测、重构误差展示
import re
import sys
import os
import csv
from datetime import datetime
import time
# import typing
from PyQt5 import QtCore
import numpy as np
import pandas as pd
# import seaborn as sns
# import pylab as plt
from PyQt5.QtCore import QAbstractTableModel, Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog, QMainWindow, QWidget, QVBoxLayout, QGridLayout,QMessageBox,QInputDialog,QLineEdit
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from pylab import mpl
import numpy as np
from numpy import array, sign, zeros
from scipy.interpolate import interp1d
import scipy.signal
mpl.rcParams["font.sans-serif"] = ["SimHei"]
mpl.rcParams["axes.unicode_minus"] = False  # 设置正常显示符号
import pyqtgraph as pg
import pyqtgraph.exporters
from minepy import MINE
import torch
import argparse
from exp.exp_informer_pyqt import Exp_Informer
from informer_ui0407 import *
from utils.tools import dotdict

class Ctest_readscv(QMainWindow, Ui_MainWindow): # 父类QMainWindow与Ui_MainWindow
    def __init__(self, parent=None): # 可输入parent
        super(Ctest_readscv, self).__init__(parent) # 继承父类初始化
        self.setupUi(self)
        self.setWindowTitle("卫星异常检测测试软件_HIT")
        self.model = QtGui.QStandardItemModel(self) # 表格实例
        self.tableView.setModel(self.model) # 表格视图，不是控件
        self.model.dataChanged.connect(self.finishedEdit)
        self.selectionModel = self.tableView.selectionModel()
        self.pushButton_csv.clicked.connect(self.read_csv)
        self.pushButton_all_value.clicked.connect(self.all_value)
        self.pushButton_no_value.clicked.connect(self.no_value)
        self.pushButton_choose.clicked.connect(self.df_choose)
        self.pushButton_choose_out.clicked.connect(self.output_choose)
        self.pushButton_pth.clicked.connect(self.readpth)
        self.pushButton_showdataset.clicked.connect(self.datasetshow)
        self.pushButton_train.clicked.connect(self.train_ui)
        self.pushButton_pred.clicked.connect(self.pred_show)
        self.pushButton_pred_2.clicked.connect(self.error_show)
        self.pushButton_allerror.clicked.connect(self.allerror)
        self.pushButton_alarm.clicked.connect(self.alarm)
        self.comboBox_alarm_value.currentIndexChanged.connect(
            lambda: self.WrittingNotOfOther(self.comboBox_alarm_value.currentIndex()))  # 点击下拉列表，触发对应事件
        self.comboBox_label.currentIndexChanged.connect(
            lambda: self.WrittingNotOfOther_label(self.comboBox_label.currentIndex()))  # 点击下拉列表，触发对应事件
        

        # tab_1布局
        # 可以在手动布局的基础上继续设定，优先级更高
        self.tab_1_layout = QGridLayout()
        self.tab_1.setLayout(self.tab_1_layout)
        self.tab_1_layout.addWidget(self.pushButton_csv, 0, 0, 1, 2) # 左上角纵坐标、横坐标、上下宽度、左右宽度
        self.tab_1_layout.addWidget(self.pushButton_all_value,0,2,1,2) # 全选按键
        self.tab_1_layout.addWidget(self.pushButton_no_value,0,4,1,2) # 全选按键
        self.tab_1_layout.addWidget(self.label, 0, 6, 1, 6) # 布局由addwidget最终决定
        self.tab_1_layout.addWidget(self.tableView, 1, 0, 7, 12)
        self.tab_1_layout.addWidget(self.pushButton_choose, 8, 0, 1, 2)
        self.tab_1_layout.addWidget(self.pushButton_choose_out, 8, 2, 1, 2)
        self.tab_1_layout.addWidget(self.label_22, 8, 4, 1, 8)

        # 为tab2添加画布
        self.win = pg.GraphicsLayoutWidget(title='数据集划分展示')
        self.win.setBackground("w")
        # tab_2 布局
        self.tab_2_layout = QGridLayout()
        self.tab_2.setLayout(self.tab_2_layout)
        self.tab_2_layout.addWidget(self.label_2,0,0,1,2)
        # self.tab_2_layout.addWidget(self.verticalLayout_4,3,0,3,2)
        # self.tab_2_layout.addWidget(self.verticalLayout,3,2,3,2)
        self.tab_2_layout.addWidget(self.comboBox_model,0,2,1,4)
        self.tab_2_layout.addWidget(self.label_17, 0, 6, 1, 2)
        self.tab_2_layout.addWidget(self.comboBox_lossfc, 0, 8, 1, 2)
        model_name = ['informer','transformer','transformerXL','autoformer']
        for name in model_name:
            self.comboBox_model.addItem(name) # 提供可选择模型
        
        self.tab_2_layout.addWidget(self.pushButton_pth,1, 0, 1, 2)
        self.tab_2_layout.addWidget(self.label_21,1, 2, 1, 16)
        self.tab_2_layout.addWidget(self.label_3,2,0,1,2)
        self.tab_2_layout.addWidget(self.label_4,3,0,1,2)
        self.tab_2_layout.addWidget(self.label_5,4,0,1,2)
        self.tab_2_layout.addWidget(self.lineEdit_trainstarttime, 2, 2, 1, 2) # lineedit既可以显示文字也可以获取文字，而qlabel只能显示内容
        self.tab_2_layout.addWidget(self.lineEdit_validstarttime, 3, 2, 1, 2)
        self.tab_2_layout.addWidget(self.lineEdit_teststarttime, 4, 2, 1, 2)
        self.tab_2_layout.addWidget(self.lineEdit_trainendtime, 2, 5, 1, 2)
        self.tab_2_layout.addWidget(self.lineEdit_validendtime, 3, 5, 1, 2)
        self.tab_2_layout.addWidget(self.lineEdit_testendtime, 4, 5, 1, 2)
        self.tab_2_layout.addWidget(self.label_12, 2, 4, 1, 1)
        self.tab_2_layout.addWidget(self.label_11, 3, 4, 1, 1)
        self.tab_2_layout.addWidget(self.label_10, 4, 4, 1, 1)
        self.tab_2_layout.addWidget(self.label_6, 2, 8, 1, 2)
        self.tab_2_layout.addWidget(self.label_7, 3, 8, 1, 2)
        self.tab_2_layout.addWidget(self.label_8, 4, 8, 1, 2)
        self.tab_2_layout.addWidget(self.lineEdit_epoch, 2, 10, 1, 2)
        self.tab_2_layout.addWidget(self.lineEdit_batchsize, 3, 10, 1, 2)
        self.tab_2_layout.addWidget(self.lineEdit_rate, 4, 10, 1, 2)
        self.tab_2_layout.addWidget(self.label_18, 2, 13, 1, 2)
        self.tab_2_layout.addWidget(self.label_19, 3, 13, 1, 2)
        self.tab_2_layout.addWidget(self.label_20, 4, 13, 1, 2)
        self.tab_2_layout.addWidget(self.lineEdit_inputlength, 2, 15, 1, 2)
        self.tab_2_layout.addWidget(self.lineEdit_predlength, 3, 15, 1, 2)
        self.tab_2_layout.addWidget(self.lineEdit_labellength, 4, 15, 1, 2)
        self.tab_2_layout.addWidget(self.pushButton_showdataset,5,0,1,2)
        self.tab_2_layout.addWidget(self.comboBox_model_2,5,2,1,4)
        self.tab_2_layout.addWidget(self.label_dataset,5,6,1,12)
        self.tab_2_layout.addWidget(self.win,6,0,4,18)
        self.tab_2_layout.addWidget(self.pushButton_train,11,0,1,2)
        self.tab_2_layout.addWidget(self.plainTextEdit,12,0,4,18)
        
        # 添加画布
        self.win_show = pg.GraphicsLayoutWidget(title='预测展示')
        self.win_show.setBackground("w")
        # tab_3 布局
        self.tab_3_layout = QGridLayout()
        self.tab_3.setLayout(self.tab_3_layout)
        self.tab_3_layout.addWidget(self.label_9, 0, 0, 1, 1)
        self.tab_3_layout.addWidget(self.lineEdit_showstarttime, 0, 1, 1, 1)
        self.tab_3_layout.addWidget(self.label_13, 0, 2, 1, 1)
        self.tab_3_layout.addWidget(self.lineEdit_showendtime, 0, 3, 1, 1)
        self.tab_3_layout.addWidget(self.label_pred_time, 0, 4, 1, 14)
        self.tab_3_layout.addWidget(self.pushButton_pred, 1, 0, 1, 1)
        self.tab_3_layout.addWidget(self.comboBox_pred,1,1,1,1)
        self.tab_3_layout.addWidget(self.label_23,1,2,1,1)
        self.tab_3_layout.addWidget(self.comboBox_stepshow,1,3,1,1)
        self.tab_3_layout.addWidget(self.label_pred_show, 1, 4, 1, 16)
        self.tab_3_layout.addWidget(self.win_show,2,0,6,18)
        self.tab_3_layout.addWidget(self.label_26,9,0,1,1)
        self.tab_3_layout.addWidget(self.label_pred_path, 9, 1, 1, 16)

        # # 添加画布
        self.win_error = pg.GraphicsLayoutWidget(title='重构误差展示')
        self.win_error.setBackground("w")
        # tab_4 布局
        self.tab_4_layout = QGridLayout()
        self.tab_4.setLayout(self.tab_4_layout)
        self.tab_4_layout.addWidget(self.label_14, 0, 0, 1, 1)
        self.tab_4_layout.addWidget(self.lineEdit_errorshowstarttime, 0, 1, 1, 1)
        self.tab_4_layout.addWidget(self.label_15, 0, 2, 1, 1)
        self.tab_4_layout.addWidget(self.lineEdit_errorshowendtime, 0, 3, 1, 1)
        self.tab_4_layout.addWidget(self.label_16, 0, 4, 1, 1)
        self.tab_4_layout.addWidget(self.comboBox_error, 0, 5, 1, 1)
        self.tab_4_layout.addWidget(self.pushButton_pred_2, 1, 0, 1, 1)
        self.tab_4_layout.addWidget(self.comboBox_errorshow, 1, 1, 1, 1)
        self.tab_4_layout.addWidget(self.label_24, 1, 2, 1, 1)
        self.tab_4_layout.addWidget(self.comboBox_stepshow_2,1,3,1,1)
        self.tab_4_layout.addWidget(self.label_25, 1, 4, 1, 1)
        self.tab_4_layout.addWidget(self.lineEdit_threshold,1,5,1,1)
        self.tab_4_layout.addWidget(self.label_thresholderror,1,6,1,12)
        self.tab_4_layout.addWidget(self.win_error, 2, 0, 6, 18)

        # # 添加画布
        self.win_allerror = pg.GraphicsLayoutWidget(title='总误差展示')
        self.win_allerror.setBackground("w")
        # tab_5 布局
        self.tab_5_layout = QGridLayout()
        self.tab_5.setLayout(self.tab_5_layout)
        self.tab_5_layout.addWidget(self.label_27, 0, 0, 1, 1)
        self.tab_5_layout.addWidget(self.lineEdit_allerrorshowstarttime, 0, 1, 1, 1)
        self.tab_5_layout.addWidget(self.label_28, 0, 2, 1, 1)
        self.tab_5_layout.addWidget(self.lineEdit_allerrorshowendtime, 0, 3, 1, 1)
        self.tab_5_layout.addWidget(self.label_29, 0, 4, 1, 1)
        self.tab_5_layout.addWidget(self.comboBox_allerror, 0, 5, 1, 1)
        self.tab_5_layout.addWidget(self.pushButton_allerror, 1, 0, 1, 1)
        self.tab_5_layout.addWidget(self.label_30, 1, 2, 1, 1)
        self.tab_5_layout.addWidget(self.comboBox_stepshow_3,1,3,1,1)
        self.tab_5_layout.addWidget(self.label_31, 1, 4, 1, 1)
        self.tab_5_layout.addWidget(self.lineEdit_allerrorthreshold,1,5,1,1)
        self.tab_5_layout.addWidget(self.label_thresholdallerror,1,6,1,12)
        self.tab_5_layout.addWidget(self.win_allerror, 2, 0, 6, 18)

        # # 添加画布
        self.win_alarm = pg.GraphicsLayoutWidget(title='异常报警')
        self.win_alarm.setBackground("w")
        # tab_6 布局
        self.tab_6_layout = QGridLayout()
        self.tab_6.setLayout(self.tab_6_layout)
        self.tab_6_layout.addWidget(self.comboBox_label, 0, 0, 1, 1)
        self.tab_6_layout.addWidget(self.lineEdit_alarm, 0, 1, 1, 1)
        self.tab_6_layout.addWidget(self.label_33, 0, 2, 1, 1)
        self.tab_6_layout.addWidget(self.lineEdit_alarm_time, 0, 3, 1, 1)
        self.tab_6_layout.addWidget(self.label_alarm, 0, 4, 1, 14)
        self.tab_6_layout.addWidget(self.label_34, 1, 0, 1, 1)
        self.tab_6_layout.addWidget(self.comboBox_alarm,1,1,1,1)
        self.tab_6_layout.addWidget(self.label_35,1,2,1,1)
        self.tab_6_layout.addWidget(self.comboBox_alarm_stepshow,1,3,1,1)
        self.tab_6_layout.addWidget(self.comboBox_alarm_value,2,1,1,1)
        self.tab_6_layout.addWidget(self.pushButton_alarm, 2, 0, 1, 1)
        self.tab_6_layout.addWidget(self.win_alarm,3,0,5,18)
        self.tab_6_layout.addWidget(self.plainTextEdit_alarm,8,0,4,18)

        self.label.setText('请选择gbk编码格式文件')
        self.fileName = None # 数据集文件路径
        self.time_col = None # 数据时间列
        self.dic = None # 选取的数据字典（无时间列）
        self.newfilename = None # 手动选择后的文件路径
        self.head_name = [] # 选取工程值列名
        self.all_choose = False
        self.data_df = None # 选取数据dataframe（有时间列）
        self.data_df_bool = None # 读取dataframe布尔值
        self.trainstarttime = None # 训练开始时间
        self.trainendtime = None # 训练结束时间
        self.validstarttime = None # 训练开始时间
        self.validendtime = None # 训练结束时间
        self.teststarttime = None # 训练开始时间
        self.testendtime = None # 训练结束时间
        self.showstarttime = None # 展示开始时间
        self.showendtime = None # 展示结束时间
        self.errorshowstarttime = None # 展示开始时间
        self.errorshowendtime = None # 展示结束时间
        self.allerrorshowstarttime = None # 总误差展示开始时间
        self.allerrorshowendtime = None # 总误差展示结束时间
        self.epoch = 5 # 默认训练轮次
        self.batch_size = 32 # 默认训练批次
        self.learning_rate = 0.0001
        self.threshold = None
        self.allthreshold = None
        self.checkpoints = None
        self.firstdataset = True
        self.alarm_threshold = 1.5 # 报警阈值
        self.alarm_time_threshold = 20 # 异常报警时间阈值
        self.multi_pred = True # 多对多预测布尔值，当不决定输出数据时，默认为True
        self.output_columns_list = [] # 输出少值时，选择出少值的列名
        self.output_index_list = [] # 输出少值时，选择出少值的列序号


        self.seq_len = 96 # 默认输入步长
        self.label_len = 24 # 默认重合步长
        self.pred_len = 48 # 默认预测步长
        self.show_step = 1 # 默认展示步长（1--pred_len）

        self.datasetshow_plot = None
        self.modelpath = None # 若不输入预训练模型，则模型从头开始训练
        self.modelname = None
        self.setting = None
        self.show_plot = None
        self.errorshow_plot = None
        self.allerrorshow_plot = None
        self.alarm_plot = None

        # 上下四分位数
        self.midpoint_dic = {}

        self.error1 = "!!!未选择文件!!!"
        self.error2 = "!!!请选择正确的模型文件!!!"
        self.error3 = "!!!请确保保存pth文件夹名称是规范的格式，否则无法获取模型信息!!!"
        self.error4 = "!!!请选择测试集范围内时间!!!"
        self.error5 = "!!!请输入正确数字!!!"
        self.error6 = "!!!请选择数据!!!"

    # 判断数值或者字符串是否是数值
    def is_number(self,s):
        try:  # 如果能运行float(s)语句，返回True（字符串s是浮点数）
            float(s)
            return True
        except ValueError:  # ValueError为Python的一种标准异常，表示"传入无效的参数"
            pass  # 如果引发了ValueError这种异常，不做任何事情（pass：不做任何事情，一般用做占位语句）
        try:
            import unicodedata  # 处理ASCii码的包
            unicodedata.numeric(s)  # 把一个表示数字的字符串转换为浮点数返回的函数
            return True
        except (TypeError, ValueError):
            pass
        return False
    # 读取文件
    def read_csv(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open CSV", "*.csv")
        if fileName:
            print(fileName)
            self.comboBox_lossfc.clear()
            self.label_22.setText('您无需选择时间列，只需选择想要的工程值列即可')
            self.fileName = fileName
            (_, filename) = os.path.split(self.fileName) # 分离文件路径
            self.label.setText(filename)  # 显示选择的文件名
            self.tableView.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
            ff = open(fileName, 'r', newline='')
            mytext = ff.read()
            ff.close()
            f = open(fileName, 'r')
            with f:
                if mytext.count(';') <= mytext.count('\t'):
                    reader = csv.reader(f)
                    self.model.clear()
                    count = 0
                    for row in reader:
                        count = count+1
                        items = [QtGui.QStandardItem(field) for field in row]
                        self.model.appendRow(items)
                        if count==200:
                            break
                    self.tableView.resizeColumnsToContents()
                else:
                    reader = csv.reader(f, delimiter=';')

                    self.model.clear()
                    for row in reader:
                        items = [QtGui.QStandardItem(field) for field in row]
                        self.model.appendRow(items)
                    self.tableView.resizeColumnsToContents()

            f_temp = open(fileName, 'r')
            reader_temp = csv.reader(f_temp)
            # 设定默认训练、验证与测试集划分
            self.time_col = [row[0]for row in reader_temp]
            length_timecol = len(self.time_col)
            self.trainstarttime = self.time_col[1]# 训练开始时间初始化
            self.trainendtime = self.time_col[int(5/10*length_timecol)] # 训练结束时间初始化
            self.validstarttime = self.time_col[int(5/10*length_timecol)+1] # 验证开始时间初始化
            self.validendtime = self.time_col[int(6/10*length_timecol)] # 验证结束时间初始化
            self.teststarttime = self.time_col[int(6/10*length_timecol)+1] # 测试开始时间初始化
            self.testendtime = self.time_col[int(length_timecol)-self.pred_len] # 测试结束时间初始化
            self.showstarttime = self.time_col[int(6/10*length_timecol)+1] # 展示开始时间初始化
            self.showendtime = self.time_col[int(length_timecol)-self.pred_len] # 展示结束时间初始化
            self.errorshowstarttime = self.time_col[int(6/10*length_timecol)+1] # 展示开始时间初始化
            self.errorshowendtime = self.time_col[int(length_timecol)-self.pred_len] # 展示结束时间初始化
            self.allerrorshowstarttime = self.time_col[int(6/10*length_timecol)+1] # 总误差展示开始时间初始化
            self.allerrorshowendtime = self.time_col[int(length_timecol)-self.pred_len] # 总误差展示结束时间初始化
            # 显示默认时间
            self.label_21.clear()
            self.label_21.setText('训练、验证与测试集默认按照5：1：4分配，数据集划分与训练设置均可自行修改。') # 显示项目名
            self.lineEdit_trainstarttime.clear()
            self.lineEdit_trainstarttime.setText(self.trainstarttime)
            self.lineEdit_trainendtime.setText(self.trainendtime)
            self.lineEdit_validstarttime.setText(self.validstarttime)
            self.lineEdit_validendtime.setText(self.validendtime)
            self.lineEdit_teststarttime.setText(self.teststarttime)
            self.lineEdit_testendtime.setText(self.testendtime)
            self.lineEdit_showstarttime.setText(self.showstarttime)
            self.lineEdit_showendtime.setText(self.showendtime)
            self.lineEdit_errorshowstarttime.setText(self.errorshowstarttime)
            self.lineEdit_errorshowendtime.setText(self.errorshowendtime)
            self.lineEdit_allerrorshowstarttime.setText(self.allerrorshowstarttime)
            self.lineEdit_allerrorshowendtime.setText(self.allerrorshowendtime)
            # 显示默认训练设置
            self.lineEdit_epoch.setText(str(self.epoch))
            self.lineEdit_batchsize.setText(str(self.batch_size))
            self.lineEdit_rate.setText(str(self.learning_rate))
            self.lineEdit_inputlength.setText(str(self.seq_len))
            self.lineEdit_predlength.setText(str(self.pred_len))
            self.lineEdit_labellength.setText(str(self.label_len))
            # header = next(reader_temp)
            lossname = ['mae','mse','dtw']
            for name in lossname:
                self.comboBox_lossfc.addItem(name)
            errorname = ['error','AE','SE','APE']
            for name in errorname:
                self.comboBox_error.addItem(name)
            allerrorname = ['AE','SE']
            for name in allerrorname:
                self.comboBox_allerror.addItem(name)
                self.comboBox_alarm.addItem(name)
            
        else:
            self.label.setText(self.error1)

    def finishedEdit(self):
        self.tableView.resizeColumnsToContents()

    # 选择列，返回列索引
    def selectedColumns(self):
        # return indexes
        indexes = self.selectionModel.selectedIndexes()
        index_columns = []

        for index in indexes:
            if not index.column() in index_columns:
                index_columns.append(index.column())
        # index_columns = list(set(index_columns))

        return index_columns
    

    def all_value(self):
        self.selectionModel.clear()
        f = open(self.fileName, 'r')
        reader = csv.reader(f)
        header = next(reader)
        self.head_name = header[1::]
        self.all_choose = True

        # model = self.tableView.model()
        # for i,header_name in enumerate(header[1::]):
        #     self.head_name.append(model.item(0,i).text())
    
    def no_value(self):
        self.selectionModel.clear()
        
    # 选择dataframe
    def df_choose(self):
        # 获取所选数据
        f = open(self.fileName, 'r')
        reader = csv.reader(f)
        header = next(reader)
        self.dic = {}
        self.spoon_dic = {}
        self.midpoint_dic = {}
        # 直接添加时间列
        self.dic['date'] = []
        for row in reader:
             self.dic['date'].append(row[0])

        if self.all_choose:
            selectedcolumns = [i+1 for i in range(len(self.head_name))]
            for i, index in enumerate(selectedcolumns):
                self.dic[header[index]] = []
                self.spoon_dic[header[index]] = []
        else:
            self.head_name=[]
            selectedcolumns = self.selectedColumns()
            for i, index in enumerate(selectedcolumns):
                self.dic[header[index]] = []
                self.spoon_dic[header[index]] = []
                self.head_name.append(header[index])

        self.label_22.setText('正在读取数据')
        QApplication.processEvents() # 实时刷新显示
        for i, index in enumerate(selectedcolumns):
            try:
                f = open(self.fileName, 'r')
                with f:
                    reader = csv.reader(f)
                    offhead = next(reader) # 去掉第一行
                    self.dic[header[index]] = [float(row[selectedcolumns[i]]) for row in reader]
                    self.spoon_dic[header[index]] = self.dic[header[index]][::int(len(self.dic[header[index]])/100)]
                    ## 求解上下四分位数
                    self.midpoint_dic[header[index]] = np.percentile(self.spoon_dic[header[index]], (25, 50, 75), interpolation='midpoint')
                    
            except:
                self.label_22.setVisible(True)
                self.label_22.setText("!!!请检查数据格式!!!")
                QApplication.processEvents() # 实时刷新显示
                return -999    
        self.label_22.clear()
        self.label_22.setVisible(True)
        self.label_22.setText('选择数据：'+str(self.head_name))
        QApplication.processEvents() # 实时刷新显示
        self.data_df = pd.DataFrame(self.dic)
        (filepath, filename) = os.path.split(self.fileName) # 分离文件路径
        self.newfilename = os.path.join(filepath,'data_df.csv')
        # self.data_df.to_csv(self.newfilename,index=False,encoding='gbk') # 新建dataframe保存到同一文件夹下
        self.data_df_bool = True
        self.comboBox_model_2.clear()
        self.comboBox_alarm_value.clear()
        self.comboBox_pred.clear()
        self.comboBox_errorshow.clear()
        for name in self.head_name:
            self.comboBox_model_2.addItem(name) # 输出数据也应当为选择的数据
            self.comboBox_pred.addItem(name) # 展示数据也应当为选择的数据
            self.comboBox_errorshow.addItem(name) # 展示数据也应当为选择的数据
            self.comboBox_alarm_value.addItem(name) # 异常数报警也应当为选择的数据
        self.comboBox_alarm_value.addItem('全部遥测数据')
        self.all_choose=False # 全选取消
        self.label.setText('输出数据请在输入数据中选择')
        self.output_columns_list = self.head_name # 当没选择输出列时，默认所有列输出 
        self.output_index_list = list(range(len(self.head_name))) # 当没选择输出列时，默认所有列输出
        


    # 选取输出列

    def output_choose(self):
        self.multi_pred = False
        f = open(self.fileName, 'r')
        reader = csv.reader(f)
        header = next(reader)
        selectedcolumns = self.selectedColumns()
        self.output_columns_list = []
        self.output_index_list = []
        for i, index in enumerate(selectedcolumns):
            self.output_columns_list.append(header[index])
        
        self.output_index_list = [self.head_name.index(output_head) for output_head in self.output_columns_list]
        # print(self.output_columns_list)
        # print(self.output_index_list)
        self.label_22.clear()
        self.label_22.setVisible(True)
        self.label_22.setText('选择数据：'+str(self.output_columns_list))
        self.comboBox_alarm_value.clear()
        self.comboBox_pred.clear()
        self.comboBox_errorshow.clear()
        for name in self.output_columns_list:
            self.comboBox_pred.addItem(name) # 展示数据也应当为选择的数据
            self.comboBox_errorshow.addItem(name) # 展示数据也应当为选择的数据
            self.comboBox_alarm_value.addItem(name) # 异常数报警也应当为选择的数据


    # 读取
    def readpth(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open PTH", "*.pth")
        if fileName:
            print(fileName)
            self.modelpath = fileName # 把pth预训练存成一种属性，同时需要注意此时模型要与选择变量、输入输出步长匹配
            try:
                self.issue_pth = fileName.split('/')[-2] # 这个pth代表的项目信息：包括模型名称、输入输出步长
                self.label_21.clear()
                self.label_21.setText(fileName.split('/')[-2]) # 显示项目名
                # self.label_21.setText('预训练模型已加载')
                self.label_dataset.setText('加载预训练模型后，你可以选择训练0次直接测试')
                # 选择预训练模型后，一些参数不能再更改了
                self.comboBox_model.clear()
                self.comboBox_model.addItem(self.issue_pth.split('_')[0])
                self.modelname = self.issue_pth.split('_')[0]
                self.lineEdit_inputlength.clear()
                self.lineEdit_inputlength.setText(self.issue_pth.split('_')[3].split('sl')[-1])
                self.lineEdit_labellength.clear()
                self.lineEdit_labellength.setText(self.issue_pth.split('_')[4].split('ll')[-1])
                self.lineEdit_predlength.clear()
                self.lineEdit_predlength.setText(self.issue_pth.split('_')[5].split('pl')[-1])
            except:
                self.label_21.setText(self.error3)

        else:
            self.label_21.setText(self.error2)

    # 数据集划分与展示
    def datasetshow(self):
        if self.datasetshow_plot:
            self.win.removeItem(self.datasetshow_plot)
            self.datasetshow_plot = None
        line_color = ['r','b','g'] # 训练、验证、测试三种线色
        self.label_dataset.clear()
        self.label_dataset.setVisible(False)
        # 选择了数据集、选择指定数据、且有有效时间输入的情况下，才可以画图
        # if self.fileName and self.data_df.bool():
        if self.fileName and self.data_df_bool:
            # if not self.lineEdit_trainstarttime.text() in self.time_col:
            #     self.label_dataset.setVisible(True)
            #     self.label_dataset.setText('请输入合理的训练开始时间。')
            #     return -999
            # if not self.lineEdit_trainendtime.text() in self.time_col:
            #     self.label_dataset.setVisible(True)
            #     self.label_dataset.setText('请输入合理的训练结束时间。')
            #     return -999
            # if not self.lineEdit_validstarttime.text() in self.time_col:
            #     self.label_dataset.setVisible(True)
            #     self.label_dataset.setText('请输入合理的验证开始时间。')
            #     return -999
            # if not self.lineEdit_validendtime.text() in self.time_col:
            #     self.label_dataset.setVisible(True)
            #     self.label_dataset.setText('请输入合理的验证结束时间。')
            #     return -999
            # if not self.lineEdit_teststarttime.text() in self.time_col:
            #     self.label_dataset.setVisible(True)
            #     self.label_dataset.setText('请输入合理的测试开始时间。')
            #     return -999
            # if not self.lineEdit_testendtime.text() in self.time_col:
            #     self.label_dataset.setVisible(True)
            #     self.label_dataset.setText('请输入合理的测试结束时间。')
            #     return -999
            if self.lineEdit_trainstarttime.text()>max(self.time_col) or self.lineEdit_trainstarttime.text()<min(self.time_col):
                self.label_dataset.setVisible(True)
                self.label_dataset.setText('请输入合理的训练开始时间。')
                return -999
            if self.lineEdit_trainendtime.text()>max(self.time_col) or self.lineEdit_trainendtime.text()<min(self.time_col):
                self.label_dataset.setVisible(True)
                self.label_dataset.setText('请输入合理的训练结束时间。')
                return -999
            if self.lineEdit_validstarttime.text()>max(self.time_col) or self.lineEdit_validstarttime.text()<min(self.time_col):
                self.label_dataset.setVisible(True)
                self.label_dataset.setText('请输入合理的验证开始时间。')
                return -999
            if self.lineEdit_validendtime.text()>max(self.time_col) or self.lineEdit_validendtime.text()<min(self.time_col):
                self.label_dataset.setVisible(True)
                self.label_dataset.setText('请输入合理的验证结束时间。')
                return -999
            if self.lineEdit_teststarttime.text()>max(self.time_col) or self.lineEdit_teststarttime.text()<min(self.time_col):
                self.label_dataset.setVisible(True)
                self.label_dataset.setText('请输入合理的测试开始时间。')
                return -999
            if self.lineEdit_testendtime.text()>max(self.time_col) or self.lineEdit_testendtime.text()<min(self.time_col):
                self.label_dataset.setVisible(True)
                self.label_dataset.setText('请输入合理的测试结束时间。')
                return -999
            target_name = self.comboBox_model_2.currentText()
            # 显示判断
            if self.firstdataset:
                reply = QMessageBox.question(self, "数据集显示选择框", "是否显示数据集划分结果？\n选择“No”并不影响网络训练。", QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
                self.firstdataset=False
            else:
                reply=QMessageBox.Yes
            if reply==QMessageBox.Yes:
                partition = ['训练集','验证集','测试集']
                # 总时间
                Time = self.time_col[1::]
                dic = self.dic
                xdict = dict(enumerate(np.array(Time)))
                xdict_index = list(xdict.keys())
                # 输入时间更新
                # self.trainstarttime = self.lineEdit_trainstarttime.text() # 训练开始时间初始化
                # self.trainendtime = self.lineEdit_trainendtime.text() # 训练结束时间初始化
                # self.validstarttime = self.lineEdit_validstarttime.text() # 验证开始时间初始化
                # self.validendtime = self.lineEdit_validendtime.text() # 验证结束时间初始化
                # self.teststarttime = self.lineEdit_teststarttime.text() # 测试开始时间初始化
                # self.testendtime = self.lineEdit_testendtime.text() # 测试结束时间初始化
                self.trainstarttime = self.time_approximation(self.lineEdit_trainstarttime.text(),Time) # 训练开始时间初始化
                self.trainendtime = self.time_approximation(self.lineEdit_trainendtime.text(),Time) # 训练结束时间初始化
                self.validstarttime = self.time_approximation(self.lineEdit_validstarttime.text(),Time) # 验证开始时间初始化
                self.validendtime = self.time_approximation(self.lineEdit_validendtime.text(),Time) # 验证结束时间初始化
                self.teststarttime = self.time_approximation(self.lineEdit_teststarttime.text(),Time) # 测试开始时间初始化
                self.testendtime = self.time_approximation(self.lineEdit_testendtime.text(),Time) # 测试结束时间初始化

                self.lineEdit_showstarttime.setText(self.teststarttime) # 展示开始时间文本框更新
                self.lineEdit_showendtime.setText(self.testendtime) # 展示结束时间文本框更新
                self.lineEdit_errorshowstarttime.setText(self.teststarttime) # 展示开始时间文本框更新
                self.lineEdit_errorshowendtime.setText(self.testendtime) # 展示结束时间文本框更新
                self.lineEdit_allerrorshowstarttime.setText(self.teststarttime)
                self.lineEdit_allerrorshowendtime.setText(self.testendtime)
                # 训练时间
                xdict_train = xdict_index[Time.index(self.trainstarttime):Time.index(self.trainendtime)]
                # xdict_train = 
                # 验证时间
                xdict_valid = xdict_index[Time.index(self.validstarttime):Time.index(self.validendtime)]
                # 测试时间
                xdict_test = xdict_index[Time.index(self.teststarttime):Time.index(self.testendtime)]
                # 汇总
                xdict_all = [xdict_train, xdict_valid, xdict_test]
                start_end_index = [[Time.index(self.trainstarttime),Time.index(self.trainendtime)],[Time.index(self.validstarttime),Time.index(self.validendtime)],[Time.index(self.teststarttime),Time.index(self.testendtime)]]
                # 坐标轴设置
                axis_1 = [(i, Time[i]) for i in range(0, len(Time), 24 * int(len(Time)/100))]
                stringaxis = pg.AxisItem(orientation='bottom')
                stringaxis.setTicks([axis_1])
                # stringaxis.setTicks([axis_1, xdict.items()[::int(len(xdict.items())/1000)+1]])
                # stringaxis.setTicks([axis_1, xdict.items()])
                self.datasetshow_plot = self.win.addPlot(axisItems={'bottom': stringaxis})
                self.datasetshow_plot.addLegend(size=(150, 80))
                self.datasetshow_plot.showGrid(x=True, y=True, alpha=0.5)
                self.datasetshow_plot.setLabel(axis='left')
                self.datasetshow_plot.setLabel(axis='bottom', text='日期')
                if len(Time)>=100000:
                    x=xdict_index[0::int(len(Time)/10000)]
                    self.datasetshow_plot.plot(x=xdict_index[0::int(len(Time)/10000)], y=dic[target_name][0::int(len(Time)/10000)], pen='grey',
                                            name=target_name)
                else:
                    self.datasetshow_plot.plot(x=xdict_index, y=dic[target_name], pen='grey',
                                            name=target_name)
                for i,part in enumerate(partition):
                    if len(Time)>=100000:
                        self.datasetshow_plot.plot(x=xdict_all[i][0::int(len(Time)/10000)], y=dic[target_name][start_end_index[i][0]:start_end_index[i][1]][0::int(len(Time)/10000)], pen=line_color[i],
                                                name=target_name+part)
                    else:
                        self.datasetshow_plot.plot(x=xdict_all[i], y=dic[target_name][start_end_index[i][0]:start_end_index[i][1]], pen=line_color[i],
                                                name=target_name+part)

    def time_approximation(self,time_str_real,time_list):
        '''
        时间逼近函数，给定一个真实时间，在time_list寻找与其最接近的时间
        '''
        time_strp_real = datetime.strptime(time_str_real, "%Y-%m-%d %H:%M:%S")
        time_sec_real = time.mktime(time_strp_real.timetuple())

        for i, time_str in enumerate(time_list):
            if time_str_real==time_str:
                return time_str_real # 若存在，则直接返回
            elif time_str_real>time_list[i] and time_str_real<time_list[i+1]:
                # 若不存在，则寻找范围
                time_strp = datetime.strptime(time_list[i], "%Y-%m-%d %H:%M:%S")
                time_strp_1 = datetime.strptime(time_list[i+1], "%Y-%m-%d %H:%M:%S")
                time_sec = time.mktime(time_strp.timetuple())
                time_sec_1 =time.mktime(time_strp_1.timetuple())
                time_sec_return = min([time_sec,time_sec_1], key=lambda x: abs(x - time_sec_real))
                time_str_return = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time_sec_return))
                return time_str_return

    # 开始训练
    def train_ui(self):

        args = dotdict()
        args.model = self.comboBox_model.currentText() # 复选框选择模型
        # args.data = 'beartem0529' # 数据任务
        # if self.modelpath==None:
        #     args.data,_ = QInputDialog.getText(self, "任务命名", "不要输入'?','!','_'等符号\n\n请输入任务名称:", QLineEdit.Normal, "project1")
        # else:
        #     args.data = self.issue_pth.split('_')[1]
        # args.data,_ = QInputDialog.getText(self, "任务命名", "不要输入'?','!','_'等符号\n\n请输入任务名称:", QLineEdit.Normal, "project1")

        # 命名
        if self.modelpath and self.lineEdit_epoch.text()=='0':
            args.data,bool_input = QInputDialog.getText(self, "任务命名", "测试任务无需重命名", QLineEdit.Normal, self.issue_pth.split('_')[1])
            if bool_input==False:
                return -999
        else:
            args.data,bool_input = QInputDialog.getText(self, "任务命名", "不要输入'?','!','_'等符号\n\n请输入任务名称:", QLineEdit.Normal, "project1")
            if bool_input==False:
                return -999
        (filepath, filename) = os.path.split(self.newfilename) # 分离文件路径
        args.root_path = filepath # root path of data file
        args.data_path = filename # data file
        args.data_df = self.data_df
        args.features = 'M' # forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate
        args.target = '动量轮FMW2轴承温度' # target feature in S or MS task
        args.freq = 'h' # freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h
        if self.modelpath:
            (args.checkpoints,_) = os.path.split(self.modelpath) # 分离模型参数文件路径
            args.checkpoints = args.checkpoints.split(self.issue_pth)[0] # 取名称文件夹之前的路径
        else:
            args.checkpoints = './informer_checkpoints' # location of model checkpoints
        self.checkpoints = args.checkpoints
        args.train_time_range = [self.time_approximation(self.lineEdit_trainstarttime.text(),self.time_col[1::]),
                                 self.time_approximation(self.lineEdit_trainendtime.text(),self.time_col[1::])] # 训练时间范围
        args.valid_time_range = [self.time_approximation(self.lineEdit_validstarttime.text(),self.time_col[1::]),
                                 self.time_approximation(self.lineEdit_validendtime.text(),self.time_col[1::])] # 验证时间范围
        args.test_time_range = [self.time_approximation(self.lineEdit_teststarttime.text(),self.time_col[1::]),
                                self.time_approximation(self.lineEdit_testendtime.text(),self.time_col[1::])] # 测试时间范围
        args.seq_len = self.lineEdit_inputlength.text() # input sequence length of Informer encoder
        if not args.seq_len.isdigit():
            self.label_dataset.setVisible(True)
            self.label_dataset.setText('请输入正确的输入步长')
            return -999
        else:
            args.seq_len = int(args.seq_len)
            self.seq_len = int(args.seq_len)
        args.label_len = self.lineEdit_labellength.text() # start token length of Informer decoder
        if not args.label_len.isdigit():
            self.label_dataset.setVisible(True)
            self.label_dataset.setText('请输入正确的重叠步长')
            return -999
        else:
            args.label_len = int(args.label_len)
            self.label_len = int(args.label_len)
        args.pred_len = self.lineEdit_predlength.text() # prediction sequence length
        if not args.pred_len.isdigit():
            self.label_dataset.setVisible(True)
            self.label_dataset.setText('请输入正确的预测步长')
            return -999
        else:
            args.pred_len = int(args.pred_len)
            self.pred_len = int(args.pred_len)
            for i in range(self.pred_len):
                self.comboBox_stepshow.addItem(str(i+1))
                self.comboBox_stepshow_2.addItem(str(i+1))
                self.comboBox_stepshow_3.addItem(str(i+1))
                self.comboBox_alarm_stepshow.addItem(str(i+1))
        # Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]
        args.enc_in = len(self.head_name) # encoder input size
        args.dec_in = len(self.output_columns_list) # decoder input size
        args.c_out = len(self.output_columns_list) # output size
        args.factor = 5 # probsparse attn factor
        args.d_model = 512 # dimension of model
        # args.d_model = 256 # dimension of model
        args.n_heads = 8 # num of heads
        args.e_layers = 2 # num of encoder layers
        # args.e_layers = 1 # num of encoder layers
        args.d_layers = 2 # num of decoder layers
        args.s_layers = '3,2,1' # num of layers
        args.d_ff = 2048 # dimension of fcn in model
        args.dropout = 0.05 # dropout
        args.attn = 'prob' # attention used in encoder, options:[prob, full]
        args.embed = 'timeF' # time features encoding, options:[timeF, fixed, learned]
        args.activation = 'gelu' # activation
        args.distil = True # whether to use distilling in encoder
        args.output_attention = False # whether to output attention in ecoder
        args.mix = True
        args.padding = 0
        args.inverse = True
        args.des = 'train'
        args.batch_size = self.lineEdit_batchsize.text() # 当前设置batchsize
        args.output_index_list = self.output_index_list # 多对少输出时决定输出的列
        if not args.batch_size.isdigit():
            self.label_dataset.setVisible(True)
            self.label_dataset.setText('请输入正确的batchsize')
            return -999
        else:
            args.batch_size = int(args.batch_size)

        args.learning_rate = self.lineEdit_rate.text()
        if not self.is_number(args.learning_rate):
            self.label_dataset.setVisible(True)
            self.label_dataset.setText('请输入正确的初始学习率')
            return -999
        else:
            args.learning_rate = float(args.learning_rate)

        args.loss = self.comboBox_lossfc.currentText() # 当前选择的损失函数
        args.lradj = 'type1' # 'type1','type2'
        args.use_amp = False # whether to use automatic mixed precision training
        args.num_workers = 0
        args.itr = 1
        args.train_epochs = self.lineEdit_epoch.text() # 当前设置的epoch
        if not args.train_epochs.isdigit():
            self.label_dataset.setVisible(True)
            self.label_dataset.setText('请输入正确的训练轮数')
            return -999
        else:
            args.train_epochs = int(args.train_epochs)
            self.epoch = args.train_epochs
        args.patience = 20
        args.use_gpu = True if torch.cuda.is_available() else False
        args.gpu = 0
        args.use_multi_gpu = False
        args.devices = '0'
        # cuda调用
        args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
        if args.use_gpu and args.use_multi_gpu:
            args.devices = args.devices.replace(' ','')
            device_ids = args.devices.split(',')
            args.device_ids = [int(id_) for id_ in device_ids]
            args.gpu = args.device_ids[0]
        # 频率
        args.detail_freq = args.freq
        args.freq = args.freq[-1:]

        # 开始调用函数
        Exp = Exp_Informer
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}'.format(args.model, args.data, args.features, 
                        args.seq_len, args.label_len, args.pred_len,
                        args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.attn, args.factor, args.embed, args.distil, args.mix, args.des, ii)
            self.setting = setting
            # set experiments
            exp = Exp(args)
            self.plainTextEdit.appendPlainText('开始训练')
            QApplication.processEvents() # 实时刷新显示
            if args.train_epochs==0:
                self.plainTextEdit.appendPlainText('若你设置训练轮数为0，则默认采用指定任务文件夹下预训练模型进行测试')
                QApplication.processEvents() # 实时刷新显示
                self.plainTextEdit.appendPlainText('若上次模型采取了不同的工程值列，这可能导致无法显示的报错')
                QApplication.processEvents() # 实时刷新显示
            # train
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))   
            exp.train(setting,self.plainTextEdit,self.modelpath)
            
            self.plainTextEdit.appendPlainText('开始测试')
            QApplication.processEvents() # 实时刷新显示
            # test
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting,self.plainTextEdit)
            self.plainTextEdit.appendPlainText('训练与测试结束')
            QApplication.processEvents() # 实时刷新显示
            torch.cuda.empty_cache()

    def pred_show(self):
        if self.show_plot:
            self.win_show.removeItem(self.show_plot)
            self.label_pred_path.clear()
            self.show_plot = None
        # 展示真实值与预测值
        self.label_pred_time.setText('测试集时间范围：{0}--{1}'.format(self.teststarttime,self.testendtime))
        # 展示时间更新
        self.showstarttime = self.lineEdit_showstarttime.text()
        self.showendtime = self.lineEdit_showendtime.text()
        # 步长展示更新
        stepshow = int(self.comboBox_stepshow.currentText())-1
        # 测试结果
        preds = np.load('./results/'+self.setting+'/pred.npy')
        trues = np.load('./results/'+self.setting+'/true.npy')
        # 展示工程值
        target_name = self.comboBox_pred.currentText()
        col = self.output_columns_list.index(target_name)
        # 时间索引
        Time = self.time_col[1::]
        xdict = dict(enumerate(np.array(Time)))
        xdict_index = list(xdict.keys())
        time_show_origin = Time[Time.index(self.teststarttime)+stepshow:Time.index(self.testendtime)+stepshow]
        # time测试时间除以32，余数刚好小于72
        remainder = len(time_show_origin)-trues.shape[0]
        time_show = Time[Time.index(self.teststarttime)+stepshow:Time.index(self.testendtime)+stepshow-remainder]
        xdict_show = xdict_index[Time.index(self.teststarttime)+stepshow:Time.index(self.testendtime)+stepshow-remainder]
        # 总展示时间即为测试集时间
        # print(len(time_show))
        self.df_show = pd.DataFrame(preds[:,stepshow,col],columns=['pred'],index=pd.DatetimeIndex(time_show))
        self.df_show['true'] = trues[:,stepshow,col]
        # df_show['error'] = trues[:,stepshow,col]-preds[:,stepshow,col]
        
        # 坐标轴设置
        line_color = ['r','b']
        axis_1 = [(i, Time[i]) for i in range(0, len(Time), 24 * (int(len(time_show)/100)+1))]
        stringaxis = pg.AxisItem(orientation='bottom')
        stringaxis.setTicks([axis_1])
        # stringaxis.setTicks([axis_1, xdict.items()])
        # stringaxis.setTicks([axis_1, xdict.items()[::int(len(xdict.items())/1000)+1]])
        self.show_plot = self.win_show.addPlot(axisItems={'bottom': stringaxis})
        self.show_plot.addLegend(size=(150, 80))
        self.show_plot.showGrid(x=True, y=True, alpha=0.5)
        self.show_plot.setLabel(axis='left')
        self.show_plot.setLabel(axis='bottom', text='日期')
        if len(Time)>100000:
            self.show_plot.plot(x=xdict_show[0::int(len(Time)/10000)], y=self.df_show['true'].to_list()[0::int(len(Time)/10000)], pen=line_color[0],
                                    name=target_name+'真实值')
            self.show_plot.plot(x=xdict_show[0::int(len(Time)/10000)], y=self.df_show['pred'].to_list()[0::int(len(Time)/10000)], pen=line_color[1],
                                    name=target_name+'预测值')
        else:
            self.show_plot.plot(x=xdict_show, y=self.df_show['true'].to_list(), pen=line_color[0],
                                    name=target_name+'真实值')
            self.show_plot.plot(x=xdict_show, y=self.df_show['pred'].to_list(), pen=line_color[1],
                                    name=target_name+'预测值')

        path = os.path.join(self.checkpoints, self.setting)
        if not os.path.exists(path):
            os.makedirs(path)
        best_model_path = path+'/'+'checkpoint.pth'
        self.label_pred_path.setText(best_model_path)


    def error_show(self):
        if self.errorshow_plot:
            self.win_error.removeItem(self.errorshow_plot)
            self.errorshow_plot = None
        # 展示时间更新
        self.errorshowstarttime = self.lineEdit_errorshowstarttime.text()
        self.errorshowendtime = self.lineEdit_errorshowendtime.text()
        # 步长展示更新
        stepshow = int(self.comboBox_stepshow_2.currentText())-1
        # 测试结果
        preds = np.load('./results/'+self.setting+'/pred.npy')
        trues = np.load('./results/'+self.setting+'/true.npy')
        # 展示工程值
        target_name = self.comboBox_errorshow.currentText()
        col = self.output_columns_list.index(target_name)
        # 时间索引
        Time = self.time_col[1::]
        xdict = dict(enumerate(np.array(Time)))
        xdict_index = list(xdict.keys())
        time_show_origin = Time[Time.index(self.teststarttime)+stepshow:Time.index(self.testendtime)+stepshow]
        # time测试时间除以32，余数刚好小于72
        remainder = len(time_show_origin)-trues.shape[0]
        time_show = Time[Time.index(self.teststarttime)+stepshow:Time.index(self.testendtime)+stepshow-remainder]
        xdict_show = xdict_index[Time.index(self.teststarttime)+stepshow:Time.index(self.testendtime)+stepshow-remainder]
        # 总展示时间即为测试集时间
        self.df_show = pd.DataFrame(preds[:,stepshow,col],columns=['pred'],index=pd.DatetimeIndex(time_show))
        self.df_show['true'] = trues[:,stepshow,col]
        self.df_show['error'] = trues[:,stepshow,col]-preds[:,stepshow,col]
        self.df_show['AE'] = abs(trues[:,stepshow,col]-preds[:,stepshow,col]) # 占位
        self.df_show['SE'] = (trues[:,stepshow,col]-preds[:,stepshow,col])**2
        aaa = abs((trues[:,stepshow,col]-preds[:,stepshow,col])/trues[:,stepshow,col])
        self.df_show['APE'] = abs((trues[:,stepshow,col]-preds[:,stepshow,col])/trues[:,stepshow,col])
        self.df_show['APE'][np.isinf(self.df_show['APE'])]=0
        
        # 坐标轴设置
        axis_1 = [(i, Time[i]) for i in range(0, len(Time), 24 * (int(len(time_show)/100)+1))]
        stringaxis = pg.AxisItem(orientation='bottom')
        stringaxis.setTicks([axis_1])
        # stringaxis.setTicks([axis_1, xdict.items()])
        self.errorshow_plot = self.win_error.addPlot(axisItems={'bottom': stringaxis})
        self.errorshow_plot.addLegend(size=(150, 80))
        self.errorshow_plot.showGrid(x=True, y=True, alpha=0.5)
        self.errorshow_plot.setLabel(axis='left')
        self.errorshow_plot.setLabel(axis='bottom', text='日期')
        if self.comboBox_error.currentText()=='error':
            self.errorshow_plot.plot(x=xdict_show, y=self.df_show['error'].to_list(),pen = 'orange',name=target_name+'误差')
            self.lineEdit_threshold.setText(str(np.mean(self.df_show['error'].to_list())))
        if self.comboBox_error.currentText()=='AE':
            self.errorshow_plot.plot(x=xdict_show, y=self.df_show['AE'].to_list(),pen = 'orange',name=target_name+'平均绝对值误差')
            self.lineEdit_threshold.setText(str(np.mean(self.df_show['AE'].to_list())))
        if self.comboBox_error.currentText()=='SE':
            self.errorshow_plot.plot(x=xdict_show, y=self.df_show['SE'].to_list(),pen = 'orange',name=target_name+'均方误差')
            self.lineEdit_threshold.setText(str(np.mean(self.df_show['SE'].to_list())))
        if self.comboBox_error.currentText()=='APE':
            self.errorshow_plot.plot(x=xdict_show, y=self.df_show['APE'].to_list(),pen = 'orange',name=target_name+'平均百分比误差')
            bbb = np.nanmean(self.df_show['APE'].to_list())
            ccc = max(self.df_show['APE'].to_list())
            ddd = min(self.df_show['APE'].to_list())
            self.lineEdit_threshold.setText(str(np.nanmean(self.df_show['APE'].to_list())))


        # threshold = self.lineEdit_threshold.text()
        # if self.is_number(threshold):
        #     self.threshold=float(threshold)
        #     self.errorshow_plot.plot(x=[xdict_show[0],xdict_show[-1]], y=[self.threshold,self.threshold],pen = 'red',name=target_name+'异常阈值')
        #     self.label_thresholderror.clear()
        # else:
        #     self.label_thresholderror.setText('请输入正确格式的阈值数据')

    def allerror(self):
        if self.allerrorshow_plot:
            self.win_allerror.removeItem(self.allerrorshow_plot)
            self.allerrorshow_plot = None
        # 展示时间更新
        self.allerrorshowstarttime = self.lineEdit_allerrorshowstarttime.text()
        self.allerrorshowendtime = self.lineEdit_allerrorshowendtime.text()
        # 步长展示更新
        stepshow = int(self.comboBox_stepshow_3.currentText())-1
        # 测试结果
        preds = np.load('./results/'+self.setting+'/pred.npy')
        trues = np.load('./results/'+self.setting+'/true.npy')
        # 展示工程值
        # 时间索引
        Time = self.time_col[1::]
        xdict = dict(enumerate(np.array(Time)))
        xdict_index = list(xdict.keys())
        time_show_origin = Time[Time.index(self.teststarttime)+stepshow:Time.index(self.testendtime)+stepshow]
        # time测试时间除以32，余数刚好小于72
        remainder = len(time_show_origin)-trues.shape[0]
        time_show = Time[Time.index(self.teststarttime)+stepshow:Time.index(self.testendtime)+stepshow-remainder]
        xdict_show = xdict_index[Time.index(self.teststarttime)+stepshow:Time.index(self.testendtime)+stepshow-remainder]


        # 总展示时间即为测试集时间
        # 别忘了归一化
        self.df_show_all = pd.DataFrame(trues[:,stepshow,0]-preds[:,stepshow,0],columns=['error0'],index=pd.DatetimeIndex(time_show))
        self.df_show_all['AE'] = abs(trues[:,stepshow,0]-preds[:,stepshow,0])/(max(trues[:,stepshow,0]-preds[:,stepshow,0])-min(trues[:,stepshow,0]-preds[:,stepshow,0])) # 占位
        self.df_show_all['SE'] = ((trues[:,stepshow,0]-preds[:,stepshow,0])/(max(trues[:,stepshow,0]-preds[:,stepshow,0])-min(trues[:,stepshow,0]-preds[:,stepshow,0])))**2 # 占位
        if trues.shape[2]>1:
            for col in range(1,trues.shape[2]):
                self.df_show_all['error'+str(col)] = trues[:,stepshow,col]-preds[:,stepshow,col]
                self.df_show_all['AE'] = self.df_show_all['AE']+abs(trues[:,stepshow,col]-preds[:,stepshow,col])/(max(trues[:,stepshow,col]-preds[:,stepshow,col])-min(trues[:,stepshow,col]-preds[:,stepshow,col])) # 占位
                self.df_show_all['SE'] = self.df_show_all['SE']+((trues[:,stepshow,col]-preds[:,stepshow,col])/(max(trues[:,stepshow,col]-preds[:,stepshow,col])-min(trues[:,stepshow,col]-preds[:,stepshow,col])))**2
        # 坐标轴设置
        axis_1 = [(i, Time[i]) for i in range(0, len(Time), 24 * (int(len(time_show)/100)+1))]
        stringaxis = pg.AxisItem(orientation='bottom')
        stringaxis.setTicks([axis_1])
        # stringaxis.setTicks([axis_1, xdict.items()])
        self.allerrorshow_plot = self.win_allerror.addPlot(axisItems={'bottom': stringaxis})
        self.allerrorshow_plot.addLegend(size=(150, 80))
        self.allerrorshow_plot.showGrid(x=True, y=True, alpha=0.5)
        self.allerrorshow_plot.setLabel(axis='left')
        self.allerrorshow_plot.setLabel(axis='bottom', text='日期')
        if self.comboBox_allerror.currentText()=='AE':
            self.allerrorshow_plot.plot(x=xdict_show, y=self.df_show_all['AE'].to_list(),pen = 'orange',name='归一化总平均绝对值误差')
            self.lineEdit_allerrorthreshold.setText(str(np.nanmean(self.df_show_all['AE'].to_list())))
        if self.comboBox_allerror.currentText()=='SE':
            self.allerrorshow_plot.plot(x=xdict_show, y=self.df_show_all['SE'].to_list(),pen = 'orange',name='归一化总均方误差')
            self.lineEdit_allerrorthreshold.setText(str(np.nanmean(self.df_show_all['SE'].to_list())))
        # allthreshold = self.lineEdit_allerrorthreshold.text()
        # if self.is_number(allthreshold):
        #     self.allthreshold=float(allthreshold)
        #     self.allerrorshow_plot.plot(x=[xdict_show[0],xdict_show[-1]], y=[self.allthreshold,self.allthreshold],pen = 'red',name='异常阈值')
        #     self.label_thresholdallerror.clear()
        # else:
        #     self.label_thresholdallerror.setText('请输入正确格式的阈值数据')

    # 异常报警
    def alarm_test(self,df_show,test_time_show,threshold_error,time_count_all,error_name='AE'):
        diagnosis_time = []
        time_count = 0
        diagnosis_number = [0]*len(test_time_show)
        for i,error in enumerate(df_show[error_name].to_list()):
            if abs(error)>=threshold_error:
                time_count = time_count+1
            else:
                time_count=0
            if time_count==time_count_all:
                diagnosis_time.extend(test_time_show[i-time_count_all+1:i+1])
                diagnosis_number[i-time_count_all+1:i+1] = [1]*time_count_all
            if time_count>time_count_all:
                diagnosis_time.append(test_time_show[i])
                diagnosis_number[i] = 1
        return diagnosis_time, diagnosis_number
    
    # 时间分割
    # 根据异常数分割异常时间段
    def time_knife(self,diagnosis_number,test_time_show):
        time_start = []
        time_end = []
        for i in range(0,len(diagnosis_number)-1):
            if i==0 and diagnosis_number[i]==1:
                time_start.append(test_time_show[i])
            if diagnosis_number[i]==0 and diagnosis_number[i+1]==1:
                time_start.append(test_time_show[i+1])
        for i in range(1,len(diagnosis_number)):
            if diagnosis_number[i-1]==1 and diagnosis_number[i]==0:
                time_end.append(test_time_show[i-1])
            if i==len(diagnosis_number)-1 and diagnosis_number[i]==1:
                time_end.append(test_time_show[i])
        return time_start,time_end
    
    def general_equation(self,first_x,first_y,second_x,second_y):
        # 斜截式 y = kx + b 
        A = second_y-first_y
        B = first_x-second_x
        C = second_x * first_y - first_x * second_y
        k = -1 * A / B
        b = -1 * C / B
        return k, b
    
    # 上下包络线
    # signal是list即可
    def envelope_extraction(self,signal):
        # s = signal.astype(float )
        # q_u = np.zeros(s.shape)
        # q_l =  np.zeros(s.shape)
        s = signal
        #在插值值前加上第一个值。这将强制模型对上包络和下包络模型使用相同的起点。
        #Prepend the first value of (s) to the interpolating values. This forces the model to use the same starting point for both the upper and lower envelope models.
        u_x = [0,] #上包络的x序列
        u_y = [s[0],] #上包络的y序列

        l_x = [0,] #下包络的x序列
        l_y = [s[0],] #下包络的y序列

        # 检测波峰和波谷，并分别标记它们在u_x,u_y,l_x,l_中的位置。 
        #Detect peaks and troughs and mark their location in u_x,u_y,l_x,l_y respectively.

        for k in range(1,len(s)-1):
            if (sign(s[k]-s[k-1])==1) and (sign(s[k]-s[k+1])==1):
                u_x.append(k)
                u_y.append(s[k])

            if (sign(s[k]-s[k-1])==-1) and ((sign(s[k]-s[k+1]))==-1):
                l_x.append(k)
                l_y.append(s[k])

        u_x.append(len(s)-1) #上包络与原始数据切点x
        u_y.append(s[-1]) #对应的值

        l_x.append(len(s)-1) #下包络与原始数据切点x
        l_y.append(s[-1]) #对应的值

        #u_x,l_y是不连续的，以下代码把包络转为和输入数据相同大小的数组[便于后续处理，如滤波]
        upper_envelope_y = np.zeros(len(signal))
        lower_envelope_y = np.zeros(len(signal))
        upper_envelope_y[0] = u_y[0]#边界值处理
        upper_envelope_y[-1] = u_y[-1]
        lower_envelope_y[0] =  l_y[0]#边界值处理
        lower_envelope_y[-1] =  l_y[-1]
        #上包络
        last_idx,next_idx = 0, 0
        k, b = self.general_equation(u_x[0], u_y[0], u_x[1], u_y[1]) #初始的k,b
        for e in range(1, len(upper_envelope_y)-1):

            if e not in u_x:
                v = k * e + b
                upper_envelope_y[e] = v
            else:
                idx = u_x.index(e)
                upper_envelope_y[e] = u_y[idx]
                last_idx = u_x.index(e)
                next_idx = u_x.index(e) + 1
                #求连续两个点之间的直线方程
                k, b = self.general_equation(u_x[last_idx], u_y[last_idx], u_x[next_idx], u_y[next_idx])        
        
        #下包络
        last_idx,next_idx = 0, 0
        k, b = self.general_equation(l_x[0], l_y[0], l_x[1], l_y[1]) #初始的k,b
        for e in range(1, len(lower_envelope_y)-1):

            if e not in l_x:
                v = k * e + b
                lower_envelope_y[e] = v
            else:
                idx = l_x.index(e)
                lower_envelope_y[e] = l_y[idx]
                last_idx = l_x.index(e)
                next_idx = l_x.index(e) + 1
                #求连续两个切点之间的直线方程
                k, b = self.general_equation(l_x[last_idx], l_y[last_idx], l_x[next_idx], l_y[next_idx])     

        return upper_envelope_y, lower_envelope_y
    

    def WrittingNotOfOther(self, tag):
        if tag != len(self.head_name): # 不是最后一项
            self.comboBox_label.clear()
            allalarmlabel = ['异常阈值','阈值系数']
            for name in allalarmlabel:
                self.comboBox_label.addItem(name)
            # 报警设置初始化
            self.lineEdit_alarm.setText(str(1.5*(self.midpoint_dic[self.comboBox_alarm_value.currentText()][2]-self.midpoint_dic[self.comboBox_alarm_value.currentText()][0])))
            self.lineEdit_alarm_time.setText(str(self.alarm_time_threshold))
        else:
            self.comboBox_label.clear()
            allalarmlabel = ['异常阈值']
            self.comboBox_label.addItem(allalarmlabel[0])
            # 报警设置初始化
            self.lineEdit_alarm.setText(str(0.8))
            self.lineEdit_alarm_time.setText(str(self.alarm_time_threshold))
    def WrittingNotOfOther_label(self, tag):
        if tag==1:
            self.lineEdit_alarm.setText(str(1.5))
            self.lineEdit_alarm_time.setText(str(self.alarm_time_threshold))
        if tag==0:
            # 报警设置初始化
            if self.comboBox_alarm_value.currentText()!='全部遥测数据':
                self.lineEdit_alarm.setText(str(1.5*(self.midpoint_dic[self.comboBox_alarm_value.currentText()][2]-self.midpoint_dic[self.comboBox_alarm_value.currentText()][0])))
                self.lineEdit_alarm_time.setText(str(self.alarm_time_threshold))


    def alarm(self):
        if self.alarm_plot:
            self.win_alarm.removeItem(self.alarm_plot)
            self.alarm_plot = None
        self.plainTextEdit_alarm.clear()
        # self.df_show = pd.DataFrame()
        
        if self.is_number(self.lineEdit_alarm_time.text()):
            self.alarm_time_threshold = int(self.lineEdit_alarm_time.text())
        else:
            self.label_alarm.setText('请输入正确时间阈值')

        # 步长展示更新
        stepshow = int(self.comboBox_stepshow_3.currentText())-1
        # 测试结果
        preds = np.load('./results/'+self.setting+'/pred.npy')
        trues = np.load('./results/'+self.setting+'/true.npy')
        # 展示工程值
        # 时间索引
        Time = self.time_col[1::]
        xdict = dict(enumerate(np.array(Time)))
        xdict_index = list(xdict.keys())
        time_show_origin = Time[Time.index(self.teststarttime)+stepshow:Time.index(self.testendtime)+stepshow]
        # time测试时间除以32，余数刚好小于72
        remainder = len(time_show_origin)-trues.shape[0]
        time_show = Time[Time.index(self.teststarttime)+stepshow:Time.index(self.testendtime)+stepshow-remainder]
        xdict_show = xdict_index[Time.index(self.teststarttime)+stepshow:Time.index(self.testendtime)+stepshow-remainder]
        if self.comboBox_alarm_value.currentText()=='全部遥测数据':
            if self.is_number(self.lineEdit_alarm.text()):
                self.alarm_threshold = float(self.lineEdit_alarm.text())
            else:
                self.label_alarm.setText('请输入正确阈值')
            _,diagnosis_number = self.alarm_test(self.df_show_all,time_show,self.alarm_threshold,self.alarm_time_threshold,self.comboBox_alarm.currentText())
            time_start,time_end = self.time_knife(diagnosis_number,time_show)
        else:
            if self.comboBox_label.currentText()=='异常阈值':
                if self.is_number(self.lineEdit_alarm.text()):
                    self.alarm_threshold = float(self.lineEdit_alarm.text())
                else:
                    self.label_alarm.setText('请输入正确阈值')
            if self.comboBox_label.currentText()=='阈值系数':
                if self.is_number(self.lineEdit_alarm.text()):
                    self.alarm_threshold = float(self.lineEdit_alarm.text())*(self.midpoint_dic[self.comboBox_alarm_value.currentText()][2]-self.midpoint_dic[self.comboBox_alarm_value.currentText()][0])
                else:
                    self.label_alarm.setText('请输入正确阈值')
            self.label_alarm.setText('异常阈值=健康信号上下四分位数×阈值系数（默认1.5）')
            # 展示工程值
            target_name = self.comboBox_alarm_value.currentText()
            col = self.output_columns_list.index(target_name)
            # 时间索引
            Time = self.time_col[1::]
            xdict = dict(enumerate(np.array(Time)))
            xdict_index = list(xdict.keys())
            time_show_origin = Time[Time.index(self.teststarttime)+stepshow:Time.index(self.testendtime)+stepshow]
            # time测试时间除以32，余数刚好小于72
            remainder = len(time_show_origin)-trues.shape[0]
            time_show = Time[Time.index(self.teststarttime)+stepshow:Time.index(self.testendtime)+stepshow-remainder]
            xdict_show = xdict_index[Time.index(self.teststarttime)+stepshow:Time.index(self.testendtime)+stepshow-remainder]
            # 总展示时间即为测试集时间
            self.df_show = pd.DataFrame(preds[:,stepshow,col],columns=['pred'],index=pd.DatetimeIndex(time_show))
            self.df_show['true'] = trues[:,stepshow,col]
            self.df_show['error'] = trues[:,stepshow,col]-preds[:,stepshow,col]
            self.df_show['AE'] = abs(trues[:,stepshow,col]-preds[:,stepshow,col]) # 占位
            self.df_show['SE'] = (trues[:,stepshow,col]-preds[:,stepshow,col])**2
            # 报警
            # 阈值使用自适应阈值
            _,diagnosis_number = self.alarm_test(self.df_show,time_show,self.alarm_threshold,self.alarm_time_threshold,self.comboBox_alarm.currentText())
            time_start,time_end = self.time_knife(diagnosis_number,time_show)
        # print(self.alarm_threshold)

        if time_start==[]:
            self.plainTextEdit_alarm.appendPlainText('没有异常区段')
            QApplication.processEvents() # 实时刷新显示
        else:
            self.plainTextEdit_alarm.appendPlainText('判断'+self.comboBox_alarm_value.currentText()+'异常时间段为：')
            QApplication.processEvents() # 实时刷新显示
            for i,starttime in enumerate(time_start):
                self.plainTextEdit_alarm.appendPlainText(starttime+'至'+time_end[i])
                QApplication.processEvents() # 实时刷新显示

        # 坐标轴设置
        axis_1 = [(i, Time[i]) for i in range(0, len(Time), 24 * (int(len(time_show)/100)+1))]
        stringaxis = pg.AxisItem(orientation='bottom')
        stringaxis.setTicks([axis_1])
        # stringaxis.setTicks([axis_1, xdict.items()])
        self.alarm_plot = self.win_alarm.addPlot(axisItems={'bottom': stringaxis})
        self.alarm_plot.addLegend(size=(150, 80))
        self.alarm_plot.showGrid(x=True, y=True, alpha=0.5)
        self.alarm_plot.setLabel(axis='left')
        self.alarm_plot.setLabel(axis='bottom', text='日期')
        self.alarm_plot.plot(x=xdict_show, y=diagnosis_number,pen = 'blue',name=self.comboBox_alarm_value.currentText()+'报警异常数')

        

        


    # 添加中文的确认退出提示框1
    def closeEvent(self, event):
        # 创建一个消息盒子（提示框）
        quitMsgBox = QMessageBox()
        # 设置提示框的标题
        quitMsgBox.setWindowTitle('确认提示')
        # 设置提示框的内容
        quitMsgBox.setText('你确认退出吗？')
        # 设置按钮标准，一个yes一个no
        quitMsgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        # 获取两个按钮并且修改显示文本
        buttonY = quitMsgBox.button(QMessageBox.Yes)
        buttonY.setText('确定')
        buttonN = quitMsgBox.button(QMessageBox.No)
        buttonN.setText('取消')
        quitMsgBox.exec_()
        # 判断返回值，如果点击的是Yes按钮，我们就关闭组件和应用，否则就忽略关闭事件
        if quitMsgBox.clickedButton() == buttonY:
            event.accept()
        else:
            event.ignore()




if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = Ctest_readscv()
    ex.show()
    sys.exit(app.exec_())
    
