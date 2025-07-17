# class_test
import re
import sys
import os
import csv
import numpy as np
import pandas as pd
import seaborn as sns
import pylab as plt
from PyQt5.QtCore import QAbstractTableModel, Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog, QMainWindow, QWidget, QVBoxLayout, QGridLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from pylab import mpl

class test(object):
    def __init__(self):
        self.aaa = 1

sss = test()
print(sss.aaa)