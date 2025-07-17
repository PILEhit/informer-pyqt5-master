# %%
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

dict_1 = {}
dict_1['date'] = ['ab','bdsdds']
dict_1['value'] = [1,2]

df_dict = pd.DataFrame(dict_1)
print(df_dict)
print(dict_1.date)