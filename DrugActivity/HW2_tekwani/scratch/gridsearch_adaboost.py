import numpy as np
import pandas as pd
from time import time
from sklearn import grid_search
from sklearn.metrics import f1_score, accuracy_score, cohen_kappa_score, classification_report
import subprocess

df_train = pd.read_csv('../data/train.csv', sep='\t', index_col=False, header=None,
                       names=['Active', 'Structure'])




subprocess.call(['spd-say', '"Finished execution."'])