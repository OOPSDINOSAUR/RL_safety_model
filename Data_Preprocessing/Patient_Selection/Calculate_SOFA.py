import mysql.connector
import numpy as np
import scipy.io as sio
import pandas as pd
import warnings
import csv
import math
import pickle
import os
from tqdm import tqdm
from scipy.interpolate import interp1d
from scipy.spatial.distance import pdist, squareform
import datetime
from constants import SOFA_PATH

print('Load reformatsah')
reformat = pd.read_csv(f'{SOFA_PATH}/reformat_vaso.csv')
reformat['max_dose_vaso'] = reformat['max_dose_vaso'].fillna(0)
reformat['output_4hourly'] = reformat['output_4hourly'].replace(0, np.nan)
s = reformat.loc[:, ['PaO2_FiO2', 'Platelet', 'Total_bili', 'MeanBP', 'max_dose_vaso', 'GCS', 'Creatinine',
                       'output_4hourly']].values


p = np.array([0, 1, 2, 3, 4])

s1 = np.transpose(np.array([s[:, 0] > 400, (s[:, 0] >= 300) & (s[:, 0] < 400), (s[:, 0] >= 200) & (s[:, 0] < 300),
                            (s[:, 0] >= 100) & (s[:, 0] < 200),
                            s[:, 0] < 100]))  # count of points for all 6 criteria of sofa
s2 = np.transpose(np.array([s[:, 1] > 150, (s[:, 1] >= 100) & (s[:, 1] < 150), (s[:, 1] >= 50) & (s[:, 1] < 100),
                            (s[:, 1] >= 20) & (s[:, 1] < 50), s[:, 1] < 20]))
s3 = np.transpose(np.array(
    [s[:, 2] < 1.2, (s[:, 2] >= 1.2) & (s[:, 2] < 2), (s[:, 2] >= 2) & (s[:, 2] < 6), (s[:, 2] >= 6) & (s[:, 2] < 12),
     s[:, 2] > 12]))
s4 = np.transpose(np.array(
    [s[:, 3] >= 70, (s[:, 3] < 70) & (s[:, 3] >= 65), s[:, 3] < 65, (s[:, 4] > 0) & (s[:, 4] <= 0.1), s[:, 4] > 0.1]))
s5 = np.transpose(np.array(
    [s[:, 5] > 14, (s[:, 5] > 12) & (s[:, 5] <= 14), (s[:, 5] > 9) & (s[:, 5] <= 12), (s[:, 5] > 5) & (s[:, 5] <= 9),
     s[:, 5] <= 5]))
s6 = np.transpose(np.array([s[:, 6] < 1.2, (s[:, 6] >= 1.2) & (s[:, 6] < 2), (s[:, 6] >= 2) & (s[:, 6] < 3.5),
                            ((s[:, 6] >= 3.5) & (s[:, 6] < 5)) | (s[:, 7] < 84), (s[:, 6] > 5) | (s[:, 7] < 34)]))

nrcol = reformat.shape[1]  # nr of variables in data
reformat = np.hstack([reformat, np.zeros((reformat.shape[0], 7))])

for i in range(0, reformat.shape[0]):
    p_s1 = p[s1[i, :]]
    p_s2 = p[s2[i, :]]
    p_s3 = p[s3[i, :]]
    p_s4 = p[s4[i, :]]
    p_s5 = p[s5[i, :]]
    p_s6 = p[s6[i, :]]

    if (p_s1.size == 0 or p_s2.size == 0 or p_s3.size == 0 or p_s4.size == 0 or p_s5.size == 0 or p_s6.size == 0):
        t = 0
    else:
        t = max(p_s1) + max(p_s2) + max(p_s3) + max(p_s4) + max(p_s5) + max(p_s6)  # SUM OF ALL 6 CRITERIA

    if (t):
        reformat[i, nrcol:nrcol + 7] = np.array([max(p_s1), max(p_s2), max(p_s3), max(p_s4), max(p_s5), max(p_s6), t]) #将每个器官系统的最大分数以及总分存储在数据矩阵中的新列中。

sofa = reformat[:, [0, 1, 2, -1]]
sofa = pd.DataFrame(sofa, columns=['timestep','icustayid', 'charttime', 'sofa'])
sofa.to_csv(f'{SOFA_PATH}/sofa.csv', index=False, na_rep='NaN')
selected_rows = sofa[sofa.iloc[:, -1] > 2]
selected_rows.to_csv(f'{SOFA_PATH}/sofa>2.csv', index=False, na_rep='NaN')

