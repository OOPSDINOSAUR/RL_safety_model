import mysql.connector
import numpy as np
import scipy.io as sio
import pandas as pd
import warnings
import csv
import math
import pickle
from tqdm import tqdm
from scipy.interpolate import interp1d
from scipy.spatial.distance import pdist, squareform
import datetime

print('Load Total_bili')
Total_bili = pd.read_csv('/RL-safety-models/data_sofa/BILIRUBIN.csv').values[:, [0, 2, 3]]

print('Load bp_mean')
MeanBP = pd.read_csv('/RL-safety-models/data_sofa/BP_MEAN.csv').values[:, [0, 2, 3]]

print('Load Creatinine')
Creatinine = pd.read_csv('/RL-safety-models/data_sofa/CREATININE.csv').values[:, [0, 2, 3]]

print('Load GCS')
GCS = pd.read_csv('/RL-safety-models/data_sofa/GCS.csv').values[:, :3]

print('Load PaO2')
PaO2 = pd.read_csv('/RL-safety-models/data_sofa/PAO2.csv').values[:, [0, 2, 3]]

print('Load FiO2')
FiO2 = pd.read_csv('/RL-safety-models/data_sofa/FiO2.csv').values[:, :3]

print('Load Platelet')
Platelet = pd.read_csv('/RL-safety-models/data_sofa/PLATELET.csv', low_memory=False).values[:, [0, 2, 3]]

dfs = [Total_bili, MeanBP, Creatinine,GCS, PaO2, FiO2, Platelet]

n = 0
for i, array in enumerate(dfs):
    # Determining the shape of an array
    num_rows = array.shape[0]
    # Create an array of all 1's and reshape it to (num_rows, 1).
    column = np.full((num_rows, 1), i + 1, dtype=array.dtype)
    # Use the hstack function to horizontally stack the original and newly created arrays
    dfs[i] = np.hstack((array, column))
    # Reassign the modified array to the position of the original array
    if i == 0:
        Total_bili = dfs[i]
    elif i == 1:
        MeanBP = dfs[i]
    elif i == 2:
        Creatinine = dfs[i]
    elif i == 3:
        GCS = dfs[i]
    elif i == 4:
        PaO2 = dfs[i]
    elif i == 5:
        FiO2 = dfs[i]
    elif i == 6:
        Platelet = dfs[i]

merged_df = np.vstack(dfs)

irow = 0
reformat = np.full((30000000, 10), np.nan)
for icustayid in range(1, 100000):
    print(icustayid)

    temp_list = []
    id_if = merged_df[:, 0] == icustayid + 200000
    temp = merged_df[id_if,:]
    temp_list.append(temp[:, 1].reshape(-1, 1))

    if (len(temp_list) == 0):
        t = np.array([])
    else:
        t = np.unique(np.vstack(temp_list)) # Finding the de-weighting time

    if (t.size != 0):
        for i in range(t.size):
            reformat[irow, 0] = i + 1  # timestep
            reformat[irow, 1] = icustayid + 200000
            charttime = pd.to_datetime(t[i])
            epoch_time = (charttime - datetime.datetime(1970, 1, 1)).total_seconds()
            reformat[irow, 2] = epoch_time  # charttime
            all_if = temp[:, 1] == t[i]
            value = temp[all_if, 2]
            col = temp[all_if, 3].astype('int64')
            for index, c in enumerate(col):
                reformat[irow, 2 + c] = value[index]  # (locb(:,1)); %store available values
            irow = irow + 1

reformat = reformat[:irow, :]  # delete extra unused rows


sample_and_hold=np.array([['Total_bili', 'MeanBP', 'Creatinine','GCS', 'PaO2', 'FiO2', 'Platelet'],['28','2','28','6','8','12','28']])

def SAH(reformat, vitalslab_hold):
    """

    Parameters
    ----------
    reformat：数据
    vitalslab_hold：保质期

    Returns
    -------

    """
    temp = reformat.copy()

    hold = vitalslab_hold[1, :].astype(int)
    nrow = temp.shape[0]
    ncol = temp.shape[1]

    lastcharttime = np.zeros(ncol)
    lastvalue = np.zeros(ncol)
    oldstayid = temp[0, 1]

    for i in range(3, 10):
        if (i % 10 == 0):
            print(i)
        for j in range(0, nrow):

            if oldstayid != temp[j, 1]:
                lastcharttime = np.zeros(ncol)
                lastvalue = np.zeros(ncol)
                oldstayid = temp[j, 1]

            if np.isnan(temp[j, i]) == 0:
                lastcharttime[i] = temp[j, 2]
                lastvalue[i] = temp[j, i]

            if j > 0:
                if (np.isnan(temp[j, i])) and (temp[j, 1] == oldstayid) and (
                        (temp[j, 2] - lastcharttime[i]) <= hold[i - 3] * 3600):
                    temp[j, i] = lastvalue[i]

    return temp


reformatsah = SAH(reformat, sample_and_hold)

mask = np.isnan(reformatsah[:, 7]) | np.isnan(reformatsah[:, 8])
PaO2_FiO2 = np.where(mask, np.nan, reformatsah[:, 7] / reformatsah[:, 8])
reformatsah = np.column_stack((reformatsah, PaO2_FiO2))

reformatsah = pd.DataFrame(reformatsah, columns=['timestep','icustayid', 'charttime', 'Total_bili', 'MeanBP', 'Creatinine','GCS', 'PaO2', 'FiO2', 'Platelet','PaO2_FiO2'])

reformatsah.to_csv('/RL-safety-models/data_sofa/reformat.csv', index=False, na_rep='NaN')
