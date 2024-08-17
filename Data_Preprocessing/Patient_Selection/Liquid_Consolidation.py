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

print('Load vasoMV')
vasoMV = pd.read_csv(SOFA_PATH + 'VASO_MV.csv')

print('Load vasoCV')
vasoCV = pd.read_csv(SOFA_PATH + 'VASO_CV.csv')

print('Load UO')
UO = pd.read_csv(SOFA_PATH + 'UO.csv')

print('Load UOpreadm')
UOpreadm = pd.read_csv(SOFA_PATH + 'PREADM_UO.csv')

print('Load reformatsah')
reformat = pd.read_csv(SOFA_PATH + 'reformat.csv').values


dfs = [vasoCV, UO, UOpreadm]

for i, df in enumerate(dfs):
    charttime = pd.to_datetime(df['charttime'])
    charttime_seconds = []
    for time in charttime:
        time_diff = time - datetime.datetime(1970, 1, 1)
        time_seconds = time_diff.total_seconds()
        charttime_seconds.append(time_seconds)
    # 将结果存储回原始数据的某一列
    df['charttime'] = charttime_seconds

    if i == 0:
        vasoCV = df.values
    elif i == 1:
        UO = df.values
    elif i == 2:
        UOpreadm = df.values

STARTTIME = pd.to_datetime(vasoMV['starttime'])
STARTTIME_seconds = []
for time in STARTTIME:
    time_diff = time - datetime.datetime(1970, 1, 1)
    time_seconds = time_diff.total_seconds()
    STARTTIME_seconds.append(time_seconds)
# 将结果存储回原始数据的某一列
vasoMV['starttime'] = STARTTIME_seconds

ENDTIME = pd.to_datetime(vasoMV['endtime'])
ENDTIME_seconds = []
for time in ENDTIME:
    time_diff = time - datetime.datetime(1970, 1, 1)
    time_seconds = time_diff.total_seconds()
    ENDTIME_seconds.append(time_seconds)
# 将结果存储回原始数据的某一列
vasoMV['endtime'] = ENDTIME_seconds

vasoMV = vasoMV.values

###改时间记得
reformat = np.insert(reformat, 11, np.nan, axis=1)
reformat = np.insert(reformat, 12, np.nan, axis=1)
reformat = np.insert(reformat, 13, np.nan, axis=1)


timestep = 4  # resolution of timesteps, in hours 时间步长为4小时
irow = 0
icustayidlist = np.unique(reformat[:, 1].astype('int64'))
npt = icustayidlist.size  # number of patients 病人数

for i in range(npt):
    if (i % 10000 == 0):
        print(i)
    icustayid = icustayidlist[i]  # 1 to 100000, NOT 200 to 300K!
    print(icustayid)
    temp = reformat[reformat[:, 1] == icustayid, :]  # subtable of interest 从reformat数组中选择与当前病人ID相关的数据子集
    beg = temp[0, 2]  # timestamp of first record
    t0 = 0
    t1 = beg

    iv = np.where(vasoMV[:, 0] == icustayid )[0]  # rows of interest in vasoMV
    vaso1 = vasoMV[iv, :]  # subset of interest
    iv = np.where(vasoCV[:, 0] == icustayid )[0]  # rows of interest in vasoCV
    vaso2 = vasoCV[iv, :]  # subset of interest
    startv = vaso1[:, 2]  # start of VP infusion
    endv = vaso1[:, 3]  # end of VP infusions
    ratev = vaso1[:, 4]  # rate of VP infusion

    # URINE OUTPUT 用于处理尿量输出相关的信息。
    iu = np.where(UO[:, 0] == icustayid )[0]  # rows of interest in inputMV
    output = UO[iu, :]  # subset of interest
    pread = UOpreadm[UOpreadm[:, 0] == icustayid , 3]  # preadmission UO
    if pread.size != 0:  # store the value, if available  预入院尿量输出信息
        UOtot = np.nansum(pread)
    else:
        UOtot = 0;

    # adding the volume of urine produced before start of recording! 计算记录开始之前产生的尿量，并将其添加到预入院尿量输出的总和中
    UOnow = np.nansum(output[(output[:, 1] >= t0) & (output[:, 1] <= t1), 3])  # t0 and t1 defined above
    UOtot = np.nansum(np.array([UOtot, UOnow]))

    time = temp[:,2]

    for j in time:
        t0 = j - 3600 * timestep# left limit of time window
        t1 = j  # right limit of time window
        ii = (temp[:, 2] >= t0) & (temp[:, 2] <= t1)  # index of items in this time period 选择了在当前时间窗口内的数据项

        if np.sum(ii) > 0:

            # CHARTEVENTS and LAB VALUES (+ includes empty cols for shock index and P/F)

            # #####################   DISCUSS ADDING STUFF HERE / RANGE, MIN, MAX ETC   ################

            # VASOPRESSORS
            # for CV: dose at timestamps.
            # for MV: 4 possibles cases, each one needing a different way to compute the dose of VP actually administered:
            # ----t0---start----end-----t1----
            # ----start---t0----end----t1----
            # -----t0---start---t1---end
            # ----start---t0----t1---end----
            # 计算每个时间步长内以上四种情况 血管活性药物（VASOPRESSORS）的剂量
            # CV只有charttime MV end\starttime

            # MV
            v = ((endv >= t0) & (endv <= t1)) | ((startv >= t0) & (endv <= t1)) | ((startv >= t0) & (startv <= t1)) | (
                    (startv <= t0) & (endv >= t1))

            # CV
            v2 = vaso2[(vaso2[:, 2] >= t0) & (vaso2[:, 2] <= t1), 3]

            temp_list = []
            if (ratev[v].size != 0):# MV的rate是否为0
                temp_list.append(ratev[v].reshape(-1, 1))
            if (v2.size != 0):
                temp_list.append(v2.reshape(-1, 1))

            if (len(temp_list) != 0):
                rv = np.vstack(temp_list)
            else:
                rv = np.array([])

            v1 = np.nanmedian(rv)# 求中位数

            if (rv.size != 0):
                v2 = np.nanmax(rv)# 求最大
            else:
                v2 = np.array([])

            if v1.size != 0 and ~np.isnan(v1) and v2.size != 0 and ~np.isnan(v2):
                reformat[irow, 11] = v1  # median of dose of VP
                reformat[irow, 12] = v2  # max dose of VP

            # UO 算尿量
            UOnow = np.nansum(output[(output[:, 1] >= t0) & (output[:, 1] <= t1), 3])
            reformat[irow, 13] = np.nansum(UOnow)  # UO at this step

            irow = irow + 1

reformatsah = pd.DataFrame(reformat, columns=['timestep','icustayid', 'charttime', 'Total_bili', 'MeanBP', 'Creatinine','GCS', 'PaO2', 'FiO2', 'Platelet','PaO2_FiO2','median_dose_vaso', 'max_dose_vaso','output_4hourly'])

reformatsah.to_csv(SOFA_PATH + 'reformat_vaso.csv', index=False, na_rep='NaN')