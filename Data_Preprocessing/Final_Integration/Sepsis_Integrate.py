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


print('Load demog')
demog = pd.read_csv(
    '/RL_safety_models/data_sepsis/DEMOG.csv')  # read as DataFrame

print('Load ce010')
ce010 = pd.read_csv(
    '/RL_safety_models/data_sepsis/CE_0_10000.csv').values

print('Load ce1020')
ce1020 = pd.read_csv(
    '/RL_safety_models/data_sepsis/CE_10000_20000.csv').values

print('Load ce2030')
ce2030 = pd.read_csv(
    '/RL_safety_models/data_sepsis/CE_20000_30000.csv').values

print('Load ce3040')
ce3040 = pd.read_csv(
    '/RL_safety_models/data_sepsis/CE_30000_40000.csv').values

print('Load ce4050')
ce4050 = pd.read_csv(
    '/RL_safety_models/data_sepsis/CE_40000_50000.csv').values

print('Load ce5060')
ce5060 = pd.read_csv(
    '/RL_safety_models/data_sepsis/CE_50000_60000.csv').values

print('Load ce6070')
ce6070 = pd.read_csv(
    '/RL_safety_models/data_sepsis/CE_60000_70000.csv').values

print('Load ce7080')
ce7080 = pd.read_csv(
    '/RL_safety_models/data_sepsis/CE_70000_80000.csv').values

print('Load ce8090')
ce8090 = pd.read_csv(
    '/RL_safety_models/data_sepsis/CE_80000_90000.csv').values

print('Load ce90100')
ce90100 = pd.read_csv(
    '/RL_safety_models/data_sepsis/CE_90000_100000.csv').values

print('Load labU')
labU = pd.read_csv(
    '/RL_safety_models/data_sepsis/LABS.csv').values

print('Load MV')
MV = pd.read_csv(
    '/RL_safety_models/data_sepsis/MECHVENT.csv').values

print('Load inputpreadm')
inputpreadm = pd.read_csv(
    '/RL_safety_models/data_sepsis/PREADM_FLUID.csv').values

print('Load inputpreadm')
inputpreadm = pd.read_csv(
    '/RL_safety_models/data_sepsis/PREADM_FLUID.csv').values

print('Load inputMV')
inputMV = pd.read_csv(
    '/RL_safety_models/data_sepsis/FLUID_MV.csv').values

print('Load inputCV')
inputCV = pd.read_csv(
    '/RL_safety_models/data_sepsis/FLUID_CV.csv').values

print('Load vasoMV')
vasoMV = pd.read_csv(
    '/RL_safety_models/data_sepsis/VASO_MV.csv').values

print('Load vasoCV')
vasoCV = pd.read_csv(
    '/RL_safety_models/data_sepsis/VASO_CV.csv').values

print('Load UOpreadm')
UOpreadm = pd.read_csv(
    '/RL_safety_models/data_sepsis/PREADM_UO.csv').values

print('Load UO')
UO = pd.read_csv(
    '/RL_safety_models/data_sepsis/UO.csv').values

print('Load sofa_period')
sofa_period = pd.read_csv(
    '/RL_safety_models/data_sepsis/SOFA_PERIOD_ADULT.csv').values


# To ignore 'Runtime Warning: Invalid value encountered in greater' caused by NaN
warnings.filterwarnings('ignore')

# Contains a mapping of project identifiers and concepts, as well as data for sampling and holding techniques.
Reflabs = pd.read_csv("/RL_safety_models/Data_Preprocessing/Consultation/reflabs.csv",header=None).values
Refvitals = pd.read_csv("/RL_safety_models/Data_Preprocessing/Consultation/refvitals.csv",header=None).values
sample_and_hold = pd.read_csv("/RL_safety_models/Data_Preprocessing/Consultation/sample_and_hold.csv",header=None) # numpy.ndarray

sample_and_hold.iloc[1,:] = sample_and_hold.iloc[1,:].astype(int)
sample_and_hold = sample_and_hold.values

# correct NaNs in DEMOG
demog.loc[np.isnan(demog.loc[:, 'morta_90']), 'morta_90'] = 0
demog.loc[np.isnan(demog.loc[:, 'morta_hosp']), 'morta_hosp'] = 0
demog.loc[np.isnan(demog.loc[:, 'elixhauser']), 'elixhauser'] = 0
# compute normalized rate of infusion
# if we give 100 ml of hypertonic fluid (600 mosm/l) at 100 ml/h (given in 1h) it is 200 ml of NS equivalent
# so the normalized rate of infusion is 200 ml/h (different volume in same duration)
inputMV = np.insert(inputMV, 7, np.nan, axis=1)  # Initialize the new column with nan
ii = inputMV[:, 4] != 0  # to avoid divide by zero
inputMV[ii, 7] = inputMV[ii, 6] * inputMV[ii, 5] / inputMV[ii, 4]

# replace itemid in labs with column number
# this will accelerate process later

def replace_item_ids(reference, data):
    temp = {}
    for i in range(data.shape[0]):
        key = data[i, 2]
        if (key not in temp):
            temp[key] = np.argwhere(reference == key)[0][0] + 1  # +1 because matlab index starts from 1
        data[i, 2] = temp[key]


replace_item_ids(Reflabs, labU)
replace_item_ids(Refvitals, ce010)
replace_item_ids(Refvitals, ce1020)
replace_item_ids(Refvitals, ce2030)
replace_item_ids(Refvitals, ce3040)
replace_item_ids(Refvitals, ce4050)
replace_item_ids(Refvitals, ce5060)
replace_item_ids(Refvitals, ce6070)
replace_item_ids(Refvitals, ce7080)
replace_item_ids(Refvitals, ce8090)
replace_item_ids(Refvitals, ce90100)

# ########################################################################
#           INITIAL REFORMAT WITH CHARTEVENTS, LABS AND MECHVENT
# ########################################################################

reformat = np.full((10000000, 68), np.nan)
irow = 0  # recording row for summary table
n = 0

for icustayid in sofa_period[:,0]:

    print(icustayid)

    if icustayid < 210000:
        temp = ce010[ce010[:, 0] == icustayid, :]
    elif icustayid >= 210000 and icustayid < 220000:
        temp = ce1020[ce1020[:, 0] == icustayid, :]
    elif icustayid >= 220000 and icustayid < 230000:
        temp = ce2030[ce2030[:, 0] == icustayid, :]
    elif icustayid >= 230000 and icustayid < 240000:
        temp = ce3040[ce3040[:, 0] == icustayid, :]
    elif icustayid >= 240000 and icustayid < 250000:
        temp = ce4050[ce4050[:, 0] == icustayid, :]
    elif icustayid >= 250000 and icustayid < 260000:
        temp = ce5060[ce5060[:, 0] == icustayid, :]
    elif icustayid >= 260000 and icustayid < 270000:
        temp = ce6070[ce6070[:, 0] == icustayid, :]
    elif icustayid >= 270000 and icustayid < 280000:
        temp = ce7080[ce7080[:, 0] == icustayid, :]
    elif icustayid >= 280000 and icustayid < 290000:
        temp = ce8090[ce8090[:, 0] == icustayid, :]
    elif icustayid >= 290000:
        temp = ce90100[ce90100[:, 0] == icustayid, :]

    ii = (temp[:, 1] >= sofa_period[n, 1]) & (
            temp[:, 1] <= sofa_period[n, 2])

    temp = temp[ii, :]

    # Select data related to the current icustayid from the LABEVENTS data and filter the data for a specific time period
    ii = labU[:, 0] == icustayid
    temp2 = labU[ii, :]
    ii = (temp2[:, 1] >= sofa_period[n, 1]) & (
            temp2[:, 1] <= sofa_period[n, 2])  # time period of interest -4h and +4h
    temp2 = temp2[ii, :]  # only time period of interest

    # Select data from Mech Vent data that is relevant to the current icustayid and filter the data for a specific time period
    ii = MV[:, 0] == icustayid
    temp3 = MV[ii, :]
    ii = (temp3[:, 1] >= sofa_period[n, 1]) & (
            temp3[:, 1] <= sofa_period[n, 2])  # time period of interest -4h and +4h

    temp3 = temp3[ii, :]  # only time period of interest

    # Combine the timestamps from the 3 data sources into one array and de-duplicate them
    temp_list = []
    if (temp.size != 0):
        temp_list.append(temp[:, 1].reshape(-1, 1))
    if (temp2.size != 0):
        temp_list.append(temp2[:, 1].reshape(-1, 1))
    if (temp3.size != 0):
        temp_list.append(temp3[:, 1].reshape(-1, 1))

    if (len(temp_list) == 0):
        t = np.array([])
    else:
        t = np.unique(
            np.vstack(temp_list))

    if (t.size != 0):
        for i in range(t.size):
            # CHARTEVENTS
            ii = temp[:, 1] == t[i]
            col = temp[ii, 2].astype('int64')
            value = temp[ii, 3]
            reformat[irow, 0] = i + 1  # timestep
            reformat[irow, 1] = icustayid
            reformat[irow, 2] = t[i]  # charttime
            for index, c in enumerate(col):
                reformat[irow, 2 + c] = value[index]  # (locb(:,1)); %store available values

            # LAB VALUES
            ii = temp2[:, 1] == t[i]
            col = temp2[ii, 2].astype('int64')
            value = temp2[ii, 3]
            for index, c in enumerate(col):
                reformat[irow, 30 + c] = value[index]  # store available values

            # MV
            ii = temp3[:, 1] == t[i]
            if np.nansum(ii) > 0:
                value = temp3[ii, 2:4]
                reformat[irow, 66:68] = value  # store available values
            else:
                reformat[irow, 66:68] = np.nan

            irow = irow + 1
    n = n + 1
reformat = reformat[:irow, :]  # delete extra unused rows

# In[ ]:


# ########################################################################
#                             OUTLIERS
# ########################################################################
#

def deloutabove(reformat, var, thres):
    # DELOUTABOVE delete values above the given threshold, for column 'var'
    ii = reformat[:, var] > thres
    reformat[ii, var] = np.nan
    return reformat

def deloutbelow(reformat, var, thres):
    # DELOUTBELOW delete values below the given threshold, for column 'var'
    ii = reformat[:, var] < thres
    reformat[ii, var] = np.nan
    return reformat

# weight
reformat = deloutabove(reformat, 4, 300)  # delete outlier above a threshold (300 kg), for variable # 5
# HR
reformat = deloutabove(reformat, 7, 250)

# BP
reformat = deloutabove(reformat, 8, 300)
reformat = deloutbelow(reformat, 9, 0)
reformat = deloutabove(reformat, 9, 200)
reformat = deloutbelow(reformat, 10, 0)
reformat = deloutabove(reformat, 10, 200)

# RR
reformat = deloutabove(reformat, 11, 80)

# SpO2
reformat = deloutabove(reformat, 12, 150)
ii = reformat[:, 12] > 100
reformat[ii, 12] = 100

# temp
ii = (reformat[:, 13] > 90) & (np.isnan(reformat[:, 14]))
reformat[ii, 14] = reformat[ii, 13]
reformat = deloutabove(reformat, 13, 90)

# interface / is in col 22

# FiO2
reformat = deloutabove(reformat, 22, 100)
ii = reformat[:, 22] < 1
reformat[ii, 22] = reformat[ii, 22] * 100
reformat = deloutbelow(reformat, 22, 20)
reformat = deloutabove(reformat, 23, 1.5)

# O2 FLOW
reformat = deloutabove(reformat, 24, 70)

# PEEP
reformat = deloutbelow(reformat, 25, 0)
reformat = deloutabove(reformat, 25, 40)

# TV
reformat = deloutabove(reformat, 26, 1800)

# MV
reformat = deloutabove(reformat, 27, 50)

# K+
reformat = deloutbelow(reformat, 31, 1)
reformat = deloutabove(reformat, 31, 15)

# Na
reformat = deloutbelow(reformat, 32, 95)
reformat = deloutabove(reformat, 32, 178)

# Cl
reformat = deloutbelow(reformat, 33, 70)
reformat = deloutabove(reformat, 33, 150)

# Glc
reformat = deloutbelow(reformat, 34, 1)
reformat = deloutabove(reformat, 34, 1000)

# Creat
reformat = deloutabove(reformat, 36, 150)

# Mg
reformat = deloutabove(reformat, 37, 10)

# Ca
reformat = deloutabove(reformat, 38, 20)

# ionized Ca
reformat = deloutabove(reformat, 39, 5)

# CO2
reformat = deloutabove(reformat, 40, 120)

# SGPT/SGOT
reformat = deloutabove(reformat, 41, 10000)
reformat = deloutabove(reformat, 42, 10000)

# Hb/Ht
reformat = deloutabove(reformat, 49, 20)
reformat = deloutabove(reformat, 50, 65)

# WBC
reformat = deloutabove(reformat, 52, 500)

# plt
reformat = deloutabove(reformat, 53, 2000)

# INR
reformat = deloutabove(reformat, 57, 20)

# pH
reformat = deloutbelow(reformat, 58, 6.7)
reformat = deloutabove(reformat, 58, 8)

# po2
reformat = deloutabove(reformat, 59, 700)

# pco2
reformat = deloutabove(reformat, 60, 200)

# BE
reformat = deloutbelow(reformat, 61, -50)

# lactate
reformat = deloutabove(reformat, 62, 30)

# In[ ]:


#####################################################################
# some more data manip / imputation from existing values

# estimate GCS from RASS
ii = (np.isnan(reformat[:, 5])) & (reformat[:, 6] >= 0)
reformat[ii, 5] = 15
ii = (np.isnan(reformat[:, 5])) & (reformat[:, 6] == -1)
reformat[ii, 5] = 14
ii = (np.isnan(reformat[:, 5])) & (reformat[:, 6] == -2)
reformat[ii, 5] = 12
ii = (np.isnan(reformat[:, 5])) & (reformat[:, 6] == -3)
reformat[ii, 5] = 11
ii = (np.isnan(reformat[:, 5])) & (reformat[:, 6] == -4)
reformat[ii, 5] = 6
ii = (np.isnan(reformat[:, 5])) & (reformat[:, 6] == -5)
reformat[ii, 5] = 3

# Ensure consistency of FiO2 data between percentage form, decimal form
ii = (~np.isnan(reformat[:, 22])) & (np.isnan(reformat[:, 23]))
reformat[ii, 23] = reformat[ii, 22] / 100
ii = (~np.isnan(reformat[:, 23])) & (np.isnan(reformat[:, 22]))
reformat[ii, 22] = reformat[ii, 23] * 100


# Matthieu Komorowski - Imperial College London 2017
# will copy a value in the rows below if the missing values are within the
# hold period for this variable (e.g. 48h for weight, 2h for HR...)
# vitalslab_hold = 2x55 cell (with row1 = strings of names ; row 2 = hold time)

def SAH(reformat, vitalslab_hold):
    """

    Parameters
    ----------
    reformat：data
    vitalslab_hold：best before date

    Returns
    -------

    """
    temp = reformat.copy()

    hold = vitalslab_hold[1, :]
    nrow = temp.shape[0]
    ncol = temp.shape[1]

    lastcharttime = np.zeros(ncol)
    lastvalue = np.zeros(ncol)
    oldstayid = temp[0, 1]

    for i in range(3, ncol):
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
                        (temp[j, 2] - lastcharttime[i]) <= hold[i - 3] * 3600):  # note : hold has 53 cols, temp has 55
                    temp[j, i] = lastvalue[i]

    return temp
# If the current value is a missing value and the time difference from the previous observation is within the hold time, the previous observation is used to populate the current value.


# ESTIMATE FiO2 /// with use of interface / device (cannula, mask, ventilator....)
reformatsah = SAH(reformat, sample_and_hold)  # do SAH first to handle this task

#Fill in missing values in FiO2 data based on different interfaces or devices (e.g., catheters, masks, ventilators, etc.)
# NO FiO2, YES O2 flow, no interface OR cannula
ii = np.where(np.isnan(reformatsah[:, 22]) & (~np.isnan(reformatsah[:, 24])) & (
        (reformatsah[:, 21] == 0) | (reformatsah[:, 21] == 2)))[0]
reformat[ii[reformatsah[ii, 24] <= 15], 22] = 70
reformat[ii[reformatsah[ii, 24] <= 12], 22] = 62
reformat[ii[reformatsah[ii, 24] <= 10], 22] = 55
reformat[ii[reformatsah[ii, 24] <= 8], 22] = 50
reformat[ii[reformatsah[ii, 24] <= 6], 22] = 44
reformat[ii[reformatsah[ii, 24] <= 5], 22] = 40
reformat[ii[reformatsah[ii, 24] <= 4], 22] = 36
reformat[ii[reformatsah[ii, 24] <= 3], 22] = 32
reformat[ii[reformatsah[ii, 24] <= 2], 22] = 28
reformat[ii[reformatsah[ii, 24] <= 1], 22] = 24

# NO FiO2, NO O2 flow, no interface OR cannula
# FiO2 is not provided, i.e. the value in column 22 is NaN.
# O2 flow is not provided, i.e., the value in column 24 is also NaN.
# The device type is either None (0) or Conduit (2), i.e., column 21 has a value of 0 or 2.
ii = np.where((np.isnan(reformatsah[:, 22])) & (np.isnan(reformatsah[:, 24])) & (
        (reformatsah[:, 21] == 0) | (reformatsah[:, 21] == 2)))[
    0]  # no fio2 given and o2flow given, no interface OR cannula
reformat[ii, 22] = 21

# NO FiO2, YES O2 flow, face mask OR.... OR ventilator (assume it's face mask)
ii = np.where((np.isnan(reformatsah[:, 22])) & (~np.isnan(reformatsah[:, 24])) & (
        (reformatsah[:, 21] == np.nan) | (reformatsah[:, 21] == 1) | (reformatsah[:, 21] == 3) | (
        reformatsah[:, 21] == 4) | (reformatsah[:, 21] == 5) | (reformatsah[:, 21] == 6) | (
                reformatsah[:, 21] == 9) | (reformatsah[:, 21] == 10)))[0]
reformat[ii[reformatsah[ii, 24] <= 15], 22] = 75
reformat[ii[reformatsah[ii, 24] <= 12], 22] = 69
reformat[ii[reformatsah[ii, 24] <= 10], 22] = 66
reformat[ii[reformatsah[ii, 24] <= 8], 22] = 58
reformat[ii[reformatsah[ii, 24] <= 6], 22] = 40
reformat[ii[reformatsah[ii, 24] <= 4], 22] = 36

# NO FiO2, NO O2 flow, face mask OR ....OR ventilator
ii = np.where((np.isnan(reformatsah[:, 22])) & (np.isnan(reformatsah[:, 24])) & (
        (reformatsah[:, 21] == np.nan) | (reformatsah[:, 21] == 1) | (reformatsah[:, 21] == 3) | (
        reformatsah[:, 21] == 4) | (reformatsah[:, 21] == 5) | (reformatsah[:, 21] == 6) | (
                reformatsah[:, 21] == 9) | (reformatsah[:, 21] == 10)))[
    0]  # no fio2 given and o2flow given, no interface OR cannula
reformat[ii, 22] = np.nan

# NO FiO2, YES O2 flow, Non rebreather mask
ii = np.where((np.isnan(reformatsah[:, 22])) & (~np.isnan(reformatsah[:, 24]) & (reformatsah[:, 21] == 7)))[0]
reformat[ii[reformatsah[ii, 24] >= 10], 22] = 90
reformat[ii[reformatsah[ii, 24] >= 15], 22] = 100
reformat[ii[reformatsah[ii, 24] < 10], 22] = 80
reformat[ii[reformatsah[ii, 24] <= 8], 22] = 70
reformat[ii[reformatsah[ii, 24] <= 6], 22] = 60

# NO FiO2, NO O2 flow, NRM
ii = np.where((np.isnan(reformatsah[:, 22])) & (np.isnan(reformatsah[:, 24]) & (reformatsah[:, 21] == 7)))[
    0]  # no fio2 given and o2flow given, no interface OR cannula
reformat[ii, 22] = np.nan

# update again FiO2 columns
ii = (~np.isnan(reformat[:, 22])) & (np.isnan(reformat[:, 23]))
reformat[ii, 23] = reformat[ii, 22] / 100
ii = (~np.isnan(reformat[:, 23])) & (np.isnan(reformat[:, 22]))
reformat[ii, 22] = reformat[ii, 23] * 100

# BP   3 * mean = sys + 2 * dia
ii = (~np.isnan(reformat[:, 8])) & (~np.isnan(reformat[:, 9])) & (np.isnan(reformat[:, 10]))
reformat[ii, 10] = (3 * reformat[ii, 9] - reformat[ii, 8]) / 2
ii = (~np.isnan(reformat[:, 8])) & (~np.isnan(reformat[:, 10])) & (np.isnan(reformat[:, 9]))
reformat[ii, 9] = (reformat[ii, 8] + 2 * reformat[ii, 10]) / 3
ii = (~np.isnan(reformat[:, 9])) & (~np.isnan(reformat[:, 10])) & (np.isnan(reformat[:, 8]))
reformat[ii, 8] = 3 * reformat[ii, 9] - 2 * reformat[ii, 10]


# some values recorded in the wrong column
ii = (reformat[:, 14] > 25) & (reformat[:, 14] < 45)  # tempF close to 37deg??!
reformat[ii, 13] = reformat[ii, 14]
reformat[ii, 14] = np.nan
ii = reformat[:, 13] > 70  # tempC > 70?!!! probably degF
reformat[ii, 14] = reformat[ii, 13]
reformat[ii, 13] = np.nan
# transform
ii = (~np.isnan(reformat[:, 13])) & (np.isnan(reformat[:, 14]))
reformat[ii, 14] = reformat[ii, 13] * 1.8 + 32
ii = (~np.isnan(reformat[:, 14])) & (np.isnan(reformat[:, 13]))
reformat[ii, 13] = (reformat[ii, 14] - 32) / 1.8

# Hb/Ht 49 Haemoglobin 50 Haematocrit Interconversion
ii = (~np.isnan(reformat[:, 49])) & (np.isnan(reformat[:, 50]))
reformat[ii, 50] = (reformat[ii, 49] * 2.862) + 1.216
ii = (~np.isnan(reformat[:, 50])) & (np.isnan(reformat[:, 49]))
reformat[ii, 49] = (reformat[ii, 50] - 1.216) / 2.862

# BILI Bilirubin interconversion 43 Total bilirubin value 44 Bilirubin value
ii = (~np.isnan(reformat[:, 43])) & (np.isnan(reformat[:, 44]))
reformat[ii, 44] = (reformat[ii, 43] * 0.6934) - 0.1752
ii = (~np.isnan(reformat[:, 44])) & (np.isnan(reformat[:, 43]))
reformat[ii, 43] = (reformat[ii, 44] + 0.1752) / 0.6934

# In[ ]:


#########################################################################
#                      SAMPLE AND HOLD on RAW DATA
#########################################################################

reformat = SAH(reformat[:, 0:68], sample_and_hold)

# In[ ]:


#########################################################################
#                             DATA COMBINATION
#########################################################################

timestep = 4  # resolution of timesteps, in hours
irow = 0
icustayidlist = sofa_period[:, 0].astype('int64')
reformat2 = np.full((reformat.shape[0], 85), np.nan)  # output array
npt = icustayidlist.size  # number of patients
# Adding 2 empty cols for future shock index=HR/SBP and P/F
reformat = np.insert(reformat, 68, np.nan, axis=1)
reformat = np.insert(reformat, 69, np.nan, axis=1)


for i in range(npt):

    if (i % 100 == 0):
        print(i)

    icustayid = icustayidlist[i]  # 200 to 300Kpython

    # CHARTEVENTS AND LAB VALUES
    temp_id = reformat[reformat[:, 1] == icustayid, :]#Select the subset of data associated with the current patient ID from the reformat array
    ii = (temp_id[:, 2] >= sofa_period[i, 1]) & (temp_id[:, 2] <= sofa_period[i, 2])
    temp = temp_id[ii, :]
    beg = temp[0, 2]  # timestamp of first record

    # Extract the start time, end time and rate of all infusions and injections from inputMV and inputCV.
    iv = np.where((inputMV[:, 0] == icustayid ))[0]# rows of interest in inputMV
    input = inputMV[iv, :]  # subset of interest
    iv = np.where((inputCV[:, 0] == icustayid ))[0]  # rows of interest in inputCV
    input2 = inputCV[iv, :]  # subset of interest
    startt = input[:, 1]  # start of all infusions and boluses
    endt = input[:, 2]  # end of all infusions and boluses
    rate = input[:, 7]  # rate of infusion (is NaN for boluses) || corrected for tonicity

    pread = inputpreadm[inputpreadm[:, 0] == icustayid , 1]  # preadmission volume
    if (pread.size != 0):  # store the value, if available
        totvol = np.nansum(pread)
    else:
        totvol = 0  # if not documented: it's zero

    # compute volume of fluid given before start of record!!!
    t0 = 0
    t1 = beg
    # Total liquid volume cv only has charttime mv has end, starttime so treat them separately

    infu = np.nansum(rate * (endt - startt) * ((endt <= t1) & (startt >= t0)) / 3600 + rate * (endt - t0) * (
            (startt <= t0) & (endt <= t1) & (endt >= t0)) / 3600 + rate * (t1 - startt) * (
                             (startt >= t0) & (endt >= t1) & (startt <= t1)) / 3600 + rate * (t1 - t0) * (
                             (endt >= t1) & (startt <= t0)) / 3600)
    # all boluses received during this timestep, from inputMV (need to check rate is NaN)
    bolus = np.nansum(input[(np.isnan(input[:, 5])) & (input[:, 1] >= t0) & (input[:, 1] <= t1), 6]) + np.nansum(
        input2[(input2[:, 1] >= t0) & (input2[:, 1] <= t1), 4])
    totvol = np.nansum(np.array([totvol, infu, bolus]))

    # VASOPRESSORS
    iv = np.where(vasoMV[:, 0] == icustayid )[0]  # rows of interest in vasoMV
    vaso1 = vasoMV[iv, :]  # subset of interest
    iv = np.where(vasoCV[:, 0] == icustayid )[0]  # rows of interest in vasoCV
    vaso2 = vasoCV[iv, :]  # subset of interest
    startv = vaso1[:, 2]  # start of VP infusion
    endv = vaso1[:, 3]  # end of VP infusions
    ratev = vaso1[:, 4]  # rate of VP infusion

    # DEMOGRAPHICS / gender, age, elixhauser, re-admit, died in hosp?, died within
    # 48h of out_time (likely in ICU or soon after), died within 90d after admission?
    demogi = np.where(demog.loc[:, 'icustay_id'] == icustayid )[0]

    dem = np.array(
        [demog.loc[demogi, 'gender'].item(), demog.loc[demogi, 'age'].item(), (demog.loc[demogi, 'elixhauser']).item(),
         (demog.loc[demogi, 'adm_order'] > 1).item(), demog.loc[demogi, 'morta_hosp'].item(),
         (np.abs(demog.loc[demogi, 'dod'] - demog.loc[demogi, 'outtime']) < (24 * 3600 * 2)).item(),
         demog.loc[demogi, 'morta_90'].item(), (sofa_period[i, 2] - sofa_period[i , 1]) / 3600])

    # URINE OUTPUT
    iu = np.where(UO[:, 0] == icustayid )[0]  # rows of interest in inputMV
    output = UO[iu, :]  # subset of interest
    pread = UOpreadm[UOpreadm[:, 0] == icustayid, 3]  # preadmission UO
    if pread.size != 0:  # store the value, if available
        UOtot = np.nansum(pread)
    else:
        UOtot = 0;

    # adding the volume of urine produced before start of recording!
    UOnow = np.nansum(output[(output[:, 1] >= t0) & (output[:, 1] <= t1), 3])  # t0 and t1 defined above
    UOtot = np.nansum(np.array([UOtot, UOnow]))

    endhour = int((sofa_period[i, 2] - sofa_period[i , 1]) / 3600)


    for j in range(0, endhour , timestep):  # Iterative processing of data within each 4-hour period
        t0 = 3600 * j + beg  # left limit of time window
        t1 = 3600 * (j + timestep) + beg  # right limit of time window
        ii = (temp[:, 2] >= t0) & (temp[:, 2] <= t1)  # index of items in this time period

        if np.sum(ii) > 0:

            # ICUSTAY_ID, OUTCOMES, DEMOGRAPHICS
            reformat2[irow, 0] = (j / timestep) + 1  # 'bloc' = timestep (1,2,3...)
            reformat2[irow, 1] = icustayid  # icustay_ID
            reformat2[irow, 2] = 3600 * j + beg  # t0 = lower limit of time window
            reformat2[irow, 3:11] = dem  # demographics and outcomes

            # CHARTEVENTS and LAB VALUES (+ includes empty cols for shock index and P/F)
            value = temp[ii, :]  # records all values in this timestep

            # #####################   DISCUSS ADDING STUFF HERE / RANGE, MIN, MAX ETC   ################

            if np.sum(ii) == 1:  # if only 1 row of values at this timestep
                reformat2[irow, 11:78] = value[:, 3:]
            else:
                reformat2[irow, 11:78] = np.nanmean(value[:, 3:], axis=0)  # mean of all available values

            # VASOPRESSORS
            # for CV: dose at timestamps.
            # for MV: 4 possibles cases, each one needing a different way to compute the dose of VP actually administered:
            # ----t0---start----end-----t1----
            # ----start---t0----end----t1----
            # -----t0---start---t1---end
            # ----start---t0----t1---end----
            # Calculate the dose of VASOPRESSORS for each of the above four scenarios for each time step.
            # CV only charttime MV end\starttime

            # MV
            v = ((endv >= t0) & (endv <= t1)) | ((startv >= t0) & (endv <= t1)) | ((startv >= t0) & (startv <= t1)) | (
                    (startv <= t0) & (endv >= t1))

            # CV
            v2 = vaso2[(vaso2[:, 2] >= t0) & (vaso2[:, 2] <= t1), 3]

            temp_list = []
            if (ratev[v].size != 0):# Whether the rate of MV is 0
                temp_list.append(ratev[v].reshape(-1, 1))
            if (v2.size != 0):
                temp_list.append(v2.reshape(-1, 1))

            if (len(temp_list) != 0):
                rv = np.vstack(temp_list)
            else:
                rv = np.array([])

            v1 = np.nanmedian(rv)# averaging

            if (rv.size != 0):
                v2 = np.nanmax(rv)# maximise
            else:
                v2 = np.array([])

            if v1.size != 0 and ~np.isnan(v1) and v2.size != 0 and ~np.isnan(v2):
                reformat2[irow, 78] = v1  # median of dose of VP
                reformat2[irow, 79] = v2  # max dose of VP

            # INPUT FLUID
            # input from MV (4 ways to compute)
            infu = np.nansum(rate * (endt - startt) * ((endt <= t1) & (startt >= t0)) / 3600 + rate * (endt - t0) * (
                    (startt <= t0) & (endt <= t1) & (endt >= t0)) / 3600 + rate * (t1 - startt) * (
                                     (startt >= t0) & (endt >= t1) & (startt <= t1)) / 3600 + rate * (t1 - t0) * (
                                     (endt >= t1) & (startt <= t0)) / 3600)
            # all boluses received during this timestep, from inputMV (need to check rate is NaN) and inputCV (simpler):
            bolus = np.nansum(
                input[(np.isnan(input[:, 5])) & (input[:, 1] >= t0) & (input[:, 1] <= t1), 6]) + np.nansum(
                input2[(input2[:, 1] >= t0) & (input2[:, 1] <= t1), 4])

            # sum fluid given
            totvol = np.nansum(np.array([totvol, infu, bolus]))

            reformat2[irow, 80] = totvol  # total fluid given
            reformat2[irow, 81] = np.nansum(np.array([infu, bolus]))  # fluid given at this step

            # UO
            UOnow = np.nansum(output[(output[:, 1] >= t0) & (output[:, 1] <= t1), 3])
            UOtot = np.nansum(np.array([UOtot, UOnow]))
            reformat2[irow, 82] = UOtot  # total UO
            reformat2[irow, 83] = np.nansum(UOnow)  # UO at this step

            # CUMULATED BALANCE
            reformat2[irow, 84] = totvol - UOtot  # cumulated balance

            irow = irow + 1

reformat2 = reformat2[:irow, :]

# In[ ]:


# ########################################################################
#    CONVERT TO TABLE AND DELETE VARIABLES WITH EXCESSIVE MISSINGNESS
# ########################################################################

# dataheaders
dataheaders = sample_and_hold[0, :].tolist() + ['Shock_Index', 'PaO2_FiO2']
dataheaders = ['bloc', 'icustayid', 'charttime', 'gender', 'age', 'elixhauser', 're_admission', 'died_in_hosp',
               'died_within_48h_of_out_time', 'mortality_90d',
               'delay_end_of_record_and_discharge_or_death'] + dataheaders
dataheaders = dataheaders + ['median_dose_vaso', 'max_dose_vaso', 'input_total', 'input_4hourly', 'output_total',
                             'output_4hourly', 'cumulated_balance'] #Adding a Table Header

reformat2t = pd.DataFrame(reformat2.copy(), columns=dataheaders)
miss = (np.sum(np.isnan(reformat2), axis=0) / reformat2.shape[0])

# Deletion of excessive missing data
# if values have less than 70% missing values (over 30% of values present): I keep them
reformat3t = reformat2t.iloc[:, np.hstack([np.full(11, True), (miss[11:74] < 0.70), np.full(11, True)])]


# In[ ]:


#########################################################################
#             HANDLING OF MISSING VALUES  &  CREATE REFORMAT4T
#########################################################################


def fixgaps(x):
    # FIXGAPS Linearly interpolates gaps in a time series
    # YOUT=FIXGAPS(YIN) linearly interpolates over NaN
    # in the input time series (may be complex), but ignores
    # trailing and leading NaN.
    # R. Pawlowicz 6/Nov/99
    # Linear interpolation in time series to fill in NaN values

    y = x
    bd = np.isnan(x)
    gd = np.where(~bd)[0]

    bd[0:min(gd)] = 0
    bd[max(gd) + 1:] = 0

    y[bd] = interp1d(gd, x[gd])(np.where(bd)[0]) # Linear interpolation between non-NaN values

    return y


# K=1
# distance = seuclidean
# Reference: matlab's knnimpute.m code
# Reference: https://github.com/ogeidix/kddcup09/blob/master/utilities/knnimpute.m

# Implementation of K-Nearest Neighbors Imputation (K-Nearest Neighbors)
def knnimpute(data):
    K = 1
    userWeights = False
    useWMean = True
    # create a copy of data for output
    imputed = data.copy()

    # identify missing vals
    nanVals = np.isnan(data)

    # use rows without nans for calculation of nearest neighbors
    noNans = (np.sum(nanVals, axis=1) == 0)
    dataNoNans = data[noNans, :]

    distances = pdist(np.transpose(dataNoNans), 'seuclidean')

    SqF = squareform(distances)

    temp = SqF - np.identity(SqF.shape[0])

    dists = np.transpose(np.sort(temp))

    ndx = np.transpose(np.argsort(temp, kind='stable'))

    equalDists = np.vstack([np.diff(dists[1:, :], axis=0) == 0.0, np.full(dists.shape[1], False)])

    rows = np.where(np.transpose(nanVals))[1]
    cols = np.where(np.transpose(nanVals))[0]

    for count in range(rows.size):
        for nearest in range(1, ndx.shape[0] - K + 1):
            L = np.where(equalDists[nearest + K - 2:, cols[count]] == 0)[0][0]
            dataVals = data[rows[count], ndx[nearest:nearest + K + L, cols[count]]]
            if (useWMean):
                if (~userWeights):
                    weights = 1 / dists[1:K + L + 1, cols[count]]
                val = wnanmean(dataVals, weights)
            if (~np.isnan(val)):
                imputed[rows[count], cols[count]] = val
                break
    return imputed

# Functions for weighted averages Weighted averages of data with missing values
def wnanmean(x, weights):
    x = x.copy()
    weights = weights.copy()
    nans = np.isnan(x)
    infs = np.isinf(weights)

    if all(nans):
        return np.nan
    if any(infs):
        return np.nanmean(x[infs])

    x[nans] = 0
    weights[nans] = 0
    weights = weights / np.sum(weights)
    return np.dot(np.transpose(weights), x)


# Do linear interpol where missingness is low (kNN imputation doesnt work if all rows have missing values)
# In cases where the proportion of missing values in the data is low (less than 5 per cent), fill in using linear interpolation

reformat3 = reformat3t.values.copy()
miss = (np.sum(np.isnan(reformat3), axis=0) / reformat3.shape[0])

ii = (miss > 0) & (miss < 0.05)  # less than 5% missingness

mechventcol = reformat3t.columns.get_loc('mechvent')

for i in range(10, mechventcol):
    if ii[i] == 1:
        reformat3[:, i] = fixgaps(reformat3[:, i])

reformat3t.iloc[:, 10:mechventcol] = reformat3[:, 10:mechventcol]

# KNN IMPUTATION -  Done on chunks of 10K records.

ref = reformat3[:, 10:mechventcol].copy()  # columns of interest

for i in range(0, (reformat3.shape[0] - 9999), 10000):  # dataset divided in 5K rows chunks (otherwise too large)
    print(i)
    ref[i:i + 10000, :] = np.transpose(knnimpute(np.transpose(ref[i:i + 10000, :])))

ref[-10000:, :] = np.transpose(
    knnimpute(np.transpose(ref[-10000:, :])))  # the last bit is imputed from the last 10K rows

# I paste the data interpolated, but not the demographics and the treatments
reformat3t.iloc[:, 10:mechventcol] = ref

reformat4t = reformat3t.copy()
reformat4 = reformat4t.values.copy()

# In[ ]:


##########################################################################
#        COMPUTE SOME DERIVED VARIABLES: P/F, Shock Index, SOFA, SIRS...
##########################################################################


# CORRECT GENDER Correction of gender information previously indicated by 1, 2
reformat4t.loc[:, 'gender'] = reformat4t.loc[:, 'gender'] - 1

# CORRECT AGE > 200  Set age over 150 years to 91.4 years。
ii = reformat4t.loc[:, 'age'] > 150 * 365.25
reformat4t.loc[ii, 'age'] = 91.4 * 365.25

# FIX MECHVENT sets the missing value to 0 (no mechanical ventilation) and the non-missing value to 1 (mechanical ventilation was carried out)
ii = np.isnan(reformat4t.loc[:, 'mechvent'])
reformat4t.loc[ii, 'mechvent'] = 0
ii = reformat4t.loc[:, 'mechvent'] > 0
reformat4t.loc[ii, 'mechvent'] = 1

# FIX Elixhauser missing values
ii = np.isnan(reformat4t.loc[:, 'elixhauser'])
reformat4t.loc[ii, 'elixhauser'] = np.nanmedian(
    reformat4t.loc[:, 'elixhauser'])  # use the median value / only a few missing data points

# vasopressors / no
a = reformat4t.columns.get_loc('median_dose_vaso')
ii = np.isnan(reformat4[:, a])
reformat4t.loc[ii, 'median_dose_vaso'] = np.zeros((np.sum(ii)))
a = reformat4t.columns.get_loc('max_dose_vaso')
ii = np.isnan(reformat4[:, a])
reformat4t.loc[ii, 'max_dose_vaso'] = np.zeros((np.sum(ii)))

# re-compute P/F with no missing values...
p = reformat4t.columns.get_loc('paO2')
f = reformat4t.columns.get_loc('FiO2_1')
reformat4t.loc[:, 'PaO2_FiO2'] = reformat4[:, p] / reformat4[:, f]

# recompute SHOCK INDEX without NAN and INF
p = reformat4t.columns.get_loc('HR')
f = reformat4t.columns.get_loc('SysBP')
a = reformat4t.columns.get_loc('Shock_Index')
reformat4[:, a] = reformat4[:, p] / reformat4[:, f]
ii = np.isinf(reformat4[:, a])
reformat4[ii, a] = np.nan # Replace inf with nan
d = np.nanmean(reformat4[:, a])
ii = np.isnan(reformat4[:, a])
reformat4[ii, a] = d  # replace NaN with average value ~ 0.8
reformat4t.loc[:, 'Shock_Index'] = reformat4[:, a]

# SOFA - at each timepoint
# need (in this order):  P/F  MV  PLT  TOT_BILI  MAP  NORAD(max)  GCS  CR  UO

s = reformat4t.loc[:, ['PaO2_FiO2', 'Platelets_count', 'Total_bili', 'MeanBP', 'max_dose_vaso', 'GCS', 'Creatinine',
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

nrcol = reformat4.shape[1]  # nr of variables in data
reformat4 = np.hstack([reformat4, np.zeros((reformat4.shape[0], 7))])

for i in range(0, reformat4.shape[0]):
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
        reformat4[i, nrcol:nrcol + 7] = np.array([max(p_s1), max(p_s2), max(p_s3), max(p_s4), max(p_s5), max(p_s6), t]) #Store the maximum score for each organ system as well as the total score in a new column in the data matrix.

# SIRS - at each timepoint |  need: temp HR RR PaCO2 WBC
s = reformat4t.loc[:, ['Temp_C', 'HR', 'RR', 'paCO2', 'WBC_count']].values  # indices of vars used in SOFA

s1 = np.transpose(
    np.array([(s[:, 0] >= 38) | (s[:, 0] <= 36)], dtype=np.int64))  # count of points for all criteria of SIRS
s2 = np.transpose(np.array([s[:, 1] > 90], dtype=np.int64))
s3 = np.transpose(np.array([(s[:, 2] >= 20) | (s[:, 3] <= 32)], dtype=np.int64))
s4 = np.transpose(np.array([(s[:, 4] >= 12) | (s[:, 4] < 4)], dtype=np.int64))

reformat4 = np.insert(reformat4, nrcol + 7, 0, axis=1)
reformat4[:, nrcol + 7] = np.transpose(s1 + s2 + s3 + s4)#A new column is inserted after the penultimate column to store the SIRS scores

# adds 2 cols for SOFA and SIRS, if necessary
if 'SIRS' not in reformat4t.columns:
    reformat4t['SOFA'] = 0
    reformat4t['SIRS'] = 0

# records values
reformat4t.loc[:, 'SOFA'] = reformat4[:, -2]
reformat4t.loc[:, 'SIRS'] = reformat4[:, -1]

# In[ ]:


#########################################################################
#                            EXCLUSION OF SOME PATIENTS
#########################################################################

print(np.unique(reformat4t.loc[:, 'icustayid']).size)  # count before

# check for patients with extreme UO = outliers = to be deleted (>40 litres of UO per 4h!!)
a = np.where(reformat4t.loc[:, 'output_4hourly'] > 12000)[0]
i = np.unique(reformat4t.loc[a, 'icustayid'])
i = np.where(np.isin(reformat4t.loc[:, 'icustayid'], i))[0]
reformat4t.drop(i, inplace=True)
reformat4t.reset_index(inplace=True, drop=True)

# some have bili = 999999
a = np.where(reformat4t.loc[:, 'Total_bili'] > 10000)[0]
i = np.unique(reformat4t.loc[a, 'icustayid'])
i = np.where(np.isin(reformat4t.loc[:, 'icustayid'], i))[0]
reformat4t.drop(i, inplace=True)
reformat4t.reset_index(inplace=True, drop=True)

# check for patients with extreme INTAKE = outliers = to be deleted (>10 litres of intake per 4h!!)
a = np.where(reformat4t.loc[:, 'input_4hourly'] > 10000)[0]
i = np.unique(reformat4t.loc[a, 'icustayid'])  # 28 ids
i = np.where(np.isin(reformat4t.loc[:, 'icustayid'], i))[0]
reformat4t.drop(i, inplace=True)
reformat4t.reset_index(inplace=True, drop=True)

# #### exclude early deaths from possible withdrawals ####
# stats per patient

q = reformat4t.loc[:, 'bloc'] == 1

# fence_posts=find(q(:,1)==1);
# Perform grouping operations Calculate the mortality count and maximum value for each ID
num_of_trials = np.unique(reformat4t.loc[:, 'icustayid']).size  # size(fence_posts,1)
a = reformat4t.loc[:, ['icustayid', 'mortality_90d', 'max_dose_vaso', 'SOFA']]
a.columns = ['id', 'mortality_90d', 'vaso', 'sofa']
cnt = a.groupby('id').count()
d = a.groupby('id').max()
d['GroupCount'] = cnt['mortality_90d']
d.reset_index(inplace=True)
d.columns = ['id', 'max_mortality_90d', 'max_vaso', 'max_sofa', 'GroupCount']
d = d[['id', 'GroupCount', 'max_mortality_90d', 'max_vaso', 'max_sofa']]

# finds patients who match our criteria
e = np.zeros([num_of_trials])

# If the patient's maximum 90-day mortality rate is 1, the maximum dose of vasopressor is 0.
# Or the maximum dose of vasopressor is greater than 0.3. Or the patient's SOFA score is greater than or equal to half the maximum SOFA score.
# Delete if match
for i in range(num_of_trials):
    if d.loc[i, 'max_mortality_90d'] == 1:
        ii = (reformat4t['icustayid'] == d.loc[i, 'id']) & (
                reformat4t['bloc'] == d.loc[i, 'GroupCount'])  # last row for this patient
        e[i] = sum((reformat4t.loc[ii, 'max_dose_vaso'] == 0) & (d.loc[i, 'max_vaso'] > 0.3) & (
                reformat4t.loc[ii, 'SOFA'] >= d.loc[i, 'max_sofa'] / 2)) > 0

r = d.loc[(e == 1) & (d['GroupCount'] < 20), 'id']  # ids to be removed
ii = np.where(np.isin(reformat4t['icustayid'], r))[0]
reformat4t.drop(ii, inplace=True)
reformat4t.reset_index(inplace=True, drop=True)

# exclude patients who died in ICU during data collection period
# Patient's first ICU stay (bloc column is 1).
# Patient died within 48 hours of ICU discharge (died_within_48h_of_out_time column is 1).
# Patient died with a delay of less than 24 hours between the time of death and the time of discharge or death at the end of the data record.
# Delete
ii = (reformat4t['bloc'] == 1) & (reformat4t['died_within_48h_of_out_time'] == 1) & (
        reformat4t['delay_end_of_record_and_discharge_or_death'] < 24)
ii = np.where(np.isin(icustayidlist, reformat4t.loc[ii, 'icustayid']))[0]
reformat4t.drop(ii, inplace=True)
reformat4t.reset_index(inplace=True, drop=True)



# ii =

# In[ ]:


#########################################################################
#                     CREATE FINAL MIMIC_TABLE
#########################################################################
dataheaders5 = ['bloc','icustayid','charttime','gender','age','elixhauser','re_admission', 'died_in_hosp', 'died_within_48h_of_out_time','mortality_90d','delay_end_of_record_and_discharge_or_death',    'Weight_kg','GCS','HR','SysBP','MeanBP','DiaBP','RR','SpO2','Temp_C','FiO2_1','Potassium','Sodium','Chloride','Glucose',    'BUN','Creatinine','Magnesium','Calcium','Ionised_Ca','CO2_mEqL','SGOT','SGPT','Total_bili','Albumin','Hb','WBC_count','Platelets_count','PTT','PT','INR',    'Arterial_pH','paO2','paCO2','Arterial_BE','HCO3','Arterial_lactate','mechvent','Shock_Index','PaO2_FiO2',    'median_dose_vaso','max_dose_vaso','input_total','input_4hourly','output_total','output_4hourly','cumulated_balance','SOFA','SIRS']
ii=np.where(np.isin(reformat4t.columns,dataheaders5))[0]
MIMICtable = reformat4t.iloc[:,ii]
MIMICtable.to_csv('/RL_safety_models/data/mimictable.csv',index=False,na_rep='NaN')
with open('/RL_safety_models/data/step_4_start_choose.pkl', 'wb') as file:
    pickle.dump(MIMICtable, file)
