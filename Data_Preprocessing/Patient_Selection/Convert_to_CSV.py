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


# Connecting to the database
#Indicators needed to calculate SOFA
mydb = mysql.connector.connect(
  host="",
  user="",
  password="",
  database="mimic3_sepsis_cleaned"
)

#pressurised medicine
mycursor = mydb.cursor()
query = "SELECT ICUSTAY_ID, ITEMID, charttime, rate_std FROM mimic3_sepsis_cleaned.ALL_VASO_CV where rate_std is not null"  # 替换为您要读取的表名和查询条件
mycursor.execute(query)

result = mycursor.fetchall()
columns = [i[0] for i in mycursor.description]
df = pd.DataFrame(result, columns=columns)
df.to_csv(SOFA_PATH + 'VASO_CV.csv', index=False)


query = "SELECT ICUSTAY_ID, ITEMID, starttime,endtime, rate_std FROM mimic3_sepsis_cleaned.ALL_VASO_MV where rate_std is not null"  # 替换为您要读取的表名和查询条件
mycursor.execute(query)

result = mycursor.fetchall()
columns = [i[0] for i in mycursor.description]
df = pd.DataFrame(result, columns=columns)
df.to_csv(SOFA_PATH + 'VASO_MV.csv', index=False)

#Clinical indicators
TABLES = ['PAO2_CELE_CLEANED','FIO2_CLEANED', 'PLATELET_CELE_CLEANED', 'BILIRUBIN_CELE_CLEANED', 'BP_MEAN_CLEANED', 'GCS_CLEANED', 'CREATININE_CELE_CLEANED']
NAMES = ['PaO2','FiO2','Platelet','BILIRUBIN','BP_MEAN','GCS','CREATININE',]
for j in range(len(TABLES)):
    query = "SELECT * FROM mimic3_sepsis_cleaned." + TABLES[j]
    mycursor.execute(query)
    result = mycursor.fetchall()
    columns = [i[0] for i in mycursor.description]
    df = pd.DataFrame(result, columns=columns)
    df.to_csv(SOFA_PATH + NAMES[j] + '.csv', index=False)

#urine
TABLES2 = ['UO','PREADM_UO']
for j in TABLES2:
    query = "SELECT * FROM mimic3_sepsis_final." + j
    mycursor.execute(query)
    result = mycursor.fetchall()
    columns = [i[0] for i in mycursor.description]
    df = pd.DataFrame(result, columns=columns)
    df.to_csv(SOFA_PATH + j + '.csv', index=False)


