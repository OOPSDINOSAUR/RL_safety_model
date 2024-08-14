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

# Extract all tables of mysql database

mydb = mysql.connector.connect(
  host="localhost",
  user="",
  password="",
)
mycursor = mydb.cursor()

TABLES = ["CE_0_10000","CE_10000_20000","CE_20000_30000","CE_30000_40000","CE_40000_50000","CE_50000_60000",
     "CE_60000_70000","CE_70000_80000","CE_80000_90000","CE_90000_100000","DEMOG","LABS","MECHVENT",
     "PREADM_FLUID","PREADM_UO","SOFA_PERIOD_ADULT","UO","VASO_CV","VASO_MV","FLUID_CV","FLUID_MV"]

for i in TABLES:
    query = "SELECT * FROM mimic3_sepsis_final." + i
    mycursor.execute(query)
    result = mycursor.fetchall()
    columns = [i[0] for i in mycursor.description]
    df = pd.DataFrame(result, columns=columns)
    df.to_csv('/RL-safety-models/data_sepsis/' + i + '.csv', index=False)