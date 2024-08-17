# importing
import sys
import os
from pathlib import Path
import os
import pandas as pd
import random

sys.path.append(os.getcwd())
path = Path(os.getcwd())
sys.path.append(str(path.parent.absolute()))

from utils.imputation_utils import preprocess_imputation
from constants import DATA_FOLDER_PATH

IMPUTED_DATA_DIR_PATH       = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data")
# IMPUTATION_N                = 1
# IMPUTATION_K                = 3
DATA_TABLE_FILE_NAME        = os.path.join(DATA_FOLDER_PATH,"mimictable_new_id.csv")
IMPUTED_DATAFRAME_PATH      = os.path.join(DATA_FOLDER_PATH,"imputed.pkl")
# Get data into pandas dataframe
MIMICtable = pd.read_csv(DATA_TABLE_FILE_NAME)
# Impute missing values
# df = preprocess_imputation(df, IMPUTATION_N, IMPUTATION_K)
dead_people = MIMICtable.loc[MIMICtable.iloc[:, 9] == 1, MIMICtable.columns[1]]
dead_people_only = dead_people.unique()
random_values = random.sample(dead_people_only.tolist(), 6000)
ii = ~MIMICtable['icustayid'].isin(random_values)
MIMICtable = MIMICtable[ii]
df = MIMICtable.dropna()
df.to_pickle(IMPUTED_DATAFRAME_PATH)
