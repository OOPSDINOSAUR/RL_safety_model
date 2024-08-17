# importing
import sys
import os
from pathlib import Path

sys.path.append(os.getcwd())
path = Path(os.getcwd())
sys.path.append(str(path.parent.absolute()))

import pandas as pd
import numpy as np
from functools import partial
from utils.compute_trajectories_utils import compute_reward, build_trajectories, compute_apache2
from constants import DATA_FOLDER_PATH
from copy import deepcopy
from tqdm import tqdm
from datetime import datetime
from datetime import timedelta
import time

STATE_SPACE = ['gender', 'mechvent', 're_admission', 'age', 'Weight_kg', 'GCS', 'HR', 'SysBP', 'MeanBP', 'DiaBP', 'RR',
               'Temp_C', 'FiO2_1', 'Potassium', 'Sodium',
               'Chloride', 'Glucose', 'Magnesium', 'Calcium', 'Hb', 'WBC_count', 'Platelets_count', 'PTT', 'PT',
               'Arterial_pH',
               'paO2', 'paCO2', 'Arterial_BE', 'HCO3', 'SOFA', 'SIRS', 'Shock_Index', 'PaO2_FiO2', 'cumulated_balance',
               'SpO2',
               'BUN', 'Creatinine', 'SGOT', 'SGPT', 'Total_bili', 'INR', 'input_total', 'output_total',
               'output_4hourly']

IMPUTED_DATAFRAME_PATH = os.path.join(DATA_FOLDER_PATH, "imputed.pkl")
# ACTION_DICT IS SHARED WITH discretize_actions.py
ACTION_DICT = {
    'input_4hourly': [0, 50, 180, 530, 1000000000],
    'max_dose_vaso': [0, 0.08, 0.22, 0.45, 1000000000]
}
df = pd.read_pickle(IMPUTED_DATAFRAME_PATH)

# Change gender to numerical data
# print("Transforming gender into numerical data...")
# df['gender'] = df['gender'].apply(lambda x: 1 if x == 'F' else 0)

# Compute reward
print("Computing intermediate reward...")
df = df.sort_values(by=['icustayid', 'charttime'])
df['icustay_id_shifted_up'] = df['icustayid'].shift(-1)
df['icustay_id_shifted_down'] = df['icustayid'].shift(1)
df['apache2'] = df.apply(compute_apache2, axis=1)
df['apache2_shifted_up'] = df['apache2'].shift(-1)

print(list(filter(lambda x: x in list(df.columns), STATE_SPACE)))

# Compute trajectories
MDP_dataset_dict = build_trajectories(
    df,
    state_space=list(filter(lambda x: x in list(df.columns), STATE_SPACE)),
    # Remove from state space columns that were removed by imputation
    action_space=list(ACTION_DICT.keys())
)

DATA_DIR_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data")

make_dir = ['states', 'rewards', 'actions', 'done_flags']

for directory in make_dir:
    if not os.path.exists(os.path.join(DATA_FOLDER_PATH, directory)):
        os.makedirs(os.path.join(DATA_FOLDER_PATH, directory))

print("Building of trajectories completed")
np.save(os.path.join(DATA_FOLDER_PATH, "states/raw_states.npy"), MDP_dataset_dict["states"])
np.save(os.path.join(DATA_FOLDER_PATH, "rewards/rewards_with_intermediate_fixed.npy"), MDP_dataset_dict["rewards"])
np.save(os.path.join(DATA_FOLDER_PATH, "actions/2dactions_not_binned.npy"), MDP_dataset_dict["actions"])
np.save(os.path.join(DATA_FOLDER_PATH, "done_flags/done_flags.npy"), MDP_dataset_dict["done_flags"])
print("Saved data to files in data folder")
