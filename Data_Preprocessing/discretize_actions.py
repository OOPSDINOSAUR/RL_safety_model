# Separate each action.py setting into bins (see ACTION_DICT below) and assign an index to each
# possible combination of bins. Save actions encoded using this index.
import sys
import os
from pathlib import Path

sys.path.append(os.getcwd())
path = Path(os.getcwd())
sys.path.append(str(path.parent.absolute()))

import itertools
from functools import partial
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from constants import DATA_FOLDER_PATH

# ACTION_DICT defines a list of discretisation thresholds for each action dimension, where each value represents a discretisation threshold
ACTION_DICT                 = {
                                'input_4hourly': [0, 50, 180, 530, 1000000000],
                                'max_dose_vaso': [0, 0.08, 0.22, 0.45, 1000000000]
                                }

# Define the bin_action function to map continuous action values to the corresponding discrete indexes based on the above list of discretised thresholds
def bin_action(bin_list, value):
  # Bin action.py
  idx = 0
  while value > bin_list[idx] and idx < len(bin_list):
    idx += 1

  return idx

# Load the undiscretised 3-dimensional action data and store it inactions variables
actions = np.load(os.path.join(DATA_FOLDER_PATH, "actions/2dactions_not_binned.npy"))

# An indexed list of all possible combinations of discrete actions is generated. These indexes correspond to discretisation thresholds of different dimensions
# Get all possible combinations of bin indices (i.e. [[0,0,0], [0,0,1],...])
indices_lists = [[i for i in range(len(ACTION_DICT[action]))] for action in ACTION_DICT]
possibilities = list(itertools.product(*indices_lists))
possibility_dict = {}

# Each possible combination of actions is mapped to a unique index
# Create dictionary mapping each possible combination to an index
for i, possibility in enumerate(possibilities):
  possibility_dict[possibility] = i

df = pd.DataFrame(data=actions, columns=["input_4hourly", "max_dose_vaso"])

# Use the previously defined bin_action function to map continuous action values to the corresponding discretised indexes
# Bin actions in dataframe
for action in ACTION_DICT:
    df[action] = df[action].apply(partial(bin_action, ACTION_DICT[action]))

binned_actions_array = df.to_numpy()

# Save the discretised action data as a .npy file
# Save 3D binned actions
np.save(os.path.join(DATA_FOLDER_PATH, "actions/2Ddiscretized_actions.npy"), binned_actions_array)

# Map each discretised 3-dimensional action combination to a corresponding 1-dimensional index and save it in an npy file
int_actions = []
# Get corresponding index for each action.py
for ele in binned_actions_array:
  int_actions.append(possibility_dict[tuple(ele)])

int_actions = np.array(int_actions)

# Save 1D discretized actions
np.save(os.path.join(DATA_FOLDER_PATH, "actions/1Ddiscretized_actions.npy"), int_actions)
