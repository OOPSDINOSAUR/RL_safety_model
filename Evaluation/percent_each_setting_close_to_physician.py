import sys
import os
from pathlib import Path

sys.path.append(os.getcwd())
path = Path(os.getcwd())
sys.path.append(str(path.parent.absolute()))

import itertools
from d3rlpy.dataset import MDPDataset
from d3rlpy.algos import DiscreteCQL, DoubleDQN
from d3rlpy.ope import DiscreteFQE
import numpy as np
import matplotlib.pyplot as plt
import pickle
from utils.load_utils import load_data
from estimator import get_final_estimator
from constants import IMAGES_PATH

print("Making physician action.py similarity plot")
NUM_RUNS = 5
total_num_close_cql__ApacheII = {
    "input_4hourly": 0,
    "max_dose_vaso": 0,
    }
total_num_close_cql = {
    "input_4hourly": 0,
    "max_dose_vaso": 0,
    }
total_num_close_ddqn__ApacheII = {
    "input_4hourly": 0,
    "max_dose_vaso": 0,
    }
total_num_close_ddqn = {
    "input_4hourly": 0,
    "max_dose_vaso": 0,
    }

for i in range(NUM_RUNS):
    print(f"Processing run {i}")
    estimator = get_final_estimator(DiscreteCQL, "raw", "intermediate", index_of_split = i)
    for key in total_num_close_cql__ApacheII:
        total_num_close_cql__ApacheII[key] += estimator.get_actions_within_one_bin()[key]

    estimator = get_final_estimator(DiscreteCQL, "raw", "no_intermediate", index_of_split=i)
    for key in total_num_close_cql:
        total_num_close_cql[key] += estimator.get_actions_within_one_bin()[key]

    estimator = get_final_estimator(DoubleDQN, "raw", "intermediate", index_of_split=i)
    for key in total_num_close_ddqn__ApacheII:
        total_num_close_ddqn__ApacheII[key] += estimator.get_actions_within_one_bin()[key]

    estimator = get_final_estimator(DoubleDQN, "raw", "no_intermediate", index_of_split = i)
    for key in total_num_close_ddqn:
        total_num_close_ddqn[key] += estimator.get_actions_within_one_bin()[key]

for key in total_num_close_cql__ApacheII:
    total_num_close_cql__ApacheII[key] /= NUM_RUNS
    total_num_close_ddqn[key] /= NUM_RUNS
    total_num_close_cql[key] /= NUM_RUNS
    total_num_close_ddqn__ApacheII[key] /= NUM_RUNS

fig, ax = plt.subplots()
ind = np.arange(2)
width=0.2
key_list = list(total_num_close_cql__ApacheII.keys())

ax.bar(ind-width*1.5, [total_num_close_cql__ApacheII[key] * 100 for key in key_list], width, color='#1d3a89', label="CQL_ApacheII")
ax.bar(ind-width/2, [total_num_close_cql[key] * 100 for key in key_list], width, color='#538fdf', label="CQL")
ax.bar(ind+width*1.5, [total_num_close_ddqn[key] * 100 for key in key_list], width, color='#aad0e3', label="DDQN")
ax.bar(ind+width/2, [total_num_close_ddqn__ApacheII[key] * 100 for key in key_list], width, color='#f5d5a2', label="DDQN_ApacheII")

ax.set_xticks(ind)
ax.set_xticklabels(key_list, fontsize=20)
ax.set_ylim(0,100)
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)
ax.set_ylabel("%", fontsize=14)
ax.legend()
ax.set_title("% of each setting within one bin of physician", fontsize=18)
fig.savefig(f'{IMAGES_PATH}/of_each_setting_within_one_bin_of_physician.png')
plt.show()