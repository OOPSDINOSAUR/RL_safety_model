import sys
import os
from pathlib import Path

sys.path.append(os.getcwd())
path = Path(os.getcwd())
sys.path.append(str(path.parent.absolute()))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from utils.load_utils import load_data
from estimator import get_final_estimator
from constants import FINAL_POLICIES_PATH, DATA_FOLDER_PATH, IMAGES_PATH
from collections import defaultdict
from d3rlpy.dataset import MDPDataset
from d3rlpy.algos import DiscreteCQL, DoubleDQN
from d3rlpy.ope import FQE
from matplotlib import cm

# action.py dictionairy with all bins as right ends
ACTION_DICT = {
    'input_4hourly': [0, 50, 180, 530, 1000000000],
    'max_dose_vaso': [0, 0.08, 0.22, 0.45, 1000000000]
}

print("Making action.py distribution plot")
# TODO: Fill this in with the policies you are comparing and their paths each algorithm should have the same
# number of policies
POLICIES = {
    'Our_Model': [
        [f'{FINAL_POLICIES_PATH}/raw_intermediate/CQL/run_0/model_2000000.pt', DiscreteCQL()],
        [f'{FINAL_POLICIES_PATH}/raw_intermediate/CQL/run_1/model_2000000.pt', DiscreteCQL()],
        [f'{FINAL_POLICIES_PATH}/raw_intermediate/CQL/run_2/model_2000000.pt', DiscreteCQL()],
        [f'{FINAL_POLICIES_PATH}/raw_intermediate/CQL/run_3/model_2000000.pt', DiscreteCQL()],
        [f'{FINAL_POLICIES_PATH}/raw_intermediate/CQL/run_4/model_2000000.pt', DiscreteCQL()]
    ],
    'CQL': [
        [f'{FINAL_POLICIES_PATH}/raw_no_intermediate/CQL/run_0/model_2000000.pt', DiscreteCQL()],
        [f'{FINAL_POLICIES_PATH}/raw_no_intermediate/CQL/run_1/model_2000000.pt', DiscreteCQL()],
        [f'{FINAL_POLICIES_PATH}/raw_no_intermediate/CQL/run_2/model_2000000.pt', DiscreteCQL()],
        [f'{FINAL_POLICIES_PATH}/raw_no_intermediate/CQL/run_3/model_2000000.pt', DiscreteCQL()],
        [f'{FINAL_POLICIES_PATH}/raw_no_intermediate/CQL/run_4/model_2000000.pt', DiscreteCQL()]
    ],
    'DDQN_Apache II': [
        [f'{FINAL_POLICIES_PATH}/raw_intermediate/DQN/run_0/model_2000000.pt', DoubleDQN()],
        [f'{FINAL_POLICIES_PATH}/raw_intermediate/DQN/run_1/model_2000000.pt', DoubleDQN()],
        [f'{FINAL_POLICIES_PATH}/raw_intermediate/DQN/run_2/model_2000000.pt', DoubleDQN()],
        [f'{FINAL_POLICIES_PATH}/raw_intermediate/DQN/run_3/model_2000000.pt', DoubleDQN()],
        [f'{FINAL_POLICIES_PATH}/raw_intermediate/DQN/run_4/model_2000000.pt', DoubleDQN()]
    ],
    'DDQN': [
        [f'{FINAL_POLICIES_PATH}/raw_no_intermediate/DQN/run_0/model_2000000.pt', DoubleDQN()],
        [f'{FINAL_POLICIES_PATH}/raw_no_intermediate/DQN/run_1/model_2000000.pt', DoubleDQN()],
        [f'{FINAL_POLICIES_PATH}/raw_no_intermediate/DQN/run_2/model_2000000.pt', DoubleDQN()],
        [f'{FINAL_POLICIES_PATH}/raw_no_intermediate/DQN/run_3/model_2000000.pt', DoubleDQN()],
        [f'{FINAL_POLICIES_PATH}/raw_no_intermediate/DQN/run_4/model_2000000.pt', DoubleDQN()]
    ]
}

for P in POLICIES.keys():
    for i in range(len(POLICIES[P])):
        data = \
            load_data(states='raw', rewards='intermediate' if P == 'Our_Model' else 'no_intermediate',
                      index_of_split=i)[1]
        POLICIES[P][i][1].build_with_dataset(data)
        POLICIES[P][i][1].load_model(POLICIES[P][i][0])

# simply repeat physician actions over from data set
pred_phys = data.actions
for i in range(len(POLICIES['Our_Model']) - 1):
    pred_phys = np.concatenate([pred_phys, data.actions])

PREDS = {}
for P in POLICIES.keys():
    for i in range(len(POLICIES[P])):
        print(f"Processing run {i} for {P}")
        predicted_actions = POLICIES[P][i][1].predict(data.observations)
        if P not in PREDS.keys():
            PREDS[P] = predicted_actions
        else:
            PREDS[P] = np.concatenate([PREDS[P], predicted_actions])
ddqn = PREDS.pop('DDQN')
PREDS['physician'] = pred_phys
PREDS['DDQN'] = ddqn
d = PREDS



# build reverse mapping to get action.py settings chosen from discrete action.py indice
def get_reverse_action_dict():
    # build list of possibilities
    indices_lists = [[ACTION_DICT[action][i] for i in range(len(ACTION_DICT[action]))] for action in ACTION_DICT]
    print(indices_lists)
    possibilities = list(itertools.product(*indices_lists))
    print(possibilities)
    possibility_dict = {}

    for i, possibility in enumerate(possibilities):
        possibility_dict[i] = possibility

    print(possibility_dict)
    return possibility_dict

rad = get_reverse_action_dict()

for i in d.keys():
    d[i] = d[i].tolist()
    for j in range(len(d[i])):
        d[i][j] = rad[d[i][j]]
    d[i] = np.array(d[i])

fig, ax = plt.subplots(len(ACTION_DICT.keys()))
fig.patch.set_alpha(0.0)  # Set the graphic background to be transparent
# Use np.unique to get the unique rows and their counts in each policy
stats = {key: np.unique(value, axis=0, return_counts=True) for key, value in d.items()}

# Creating integer label mappings
def create_label_map(labels):
    label_map = [0] * len(labels)
    label_map[-1] = ">{}".format(labels[-2])
    label_map[0] = "0"

    for i in range(1, len(label_map) - 1):
        label_map[i] = "{}-{}".format(labels[i - 1], labels[i])

    return label_map

x_labels = ACTION_DICT['max_dose_vaso']
y_labels = ACTION_DICT['input_4hourly']

# Create integer label mappings
x_label_map = {label: idx for idx, label in enumerate(x_labels)}
y_label_map = {label: idx for idx, label in enumerate(y_labels)}
x_labels = create_label_map(x_labels)
y_labels = create_label_map(y_labels)

# Plot each strategy separately
for policy, (unique_rows, counts) in stats.items():
    # Creating 3D Graphics
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Prepare data
    xpos = [x_label_map[row[1]] for row in unique_rows]
    ypos = [y_label_map[row[0]] for row in unique_rows]
    zpos = np.zeros_like(counts)
    dx = dy = 0.5
    dz = counts
    # Creating a colour map
    norm = plt.Normalize(dz.min(), dz.max())
    colors = cm.viridis(norm(dz))

    # Plotting bar charts
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, alpha=0.8)

    ax.set_title(f'{policy} policy', fontsize=18)
    ax.set_xlabel('Max_dose_vaso', fontsize=15, labelpad=20)
    ax.set_ylabel('Input_4hourly', fontsize=15, labelpad=20)
    ax.set_zlabel('Number of actions', fontsize=15, labelpad=20)
    ax.set_xticks(np.arange(len(x_label_map)))
    ax.set_yticks(np.arange(len(y_label_map)))
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)
    ax.zaxis.set_tick_params(pad=10)  # Adjustment of the distance between the scale label and the shaft

    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.tick_params(axis='z', labelsize=12)

    sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, aspect=5)
    # Modify the font size of the colourbar
    cbar.ax.tick_params(labelsize=12)  # Change to the appropriate font size
    ax.dist = 11
    plt.tight_layout()

    fig.savefig(f'{IMAGES_PATH}/'+f'{policy}'+'Number of actions.png')

    plt.show()
