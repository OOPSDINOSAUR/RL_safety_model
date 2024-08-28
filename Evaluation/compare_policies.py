import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt


sys.path.append(os.getcwd())
path = Path(os.getcwd())
sys.path.append(str(path.parent.absolute()))

# Output average initial value estimations for DeepVent, DeepVent- and physician (mean and variance over 5 runs) 
# (Corresponds to Table 1 in DeepVent paper)
import numpy as np

from d3rlpy.algos import DoubleDQN, DiscreteCQL

from estimator import ModelEstimator, PhysicianEstimator, get_final_estimator
from utils.load_utils import load_data
from constants import IMAGES_PATH

print("Getting initial value estimations for DeepVent, DeepVent- and Physician")
N_RUNS = 5
# Initialize data dictionary with 2 lists for each model (1 for train 1 for test)
data_dict = {
    "CQL_ApacheII" : [[], []],
    "CQL": [[], []],
    "DDQN": [[], []],
    "DDQN_ApacheII" : [[], []],
    "Physician": [[], []]
}

# Get initial value estimations for each model for each run using estimator clas
for i in range(N_RUNS):
    print(f"Processing run {i}")
    estimator = get_final_estimator(DiscreteCQL, "raw", "intermediate", index_of_split=i)
    data_dict["CQL_ApacheII"][0].append(estimator.get_init_value_estimation(estimator.data["train"]).mean())
    data_dict["CQL_ApacheII"][1].append(estimator.get_init_value_estimation(estimator.data["test"]).mean())

    estimator = get_final_estimator(DiscreteCQL, "raw", "no_intermediate", index_of_split=i)
    data_dict["CQL"][0].append(estimator.get_init_value_estimation(estimator.data["train"]).mean())
    data_dict["CQL"][1].append(estimator.get_init_value_estimation(estimator.data["test"]).mean())

    estimator = get_final_estimator(DoubleDQN, "raw", "no_intermediate", index_of_split=i)
    data_dict["DDQN"][0].append(estimator.get_init_value_estimation(estimator.data["train"]).mean())
    data_dict["DDQN"][1].append(estimator.get_init_value_estimation(estimator.data["test"]).mean())

    estimator = get_final_estimator(DoubleDQN, "raw", "intermediate", index_of_split=i)
    data_dict["DDQN_ApacheII"][0].append(estimator.get_init_value_estimation(estimator.data["train"]).mean())
    data_dict["DDQN_ApacheII"][1].append(estimator.get_init_value_estimation(estimator.data["test"]).mean())

    train_data, test_data = load_data("raw", "no_intermediate", index_of_split=i)
    estimator = PhysicianEstimator([train_data, test_data])
    data_dict["Physician"][0].append(estimator.get_init_value_estimation(estimator.data["train"]).mean())
    data_dict["Physician"][1].append(estimator.get_init_value_estimation(estimator.data["test"]).mean())

# Transform all results into numpy arrays
for key in data_dict:
    for i, _ in enumerate(data_dict[key]):
        data_dict[key][i] = np.array(data_dict[key][i])

# Labels for the x-axis
labels = ['1_RUNS', '2_RUNS', '3_RUNS', '4_RUNS', '5_RUNS']
x = np.arange(len(labels))  # the label locations

# Width of the bars
width = 0.15

# Colors for each bar group
colors = {
    'CQL_ApacheII': ['#1d3a89'],
    'CQL': ['#538fdf'],
    'DDQN': ['#aad0e3'],
    'DDQN_ApacheII': ['#f5d5a2'],
    'Physician': ['#e29b63']
}

# Plot each set of data
fig1, ax1 = plt.subplots(figsize=(10, 6))
# Adjust positions for each bar group
for i, (key, value) in enumerate(data_dict.items()):
    ax1.bar(x + i*width, value[1], width, label=f'{key}', color=colors[key])

# Customize the plot
ax1.set_title('Test', fontsize=24)
ax1.set_xlabel('N_RUNS', fontsize=20)
ax1.set_ylabel('Values', fontsize=20)
ax1.set_xticks(x + width * 1.5)
ax1.set_xticklabels(labels)
ax1.legend()
ax1.grid(True)
ax1.tick_params(axis='x', labelsize=16)  # Set the x-axis scale label font size to 12
ax1.tick_params(axis='y', labelsize=16)
# Add a legend and set the font size
# Move the legend below the chart
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),ncol=2,fontsize=16)
fig1.tight_layout()

fig1.savefig(f'{IMAGES_PATH}/CQL_ApacheII_CQL_DDQN_Physician_test.png')

# Display the plot
plt.show()


