# Make Out of Distribution vs. In Distribution value estimation plot for Our_Model and DDQN (Figure 4 in Our_Model paper)

# importing
import sys
import os
from pathlib import Path

sys.path.append(os.getcwd())
path = Path(os.getcwd())
sys.path.append(str(path.parent.absolute()))

from collections import Counter
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
import pickle

from d3rlpy.ope import DiscreteFQE
from d3rlpy.algos import DiscreteCQL, DoubleDQN, DiscreteRandomPolicy
from d3rlpy.dataset import MDPDataset
from estimator import ModelEstimator, PhysicianEstimator, get_final_estimator

from estimator import get_final_estimator
from utils.load_utils import load_data
print("Making OOD vs ID comparison plot")
N_RUNS = 5 # Number of runs in experiment that we are graphing for

# Initialize data dictionary with 2 lists for each model (1 for OOD, 1 for in distribution)
data_dict = {
    "Our_Model" : [[], []],
    "CQL" : [[], []],
    "DDQN" : [[], []],
    "DDQN_Apache II" : [[], []],
    "Physician": [[], []]
}

for i in range(N_RUNS):
    print(f"Processing run {i}")
    # For DDQN and CQL, for each run, get mean of initial value estimations for out of distribution and in distribution using estimator class
    estimator = get_final_estimator(DiscreteCQL, "ood", "intermediate", index_of_split=i)
    data_dict["Our_Model"][0].append(estimator.get_init_value_estimation(estimator.data["train"]).mean())
    data_dict["Our_Model"][1].append(estimator.get_init_value_estimation(estimator.data["test"]).mean())

    estimator = get_final_estimator(DiscreteCQL, "ood", "ood", index_of_split=i)
    data_dict["CQL"][0].append(estimator.get_init_value_estimation(estimator.data["train"]).mean())
    data_dict["CQL"][1].append(estimator.get_init_value_estimation(estimator.data["test"]).mean())

    estimator = get_final_estimator(DoubleDQN, "ood", "ood", index_of_split=i)
    data_dict["DDQN"][0].append(estimator.get_init_value_estimation(estimator.data["train"]).mean())
    data_dict["DDQN"][1].append(estimator.get_init_value_estimation(estimator.data["test"]).mean())

    estimator = get_final_estimator(DoubleDQN, "ood", "intermediate", index_of_split=i)
    data_dict["DDQN_Apache II"][0].append(estimator.get_init_value_estimation(estimator.data["train"]).mean())
    data_dict["DDQN_Apache II"][1].append(estimator.get_init_value_estimation(estimator.data["test"]).mean())

    train_data, test_data = load_data("ood", "ood", index_of_split=i)
    estimator = PhysicianEstimator([train_data, test_data])
    data_dict["Physician"][0].append(estimator.get_init_value_estimation(estimator.data["train"]).mean())
    data_dict["Physician"][1].append(estimator.get_init_value_estimation(estimator.data["test"]).mean())

# Transform list of means (one for each run) into numpy array to make it easier to take mean and variance
for key in data_dict:
    for i, _ in enumerate(data_dict[key]):
        data_dict[key][i] = np.array(data_dict[key][i])


# For each model in data_dict, plot mean and variance over all runs 
for i, label in enumerate(data_dict.keys()):
    means = [value.mean() for value in data_dict[label]]
    print(label)
    print(means)
    var = [value.var()  for value in data_dict[label]]
    print(var)