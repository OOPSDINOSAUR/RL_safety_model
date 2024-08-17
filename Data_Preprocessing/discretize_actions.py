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

# ACTION_DICT 定义了各个动作维度的离散化阈值列表，每个值表示一个离散化的阈值
ACTION_DICT                 = {
                                'input_4hourly': [0, 50, 180, 530, 1000000000],
                                'max_dose_vaso': [0, 0.08, 0.22, 0.45, 1000000000]
                                }

# 定义bin_action函数，根据上述离散化阈值列表，将连续的动作值映射到对应的离散索引
def bin_action(bin_list, value):
  # Bin action.py
  idx = 0
  while value > bin_list[idx] and idx < len(bin_list):
    idx += 1

  return idx

# 加载未经离散化的3维动作数据，将其存储actions变量
actions = np.load(os.path.join(DATA_FOLDER_PATH, "actions/2dactions_not_binned.npy"))

#生成了所有可能的离散动作组合的索引列表。这些索引对应于不同维度的离散化阈值
# Get all possible combinations of bin indices (i.e. [[0,0,0], [0,0,1],...])
indices_lists = [[i for i in range(len(ACTION_DICT[action]))] for action in ACTION_DICT]
possibilities = list(itertools.product(*indices_lists))
possibility_dict = {}

# 每种可能的动作组合映射到一个唯一的索引
# Create dictionary mapping each possible combination to an index
for i, possibility in enumerate(possibilities):
  possibility_dict[possibility] = i

df = pd.DataFrame(data=actions, columns=["input_4hourly", "max_dose_vaso"])

# 使用之前定义的bin_action函数将连续动作值映射到对应的离散化索引
# Bin actions in dataframe
for action in ACTION_DICT:
    df[action] = df[action].apply(partial(bin_action, ACTION_DICT[action]))

binned_actions_array = df.to_numpy()

# # 将离散化后的动作数据保存为一个.npy文件
# # Save 3D binned actions
# np.save(os.path.join(DATA_FOLDER_PATH, "actions/2Ddiscretized_actions.npy"), binned_actions_array)
#
# # 将每个离散化的3维动作组合映射为对应的1维索引并保存在npy文件中
# int_actions = []
# # Get corresponding index for each action.py
# for ele in binned_actions_array:
#   int_actions.append(possibility_dict[tuple(ele)])
#
# int_actions = np.array(int_actions)
#
# # Save 1D discretized actions
# np.save(os.path.join(DATA_FOLDER_PATH, "actions/1Ddiscretized_actions.npy"), int_actions)
