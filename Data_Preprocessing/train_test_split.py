# 分割训练和测试数据
# importing
import sys
import os
from pathlib import Path

sys.path.append(os.getcwd())
path = Path(os.getcwd())
sys.path.append(str(path.parent.absolute()))

from utils.load_utils import load_data
from d3rlpy.dataset import MDPDataset
from constants import DATA_FOLDER_PATH
from sklearn.model_selection import train_test_split
import numpy as np

VALIDATION_SPLIT = 0.2
states = np.load(f"{DATA_FOLDER_PATH}/states/raw_states.npy")
actions = np.load(f"{DATA_FOLDER_PATH}/actions/1Ddiscretized_actions.npy")
rewards = np.load(f"{DATA_FOLDER_PATH}/rewards/rewards_without_intermediate.npy")
terminals = np.load(f"{DATA_FOLDER_PATH}/done_flags/done_flags.npy")

data = MDPDataset(states, actions, rewards, terminals)
# 如果尚不存在，代码会创建一个名为indices的文件夹，用于存储索引数据
if (not os.path.isdir(f"{DATA_FOLDER_PATH}/indices")):
    os.mkdir(f"{DATA_FOLDER_PATH}/indices")

# 在每次迭代中，数据的索引会被随机划分成训练集和测试集，并分别保存到不同的文件中
# Spliting the Training and Testing Data 
def split_data(index):
    indices = np.arange(len(data.episodes))
    train_data_indices, test_data_indices = train_test_split(indices, test_size=VALIDATION_SPLIT) 
    print("Train data length:", len(train_data_indices))
    print("Test data length:", len(test_data_indices))

    np.save(f"{DATA_FOLDER_PATH}/indices/train_indices_{index}.npy", train_data_indices)
    np.save(f"{DATA_FOLDER_PATH}/indices/test_indices_{index}.npy", test_data_indices)

# 对数据集进行5次分割，每次分割生成不同的训练数据和测试数据
for i in range(5):
    split_data(i)
