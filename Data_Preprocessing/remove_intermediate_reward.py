# 从奖励中删除中间奖励（仅保留最终奖励）并将其保存在单独的文件中
# Remove intermediate rewards from rewards file and save as separate file
import sys
import os
from pathlib import Path

sys.path.append(os.getcwd())
path = Path(os.getcwd())
sys.path.append(str(path.parent.absolute()))
import numpy as np

from constants import DATA_FOLDER_PATH

# 加载包含中间和最终奖励的奖励数据
rewards = np.load(os.path.join(DATA_FOLDER_PATH, "rewards/rewards_with_intermediate_fixed.npy"))

# 定义函数，如果是中间奖励返回0，否则保留最终奖励值（1或-1）
# Return 0 if reward is not 1 or -1 (i.e. the reward is intermediate and not terminal)
def remove_intermediate(num):
    if num != 1 and num != -1:
        return 0
    
    return num

# 使用函数
# For each reward in rewards, set to 0 if isn't terminal reward
new_rewards = np.array([remove_intermediate(num) for num in rewards])

np.save(os.path.join(DATA_FOLDER_PATH, "rewards/rewards_without_intermediate.npy"), new_rewards)