"""
icustay_id（Intensive Care Unit Stay ID）：

'icustay_id'是用于唯一标识一个患者在重症监护病房（ICU）内的住院次数的标识符。
在一个医院或医疗数据集中，一个患者可能会多次被送入ICU，每次进入ICU都会被分配一个唯一的'icustay_id'。
'icustay_id'可以用于追踪特定患者在ICU内的医疗记录、治疗方案、监测数据等。
hadm_id（Hospital Admission ID）：

'hadm_id'是用于唯一标识一个患者在医院内的住院次数的标识符。
在医院中，一个患者可能会多次被录取入院，每次录取都会被分配一个唯一的'hadm_id'。
'hadm_id'可以用于跟踪患者的不同住院次数，了解他们接受的治疗、手术、医疗历史等信息。
"""
# 构建用于强化学习的状态-动作-奖励序列，以及一些与序列有关的信息，如完成标志（done_flags）
import numpy as np
import pandas as pd
from copy import deepcopy
import keras
from sklearn.preprocessing import StandardScaler
from functools import partial
from tqdm import tqdm

# Apache II，一种用于评估患者重症程度的评分系统
APACHE_RANGES = {
    "Temp_C": [(41, 4), (39, 3), (38.5, 1), (36, 0), (34, 1), (32, 2), (30, 3), (0, 4)],
    "MeanBP": [(160, 4), (130, 3), (110, 2), (70, 0), (50, 2), (0, 4)],
    "HR": [(180, 4), (140, 3), (110, 2), (70, 0), (55, 2), (40, 3), (0, 4)],
    "Arterial_pH": [(7.7, 4), (7.6, 3), (7.5, 1), (7.33, 0), (7.25, 2), (7.15, 3), (0, 4)],
    "Sodium": [(180, 4), (160, 3), (155, 2), (150, 1), (130, 0), (120, 2), (111, 3), (0, 4)],
    "Potassium": [(7, 4), (6, 3), (5.5, 1), (3.5, 0), (3, 1), (2.5, 2), (0, 4)],
    "Creatinine": [(305, 4), (170, 3), (130, 2), (53, 0), (0, 2)],
    "WBC_count": [(40, 4), (20, 2), (15, 1), (3, 0), (1, 2), (0, 4)],
}

MAX_APACHE_SCORE = sum([l[-1][1] for l in APACHE_RANGES.values()]) + 12 # +12 is for max GCS score
INTERMEDIATE_REWARD_SCALING = 1

"""
如果当前时间步所在的icustay_id与下一个时间步的icustay_id不同，即当前时间步为一个icustay（住院期间）的最后一个时间步，那么：

如果当前时间步对应的患者的hospmort90day（是否在住院后90天内死亡）值为0（即未死亡），则返回奖励值1。
如果当前时间步对应的患者的hospmort90day值为非零值（死亡），则返回奖励值-1。
否则，如果当前时间步与下一个时间步的icustay_id相同，即当前时间步不是icustay的最后一个时间步，那么：

计算奖励值，使用以下公式：
奖励值 = INTERMEDIATE_REWARD_SCALING * (-(当前时间步的apache2分数 - 上一个时间步的apache2分数)) / MAX_APACHE_SCORE
其中，INTERMEDIATE_REWARD_SCALING为中间奖励缩放系数，apache2是一种患者重症评分，MAX_APACHE_SCORE为最大的Apache II评分。
"""
#compute the reward
def compute_reward(row):
  if row['icustayid'] != row['icustay_id_shifted_up']: # is last timestep in this icu_stay
    return 1 if row['mortality_90d'] == 0 else -1
  
  else:
    return INTERMEDIATE_REWARD_SCALING * (-(row['apache2'] - row['apache2_shifted_up']))/MAX_APACHE_SCORE

# 根据Apache II评分范围计算患者的Apache II评分。根据不同的生理指标值，该函数会累加各项得分
def compute_apache2(row):
  score = 0

  for measurement, range_list in APACHE_RANGES.items():
    patient_value = row[measurement]
    for pair in range_list:
      if patient_value >= pair[0]: # If this is the current range for this value
        score += pair[1]
        break

  score += (15 - row["GCS"]) # Different calculation of score for GCS

  return score

def build_trajectories(df,state_space,action_space):
  ''' 
  This assumes that the last timewise entry for a patient is the culmination of
   their episode or 72 hours from first row is culmination and that there exists a reward column
   which is always the reward of that episode

  *************IMPORTANT**********
  I assume that the dataframe has icustay_id, hadm_id, subject_id, and charttime as columns
  and it can be sorted by charttime and end up in chronological order
  I also assume that there is a reward column in the dataset
  ********************************

  param df: the dataframe you want to build trajectories from
  param state_space: the columns of the daataframe you want to include in your state space
  '''
  
  scaler = StandardScaler()
  toscale = df[state_space]
  df[state_space] = scaler.fit_transform(toscale)

  #get the combo of variables that we'll use to distinguish between episodes
  # df['info'] = df.apply(lambda x: tuple(x['icustayid']),axis=1)

  #list of episodes that we'll poulate in the loop below
  episode_states = []
  
  #states prime, encoded list of states, sp[i]=the encoded rerpesentation of the ith state
  actions = []
  rewards = []
  states=[]
  done_flags=[]
  #loop through every unique value of the info tuple we had

  unique_infos = df['icustayid'].unique().tolist()
  print("Building Trajectories...")
  for k in tqdm(range(len(unique_infos)), ncols=100):
    i = unique_infos[k]
    #extract rows of the patient whos info we are iterating on
    episode_rows = df[df['icustayid']==i].sort_values('charttime')
  
    #instantiate the list of states, actions,... where the ith value in the list is the value at the ith timestep
  
    #iterate through rows which are sorted by charttime at the creation of episode_rows
    tdiff = episode_rows.iloc[0]['charttime']
    for row in range(len(episode_rows)):
      end_index = len(episode_rows) - 1
    
      #get the action.py, state, reward, next state, and whether or not the sequence is done in the current timestep
      state = episode_rows[state_space].iloc[row].values.tolist()

      action = episode_rows[action_space].iloc[row].values.tolist()
      
      if row == end_index:
        reward = 1 if episode_rows['mortality_90d'].iloc[row] == 0 else -1
      else:
        apache2 = INTERMEDIATE_REWARD_SCALING * (
          -(episode_rows['apache2'].iloc[row] - episode_rows['apache2_shifted_up'].iloc[row])) / MAX_APACHE_SCORE
        if apache2 > 0:
          reward = apache2 * 1.75
        else:
          reward = apache2
      dflag = 1 if row == end_index else 0

      #add the current time step info to the lists for this episode
      states.append(deepcopy(state))
      actions.append(deepcopy(action))
      rewards.append(deepcopy(reward))
      done_flags.append(deepcopy(dflag)) 
    
    #add to episodes a dictionairy with the mdp info for this episode

  actions = np.array(actions)
  rewards = np.array(rewards)
  done_flags = np.array(done_flags)
  states = np.array(states)
  print(states.mean(axis=1))
  info_dict = {
      "actions":actions,
      "rewards":rewards,
      "done_flags":done_flags,
      "states":states
  }

  # Return list of np arrays of states (one for each episode), because have to pass them through the LSTM autoencoder
  return info_dict
