"""
icustay_id（Intensive Care Unit Stay ID）：

‘icustay_id’ is an identifier used to uniquely identify the number of times a patient has been admitted to an intensive care unit (ICU).
In a hospital or healthcare dataset, a patient may be admitted to the ICU multiple times, and each time he or she is admitted to the ICU, he or she is assigned a unique ‘icustay_id’.
The ‘icustay_id’ can be used to track a specific patient's medical records, treatment regimen, monitoring data, etc. while in the ICU.
hadm_id (Hospital Admission ID):

‘hadm_id’ is an identifier used to uniquely identify the number of times a patient has been admitted within a hospital.
A patient may be admitted to the hospital multiple times and each admission is assigned a unique ‘hadm_id’.
‘hadm_id’ can be used to track the number of different hospital admissions a patient has had, to understand information about the treatments, surgeries, medical history, etc. that they have received.

Translated with DeepL.com (free version)
"""
# Constructing state-action-reward sequences for reinforcement learning, and some sequence-related information such as done_flags
import numpy as np
import pandas as pd
from copy import deepcopy
import keras
from sklearn.preprocessing import StandardScaler
from functools import partial
from tqdm import tqdm

# Apache II, a scoring system for assessing a patient's level of critical illness
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
If the icustay_id where the current time step is located is different from the icustay_id of the next time step, i.e., the current time step is the last time step of an icustay (hospitalisation period), then:

If the current time step corresponds to a patient whose hospmort90day (whether he/she died within 90 days of hospitalisation) value is 0 (i.e. did not die), the reward value 1 is returned.
If the patient's hospmort90day value corresponding to the current time step is a non-zero value (died), then the reward value -1 is returned.
Otherwise, if the current time step has the same icustay_id as the next time step, i.e., the current time step is not the last time step of the icustay:

To calculate the reward value, use the following formula:
Reward value = INTERMEDIATE_REWARD_SCALING * (-(apache2 score of the current time step - apache2 score of the previous time step)) / MAX_APACHE_SCORE
where INTERMEDIATE_REWARD_SCALING is the intermediate reward scaling factor, apache2 is a patient reward score, and MAX_APACHE_SCORE is the maximum Apache II score.
"""
#compute the reward
def compute_reward(row):
  if row['icustayid'] != row['icustay_id_shifted_up']: # is last timestep in this icu_stay
    return 1 if row['mortality_90d'] == 0 else -1
  
  else:
    return INTERMEDIATE_REWARD_SCALING * (-(row['apache2'] - row['apache2_shifted_up']))/MAX_APACHE_SCORE

# Calculates the patient's Apache II score based on the Apache II score range. Depending on the value of the different physiological indicators, the function accumulates the scores
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
