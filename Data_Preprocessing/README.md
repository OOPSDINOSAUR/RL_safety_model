# Data_Preprocessing

### Step Zero  
1. You need to get acess MIMIC III dataset  from https://mimic.physionet.org/. To get the dataset, you need to satisfy requirements from the webiste (take an online course and get approval from the manager). The MIMIC dataset is about 6G (compressed). 
2. You need to download the MySQL database and import MIMIC III into the MySQL database using the following code https://github.com/MIT-LCP/mimic-code/tree/main/mimic-iii/buildmimic/mysql/.
3. 'elixhauser-quan.sql' file (from https://github.com/MIT-LCP/mimic-code/blob/master/concepts/comorbidity/elixhauser-quan.sql) was included for your convinience. 

### Step One
Run the files in the Preliminaries, Patient_Selection, and Final_Integration folders in turn(There are corresponding explanations in the respective folders).

### Step Two
1. Compute Trajectories
-Takes imputed data and builds all trajectories with which we train our policy. Saves each state as a 48-tuple (one value for each measurement) and each action as a 2-tuple (one value for each setting we control). Saves reward including intermediate reward.
-This script allows you to build the trajectories for training, including MDP states, intermediate rewards, MDP action space.
2. Modification of actions and rewards
-discretize_actions.py: Transforms continuous 2-dimensional actions into 1-dimensional discrete actions and saves them in a separate file (necessary to have a discrete action space)
-remove_intermediate_reward.py: Removes intermediate rewards from rewards (only keeping terminal rewards) and saves them in a separate file (necessary to be able to train with and without intermediate rewards)
3. Split the data and create OOD
-generate_ood_dataset.py : generate Out of Distribution (OOD) dataset of outlier patients.
-train_test_split.py : Split the training and testing data
```
python3 Data_Preprocessing/run.py
```

