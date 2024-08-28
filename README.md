# RL_safety_model


## Table of Contents


## About the Project 
In the majority of medical-assisted diagnostic models that have been developed in recent years, the time series data are typically of a fixed length. Nevertheless, this approach is not without limitations, as it is unable to fully capture the dynamic process of patient condition change or enhance the practical application value of the data. In this study, a novel data screening criterion was adopted in order to obtain variable-length time series data from patients with acute and critical illnesses. On this basis, a novel patient state assessment rule was introduced and combined with an offline deep reinforcement learning model based on CQL (Conservative Q-Learning), resulting in a notable enhancement in the model's effectiveness on the variable-length time-series dataset. This is evidenced by the increased prediction accuracy and enhanced robustness observed. Furthermore, the model was validated for safety by comparing the action distribution statistics to ensure its stability and reliability in different situations. This innovative approach not only provides new ideas for handling complex medical data, but also demonstrates the great potential of offline deep reinforcement learning models based on CQL for clinical prediction and decision making. This has broad application prospects and contributes significantly to the advancement of the medical field and the optimisation of clinical practice.

## Installation
1. Go to the parent directory 
```
cd RL_safety_model
```
2. Create and activate virtual environment 
Linux:
```sh
python -m venv env
source env/bin/activate
```
Windows (run this from the command prompt NOT powershell):
```
python -m venv env
.\env\Scripts\activate.bat
```

3. install the required libraries
```
pip install -r requirements.txt 
```
4. install the root package (Run this from the ROOT directory of the repository, i.e. the directory which contains "data_preprocessing", "evaluation", etc.)
```
pip install -e .
```
5. install pytorch with CUDA capabilities

## Processing Data
1. Preliminaries
2. Patient_Selection
3. Final_Integration
4. Compute Trajectories
5. Modify elements
6. Split the data and create OOD

To run the data preprocessing: 
1. Obtain the raw data following the instructions in data_preprocessing/data_extraction folder. You will need to insert your path in the scripts.
2. within the parent directory, run data preprocessing 
```
python3 Data_Preprocessing/run.py
```
Note: A more detailed description for each section of data preprocessing is provided in the data preprocessing folder. 

## Training policies
1. To find the optimal hyperparameters, grid search can be conducted using:  
```
python3 Training/find_cql.py 
python3 Training/find_dqn.py
python3 Training/choose_parameter.py
```
2. Train the policy. Edit the values for the other hyperparameters such as LEARNING_RATE, N_EPOCHS, etc. within the script. The given values are the optimal values that was found in this given problem. The path to the policy weights for each epoch will be output in the console. In the same folder as the policy weights you can find the csv files for all the metrics for the policy at each epoch. 
```
python3 Training/train_eval_loop.py
python3 Training/train_eval_loop_n0.py
```
Note: Apply two GPUs separately.

Running the above scripts will generate ouputs in `d3rlpy_logs` folder. 

3. Then run `python3 Training/get_all_final_policies.py` to get all policies in the correct format for evaluation (modifying `run_num` and `model_num` and `fqe_model_num` to match the parameters you set in step 3 in `Training/train_eval_loop.py`).

## Evaluation

Evaluation directory contains all code necessary to generate graphs and results from DeepVent paper.

1. Basic evaluation

Requires having 5 runs of CQL without intermediate reward, CQL with intermediate reward, DDQN without intermediate reward and DDQN with intermediate reward in the final policies directory defined in `constants.py`.

To run evalutaion script, simply do:
```
cd evaluation
python3 compare_policies.py
python3 percent_each_setting_close_to_physician.py
python3 action_distribution_3d_chart.py
```

2. Evaluation in OOD

Requires 5 runs of CQL without intermediate rewards, CQL with intermediate rewards, DDQN without intermediate rewards, and DDQN with intermediate rewards with OOD as training data in the final policy directory defined in `constants.py`.
```
cd evaluation
python3 compare_ood_id.py
```