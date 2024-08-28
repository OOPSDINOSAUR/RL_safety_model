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