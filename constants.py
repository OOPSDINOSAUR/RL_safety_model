import os
dir_path = os.path.dirname(os.path.realpath(__file__))
DATA_FOLDER_PATH = f"{dir_path}/data"
SOFA_PATH = f"{dir_path}/data_sofa"
SEPSIS_PATH = f"{dir_path}/data_sepsis"
LOGS_PATH = f"{dir_path}/d3rlpy_logs"
IMAGES_PATH = f"{dir_path}/images"
FINAL_POLICIES_PATH = f"{LOGS_PATH}/FINAL_POLICIES"
TRAINING_RUNS_PATH = f"{LOGS_PATH}/TRAINING_RUNS"
HYPERPARAMETER_SEARCH_PATH = f"{LOGS_PATH}/hyperparameter_search"
REPRODUCED_GRAPHS_PATH = f"{dir_path}/evaluation/reproduced_graphs"