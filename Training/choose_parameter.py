import pandas as pd

from constants import HYPERPARAMETER_SEARCH_PATH

#cql
gammas = [0.99]
alphas = [0.05, 0.1, 0.5, 1, 2]
learning_rates = [1e-07, 1e-06, 1e-05, 1e-04]
best_parameter = [0,0,0]
loss = 10
for gamma in gammas:
    for alpha in alphas:
        for learning_rate in learning_rates:
            label = f"{HYPERPARAMETER_SEARCH_PATH}/lr={learning_rate}, gamma={gamma}, alpha={alpha}/loss.csv"
            loss_table = pd.read_csv(label,header=None)
            print(gamma,alpha,learning_rate,loss_table.iloc[9,2])
            if loss > loss_table.iloc[9,2]:
                loss = loss_table.iloc[9,2]
                best_parameter = [learning_rate,gamma,alpha]
print("CQL:")
print(best_parameter)

# dqn
best_parameter = [0,0,0]
loss = 10
for gamma in gammas:
    for learning_rate in learning_rates:
        label = f"{HYPERPARAMETER_SEARCH_PATH}/lr={learning_rate}, gamma={gamma}/loss.csv"
        loss_table = pd.read_csv(label,header=None)
        if loss > loss_table.iloc[9,2]:
            loss = loss_table.iloc[9,2]
            best_parameter = [learning_rate,gamma]
print("DQN:")
print(best_parameter)




