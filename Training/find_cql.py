# Find best hyperparameters for CQL

from d3rlpy.algos import DiscreteCQL

from train_utils import train
from constants import HYPERPARAMETER_SEARCH_PATH
import multiprocessing
N_STEPS_PER_EPOCH = 50000
N_EPOCHS = 10

# 每个超参数的取值范围
gammas = [0.25, 0.5, 0.75, 0.9, 0.99]
alphas = [0.05, 0.1, 0.5, 1, 2]
learning_rates = [1e-07, 1e-06, 1e-05, 1e-04]

def train_model(learning_rate, gamma, alpha):
    train(
        model_class=DiscreteCQL,
        learning_rate=learning_rate,
        gamma=gamma,
        alpha=alpha,
        states="raw",
        rewards="no_intermediate",
        index_of_split=None,
        label=f"{HYPERPARAMETER_SEARCH_PATH}/lr={learning_rate}, gamma={gamma}, alpha={alpha}",
        n_epochs=10,
        n_steps_per_epoch=50000
    )

if __name__ == "__main__":
    # 获取CPU核数的一半
    num_cpus = multiprocessing.cpu_count()
    num_processes = num_cpus // 2

    # 用于存储所有进程的列表
    processes = []

    # 创建一个信号量，限制同时运行的进程数量
    semaphore = multiprocessing.Semaphore(num_processes)

    def wrapper(learning_rate, gamma, alpha):
        with semaphore:
            train_model(learning_rate, gamma, alpha)

    # 为每种超参数组合创建一个进程
    for gamma in gammas:
        for alpha in alphas:
            for learning_rate in learning_rates:
                p = multiprocessing.Process(target=wrapper, args=(learning_rate, gamma, alpha))
                processes.append(p)
                p.start()

    # 确保所有进程都完成执行
    for process in processes:
        process.join()