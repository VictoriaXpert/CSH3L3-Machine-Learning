import fun
from qlearning import QLearning
import numpy as np

if __name__ == '__main__':
    env = fun.load_environment("DataTugasML3.txt")
    env = np.flip(env, axis=0)

    ql = QLearning(env)
    print(ql.environment, ql.explore_mat, ql.q_table)
    print("MATRIKS R",ql.r_table)
