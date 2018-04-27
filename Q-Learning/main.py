import pandas as pd
import fun
from qlearning import QLearning
import numpy as np

if __name__ == '__main__':
    env = fun.load_environment("DataTugasML3.txt")
    env = np.flip(env, axis=0)

    ql = QLearning(env)
    ql.fit(20)
    tracks, rewards = ql.predict()
    
    print("=============== HASIL TRAINING Q-LEARNING ===============")
    print("Jalur: ", tracks)
    print("Total Rewards: ", rewards)
