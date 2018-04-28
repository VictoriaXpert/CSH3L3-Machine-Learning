# Nama  : Aditya Alif Nugraha
# NIM   : 1301154183
# Kelas : IF-39-01

import pandas as pd
import fun
from qlearning import QLearning
import numpy as np

if __name__ == '__main__':
    env = fun.load_environment("DataTugasML3.txt")
    env = np.flip(env, axis=0)
    print(env)

    ql = QLearning(env)
    ql.fit(50)
    tracks, rewards = ql.predict()

    # print(ql.q_table)
    print("=============== HASIL TRAINING Q-LEARNING ===============")
    print("Jalur: ", *tracks)
    print("Total Rewards: ", rewards)
    fun.visualize_tracks(tracks)
