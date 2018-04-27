import numpy as np


def load_environment(filename):
    env = []
    with open(filename) as f:
        for line in f:
            env.append(line.split())
    return np.array(env, dtype=int)
