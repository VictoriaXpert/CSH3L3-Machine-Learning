# Nama  : Aditya Alif Nugraha
# NIM   : 1301154183
# Kelas : IF-39-01

import numpy as np
import matplotlib.pyplot as plt


def load_environment(filename):
    """Meload file environment."""
    env = []
    with open(filename) as f:
        for line in f:
            env.append(line.split())
    return np.array(env, dtype=int)


def visualize_tracks(tracks):
    tracks = np.array(tracks)
    plt.plot(tracks[:, 0], tracks[:, 1])
    plt.axis("tight")
    plt.title("Jalur Optimal")
    plt.grid(True)
    plt.show()
