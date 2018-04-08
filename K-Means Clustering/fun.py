import numpy as np
import matplotlib.pyplot as plt


def load_data(filename):
    """Membuka file data dan merubah menjadi array numpy."""
    file = open(filename, "r")
    lines = file.readlines()

    data = []
    for line in lines:
        data.append(line.split("\n")[0].split("\t"))

    return np.array(data, dtype=float)


def visualize_data(X, centroids=None):
    """Memvisualisasikan data dan/atau centroid."""
    if centroids == None:
        plt.scatter(X[:, 0], X[:, 1])
    plt.show()
