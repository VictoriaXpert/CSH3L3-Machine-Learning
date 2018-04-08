import numpy as np
import matplotlib.pyplot as plt
import csv


def load_data(filename):
    """Membuka file data dan merubah menjadi array numpy."""
    file = open(filename, "r")
    lines = file.readlines()

    data = []
    for line in lines:
        data.append(line.split("\n")[0].split("\t"))

    return np.array(data, dtype=float)


def visualize_data(X, y=None, centroids=[]):
    """Memvisualisasikan data dan/atau centroid."""
    if centroids == []:
        plt.scatter(X[:, 0], X[:, 1])
    else:
        colors = ["b", "g", "r", "c", "m", "y", "k"]
        for i in range(len(y)):
            plt.scatter(X[i, 0], X[i, 1], c=colors[y[i]])
        plt.scatter(centroids[:, 0], centroids[:, 1], marker="*", c="k")
    plt.show()


def write_to_file(filename, result):
    with open(filename, "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for res in result:
            writer.writerow(str(res+1))
    csv_file.close()
