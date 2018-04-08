import random
import numpy as np


def init_centroids(k, X):
    """Membangkitkan nilai acak untuk menentukan titik awal centroid."""
    centroids = []
    for _ in range(k):
        np.random.seed(100)
        centroids.append([random.random()*max(X[:, 0]),
                          random.random()*max(X[:, 1])])
    return np.array(centroids)


def train(k, X, iteration):
    """Mencari titik centroid terbaru yang dapat mengcluster data dengan baik."""
    centroids = init_centroids(k, X)
    old_centroids = np.zeros_like(centroids)

    for _ in range(iteration):
        clusters = {i: [] for i in range(k)}  # Mengosongkan anggota cluster.

        # Membuat kluster dengan mengelompokkan data sesuai centroid terdekatnya.
        for row in X:
            distances = np.linalg.norm(row-centroids, axis=1)
            clusters[np.argmin(distances)].append(row)

        # Digunakan untuk menghitung kondisi konvergen
        old_centroids[:, :] = centroids[:, :]

        # Mengupdate titik centroid dengan rata-rata anggota kluster centroid tersebut.
        for i in range(len(centroids)):
            centroids[i] = np.mean(clusters[i], axis=0)

        # Kondisi berhenti saat tidak ada perubahaan titik di semua centroid
        if np.linalg.norm(centroids-old_centroids, axis=None) == 0:
            break
    return np.array(centroids), clusters


def predict(X, centroids):
    """Memprediksi class dari data dengan input cluster yang telah ditrain."""
    return [np.argmin(np.linalg.norm(row-centroids, axis=1)) for row in X]


def getSumSquaredError(centroids, clusters):
    """Mendapatkan nilai Sum Squared Error."""
    sse = 0
    for centroid, members in clusters.items():
        sse += np.sum((centroids[centroid]-members) ** 2)
    return sse
