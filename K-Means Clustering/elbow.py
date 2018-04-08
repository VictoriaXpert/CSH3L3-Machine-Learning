"""
Nama    : Aditya Alif Nugraha
NIM     : 1301154183
Kelas   : IF-39-01
"""

import KMeans as kmeans
import matplotlib.pyplot as plt
from fun import load_data

X_train = load_data("TrainsetTugas2.txt")

err = []
idx = []
for i in range(1, 9):
    print(i)
    idx.append(i)
    centroids, clusters = kmeans.train(i, X_train, 300)
    err.append(kmeans.getSumSquaredError(centroids, clusters))

print(err)
plt.plot(idx, err)
plt.show()
