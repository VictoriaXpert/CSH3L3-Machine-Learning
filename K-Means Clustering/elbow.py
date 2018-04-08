from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from fun import load_data

X_train = load_data("TrainsetTugas2.txt")

err = []
idx = []
for i in range(1, 10):
    idx.append(i)
    kmn = KMeans(n_clusters=i, tol=0)
    kmn.fit(X_train)
    err.append(abs(kmn.score(X_train)))

print(err)
plt.plot(idx, err)
plt.show()
