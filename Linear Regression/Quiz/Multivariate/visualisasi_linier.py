import pandas as pd
import numpy as np

df_train = pd.read_csv("trainset2-15.csv")

X = df_train.iloc[:, 0:-1].values
y = df_train.iloc[:, -1].values

w = np.array([1.6441,2.0457])

y_pred = np.dot(X, w)
print(y_pred)
# Visualizing
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X[:,0], X[:,1], y)
ax.plot(X[:,0], X[:,1], y_pred)

ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')

plt.xticks(())
plt.yticks(())

plt.show()