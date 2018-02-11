import pandas as pd
import numpy as np

df_train = pd.read_csv("trainset2-15.csv")

X = df_train.iloc[:, 0:-1].values
y = df_train.iloc[:, -1].values

w = np.array([8.7519,5.3491,-1.4437,14.0686,-3.5229 ])

y_pred = np.dot(X, w)
print(y_pred)
# Visualizing
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X[:,0], X[:,1], y)
ax.plot(X[:,0], X[:,1], y_pred)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.xticks(())
plt.yticks(())

plt.show()