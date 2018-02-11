from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Importing the dataset
df_train = pd.read_csv("trainset2-15.csv")

X = df_train.iloc[:, 0:-1].values
y = df_train.iloc[:, -1].values

# Building the model
lm = LinearRegression()
lm.fit(X,y)

# Predict the model
predictions = lm.predict(X)

# Visualizing
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X[:,0], X[:,1], y)
ax.plot(X[:,0], X[:,1], predictions)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.xticks(())
plt.yticks(())

plt.show()