import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def countWeight(X, y):
    w = np.sum(((X - X.mean()) * (y - y.mean()))) / np.sum((X - X.mean())**2)
    return w


def polynomialUni(X, w, degrees):
    y = 0
    for degree in range(degrees+1):
        y = w * (X ** degree)
    return y


# Importing the dataset
df = pd.read_csv("trainset1-15.csv", header=None)
X = df.iloc[:, 0:-1].values
y = df.iloc[:, -1].values

# Find w and a predicted value
w = countWeight(X, y)
w = float(str(w).split("e")[0])

# Predict the value
y_predict = polynomialUni(X, w, 3)

# Visualizing
plt.scatter(X, y, color="yellowgreen", label="Data")
plt.plot(X, y_predict, label="Model")
plt.xticks(())
plt.yticks(())
plt.legend(loc="best")
plt.show()

from sklearn.metrics import mean_squared_error

print(mean_squared_error(y, y_predict))