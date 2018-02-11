import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def countWeight(X, y):
    x_mean = X.mean()
    y_mean = y.mean()
    w = np.sum((X-x_mean) * (y - y_mean)) / np.sum(((X-x_mean) ** 2))
    return w


def countA(X, y, w):
    a = y.mean() - (w * X.mean())
    return a


def countSquaredError(y, y_predicted):
    return np.sum((y - y_predict) ** 2)


# Importing the dataset
df = pd.read_csv("trainset1-15.csv", header=None)
X = df.iloc[:, 0:-1].values
y = df.iloc[:, -1].values

# Find w and a predicted value
w = countWeight(X, y)
a = countA(X, y, w)

# Predict y value
y_predict = np.cross(w, X)

# Count Err0r value
error_val = countSquaredError(y, y_predict)

# Summaryzing
print("Berikut persamaan garis regresi linier:")
print("y = " + str(w) + "X + " + str(a))
print("Dengan nilai error: ", error_val)


# Visualizing
plt.scatter(X, y, color="yellowgreen", label="Data")
plt.plot(X, y_predict, label="Model")
plt.xticks(())
plt.yticks(())
plt.legend(loc="best")
plt.show()

# Answering test data
print("")
print("Answering testset1-15.csv")
df = pd.read_csv("testset1-15.csv", header=None)
X = df.iloc[:, :].values

count = 0
for x in X:
    count += 1
    print(str(count) + ". " + str(w * x[0] + a))


from sklearn.metrics import mean_squared_error

print(mean_squared_error(y, y_predict))
