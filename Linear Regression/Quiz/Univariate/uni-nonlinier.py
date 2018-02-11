import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Importing the dataset
df_train = pd.read_csv("trainset1-15.csv")
X = df_train.iloc[:, 0:-1].values
y = df_train.iloc[:, -1].values

# create matrix versions of these arrays

colors = ["teal", "yellowgreen", "gold", "blue", "green", "yellow", "red"]
lw = 2
# plt.plot(X, y, color="cornflowerblue", linewidth=lw,
#          label="ground truth")
plt.scatter(X, y, color="navy", s=30, marker="o", label="train set")

for count, degree in enumerate([9,8,7,6,5,4,3]):
    model = make_pipeline(PolynomialFeatures(degree), Ridge())
    model.fit(X, y)
    prediction = model.predict(X)
    plt.plot(X, prediction, color=colors[count], linewidth=lw,
             label="x^%d" % degree)

    model.

plt.legend(loc="best")



plt.show()