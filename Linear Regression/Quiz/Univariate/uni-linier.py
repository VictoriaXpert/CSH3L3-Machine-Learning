import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.model_selection import train_test_split

# Importing the dataset
df = pd.read_csv("trainset1-15.csv", header=None)
X = df.iloc[:,0:-1].values
y = df.iloc[:,-1].values
X = np.array(X)
print(X.sum())
# Split the dataset
# X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

# Building the model
model = LinearRegression()
model.fit(X, y)
print(model.coef_)

# Predict the test data
predictions = model.predict(X)

# Visualizing


plt.scatter(X, y)
plt.plot(X, predictions, linewidth=3, color="green")

plt.xticks(())
plt.yticks(())
plt.show()

from sklearn.metrics import mean_squared_error
print(mean_squared_error(y, predictions) * len(y))
print(model.coef_)