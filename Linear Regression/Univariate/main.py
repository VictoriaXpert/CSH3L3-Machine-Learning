import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# Importing the dataset
df_train = pd.read_csv("trainset1-15.csv")
df_test = pd.read_csv("testset1-15.csv")

X = df_train.iloc[:,0:-1].values
y = df_train.iloc[:,-1].values

# Split the train data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Building the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict the splitted test data from train data
y_pred = model.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)