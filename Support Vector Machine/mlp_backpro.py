# Artificial Neural Network

# Data Preprocessing
import numpy as np
import pandas as pd

dataset = pd.read_csv("data_train_PNN.csv", header=None)
X = dataset.iloc[:, :-1].values.astype(float)
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33333333333333333333333333333333)
print(X_train)

# Part 2 - Building the ANN!
# Improting the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initializing the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(6, input_dim=3, activation="relu"))

# Adding the output layer
classifier.add(Dense(3, activation="softmax"))

# Compiling the ANNm
classifier.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size=5, epochs=100)


# Part 3 - Making the predictions and evaluating model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
from keras.utils import to_categorical
y_binary = to_categorical(y_pred)

# Part 4 - Evaluating, Improving, and tuning the ANN

# Evaluating
#from keras.wrappers.scikit_learn import KerasClassifier
#from sklearn.model_selection import cross_val_score

# def build_classifier():
#     classifier = Sequential()
#     classifier.add(Dense(units=6, activation="relu", input_dim=11))
#     classifier.add(Dense(units=6, activation="relu"))
#     classifier.add(Dense(units=1, activation="sigmoid"))
#     classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
#     return classifier

# classifier = KerasClassifier(build_fn=build_classifier,batch_size=10, epochs=100)
# accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10, n_jobs=-1)
