import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.neural_network import MLPClassifier
import fun
import pnn
# plt.figure(1)


df = pd.read_csv("data_train_PNN.csv", header=None)
df = df.sample(frac=1).reset_index(drop=True)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33333333333333333333333333333333)
# plt.subplot(211)
fun.scatter3d_visualize(X_test, y_test, "Train Set")

# SVM
clf_svm = svm.SVC()
clf_svm.fit(X_train, y_train)
y_pred = clf_svm.predict(X_test)
accuracy = fun.count_accuracy(y_pred, y_test)
print("Akurasi SVM: " + str(accuracy))
# plt.subplot(212)
fun.scatter3d_visualize(X_test, y_pred, "SVM")


# PNN
y_pred = pnn.classification(X_train, y_train, X_test, 1.42)
accuracy = fun.count_accuracy(y_pred, y_test)
print("Akurasi PNN: " + str(accuracy))
# plt.subplot(213)
fun.scatter3d_visualize(X_test, y_pred, "PNN")


# Naive Bayes
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
accuracy = fun.count_accuracy(y_pred, y_test)
print("Akurasi Naive Bayes: " + str(accuracy))
# plt.subplot(214)
fun.scatter3d_visualize(X_test, y_pred, "Naive Bayes")


# MLP - Backpropagation
mlp_clf = MLPClassifier(hidden_layer_sizes=(15, 15))
mlp_clf.fit(X_train, y_train)
y_pred = mlp_clf.predict(X_test)
print("Akurasi MLP - Backpropagation: " + str(accuracy))
# plt.subplot(215)
fun.scatter3d_visualize(X_test, y_pred, "MLP - Backpropagation")


plt.show()
