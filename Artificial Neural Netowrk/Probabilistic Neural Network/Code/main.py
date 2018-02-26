import math
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


def open_csv_file(filename, set_type=""):
    # Soal A nomer 1
    """
    Open csv file return X and/or y based on choosen set type.
    Set type must: "train" or "test".
    """
    df = pd.read_csv(filename, header=None)

    if set_type == "train":
        df = df.sample(frac=1).reset_index(drop=True) # Mengacak data train
        X = df.iloc[:, 0:-1].values
        y = df.iloc[:, -1].values
        return X, y
    elif set_type == "test":
        X = df.iloc[:, :].values
        return X
    else:
        raise TypeError("Err: set_type not available")


def train_validation_split(X, y, train_size):
    """
    Split the dataset to train and validation set.
    This function return X_train, y_train, X_validation, y_validation.
    """
    size_limit = int(len(X) * train_size)
    X_train = X[0:size_limit, :]
    y_train = y[0:size_limit]
    X_validation = X[size_limit:len(X) + 1, :]
    y_validation = y[size_limit:len(X) + 1]

    return X_train, y_train, X_validation, y_validation


def scatter3d_visualize(X, y):
    # Soal A nomer 1
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(X)):
        xs = X[i][0]
        ys = X[i][1]
        zs = X[i][2]

        if y[i] == 0:
            ax.scatter(xs, ys, zs, c="r", marker="o")
        elif y[i] == 1:
            ax.scatter(xs, ys, zs, c="g", marker="s")
        else:
            ax.scatter(xs, ys, zs, c="b", marker="^")

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()


def count_pdf(X, y, X_input, sigma):
    """Fungsi untuk menghitung PDF pada PNN dengan input X, y, X input, dan sigma.
    Fungsi ini mengembalikan nilai PDF untuk semua kelas terurut sesuai index."""
    n = len(X)
    #print(X_input)
    sum_gx = [0 for i in range(max(y) + 1)]
    #print(sum_gx)
    count_0 = 0
    count_1 = 0
    count_2 = 0
    # fx = (1/len(X)) * sum(math.exp(-1*((sum(abs(x_input-X)) ** 2))/2*(sigma**2)))
    for i in range(n):
        sum_gx[y[i]] += math.exp(-1 * (((X_input[0] - X[i][0])**2 + (X_input[1] - X[i][1])**2 + (X_input[2] - X[i][2])**2) / (2 * (sigma**2))))  # menghitung gx
        if y[i] == 0:
            count_0 += 1
        if y[i] == 0:
            count_1 += 1
        else:
            count_2 += 1
    
    return [sum_gx[0] / count_0, sum_gx[1] / count_1, sum_gx[2] / count_2] # mengembalikan nilai PDF


def getIndexOfClass(X, y, selected_class):
    """Fungsi untuk mendapatkan index dari class yang diinginkan.
    Fungsi ini mengembalikan index, X, dan y dari kelas yang diinginkan."""
    idx = []
    X = []
    y_class = []
    for i in range(len(y)):
        if y[i] == selected_class:
            idx.append(i)
            X.append(X[i])
            y_class.append(selected_class)

    return idx, X, y_class


def classification(X_train, y_train, X_input, sigma):
    """Fungsi ini untuk memprediksi kelas dari input X_train, y_train, X_input, dan sigma."""
    y_pred = []
    for X in X_input:
        # print(X)
        pdf = count_pdf(X_train, y_train, X, sigma)
        y_pred.append(pdf.index(max(pdf)))
    return y_pred


def countAccuracy(y_real, y_pred):
    acc = 0
    n = len(y_real)
    for i in range(n):
        if y_real[i] == y_pred[i]:
            acc += 1
    return acc / n


if __name__ == '__main__':
    # Perintah soal nomor 1
    X, y = open_csv_file("data_train_PNN.csv", "train")
    X_test = open_csv_file("data_test_PNN.csv", "test")
    scatter3d_visualize(X, y)

    X_train, y_train, X_validation, y_validation = train_validation_split(
        X, y, 0.75)

    # print(X_train, y_train, X_validation, y_validation)
    y_pred = classification(X_train, y_train, X_validation, 0.15)
    print(y_pred)
    print("Accuracy: " + str(countAccuracy(y_validation, y_pred)))

    y_test = classification(X, y, X_test, 0.1)
    print(y_test)