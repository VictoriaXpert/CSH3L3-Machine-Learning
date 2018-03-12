import math
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import random
import fun
##### Aditya Alif Nugraha #####
##### 1301154183          #####
##### IF 39-01            #####


def count_pdf(X, y, X_input, sigma):
    """Fungsi untuk menghitung PDF pada PNN dengan input X, y, X input, dan sigma.
    Fungsi ini mengembalikan nilai PDF untuk semua kelas terurut sesuai index."""
    n = len(X)
    # print(X_input)
    sum_gx = [0 for i in range(max(y) + 1)]
    # print(sum_gx)
    count_0 = 0
    count_1 = 0
    count_2 = 0
    # fx = (1/len(X)) * sum(math.exp(-1*((sum(abs(x_input-X)) ** 2))/2*(sigma**2)))
    for i in range(n):
        sum_gx[y[i]] += math.exp(-1 * (((X_input[0] - X[i][0])**2 + (X_input[1] - X[i][1])
                                        ** 2 + (X_input[2] - X[i][2])**2) / (2 * (sigma**2))))  # menghitung gx
        if y[i] == 0:
            count_0 += 1
        if y[i] == 0:
            count_1 += 1
        else:
            count_2 += 1

    # mengembalikan nilai PDF
    return [sum_gx[0] / count_0, sum_gx[1] / count_1, sum_gx[2] / count_2]


def classification(X_train, y_train, X_input, sigma):
    """Fungsi ini untuk memprediksi kelas dari input X_train, y_train, X_input, dan sigma."""
    y_pred = []
    for X in X_input:
        # print(sigma)
        pdf = count_pdf(X_train, y_train, X, sigma)
        y_pred.append(pdf.index(max(pdf)))
    return y_pred


def observe_sigma(start, end, X_train, y_train, X_validation, y_validation):
    """Fungsi observasi Sigma secara otomatis.
    Mencari nilai Sigma dengan interval dari nilai start-end yang diinputkan."""

    # Tier 1 - Mencari Sigma pada range start - end dengan step 1
    sig = []
    for i in range(start, end):
        if i == 0:
            i = 0.1
        # print(i)
        sig.append(fun.count_accuracy(y_validation, classification(
            X_train, y_train, X_validation, i)))

    # Tier 2 - Mencari Sigma pada range yang memiliki accuracy terbaik dengan step 0,1
    start_tier2 = sig.index(max(sig))
    i = start_tier2
    end_tier2 = start_tier2 + sig.count(max(sig))
    # print(start_tier2, end_tier2)
    sig = []
    while i <= end_tier2:
        if i == 0:
            i = 0.1
        sig.append(fun.count_accuracy(y_validation, classification(
            X_train, y_train, X_validation, i)))
        i += 0.1
    # print(sig)

    # Tier 3 - Mencari Sigma pada range yang memiliki accuracy terbaik dengan step 0,01
    start_tier3 = start_tier2 + sig.index(max(sig)) / 10
    i = start_tier3
    end_tier3 = start_tier3 + sig.count(max(sig)) / 10
    # print(start_tier3, end_tier3)
    sig = []
    while i <= end_tier3:
        if i == 0:
            i = 0.1
        sig.append(fun.count_accuracy(y_validation, classification(
            X_train, y_train, X_validation, i)))
        i += 0.01

    start_best_sigma = start_tier3 + (sig.index(max(sig)) / 100)
    end_best_sigma = start_best_sigma + (sig.count(max(sig)) / 100)
    # print(start_best_sigma, end_best_sigma)
    # Mengembalikan nilai sigma terbaik
    return (start_best_sigma + end_best_sigma) / 2
