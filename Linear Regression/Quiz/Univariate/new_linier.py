import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt

def mean(val):
    return sum(val) / float(len(val))

def covariance(x, x_mean, y, y_mean):
    covar = 0.0
    for i in range(len(x)):
        covar += (x[i] - x_mean) * (y[i] - mean_y)