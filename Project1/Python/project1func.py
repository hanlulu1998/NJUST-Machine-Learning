'''
Description: Project1 function
version: 1.0
Author: Han Lulu
Date: 2020-09-18 20:25:07
LastEditors: Han Lulu
LastEditTime: 2020-09-18 23:08:21
'''
import matplotlib.pyplot as plt
import numpy as np
import copy


def plotData(X, y):
    plt.plot(X, y, 'rx', markersize=10)
    plt.xlabel('Year')
    plt.ylabel('Price')
    return


def costFunction(X, y, theta, lamb):
    m = len(y)
    h = X.dot(theta)
    t = copy.deepcopy(theta)
    t[0] = 0
    # 添加了正则化
    J = 1 / (2 * m) * sum((h - y) ** 2) + lamb / (2 * m) * sum(t ** 2)
    grad = 1 / m * X.T.dot(h - y) + lamb / m * t
    return J, grad


def gradientDescent(X, y, theta, alpha, iterations, lamb):
    m = len(y)
    J_history = []
    for i in range(iterations):
        h = np.dot(X, theta)
        temp = h - y
        temp0 = theta[0] - alpha * (1 / m) * sum(temp)
        temp1 = theta[1] - alpha * (1 / m) * sum(temp * X[:, 1])
        theta = np.array([temp0, temp1])
        J_history.append(costFunction(X, y, theta, lamb))

    return theta, J_history


def featureNormalize(X):
    mu = np.mean(X, axis=0)
    X_norm = X - mu
    sigma = np.std(X_norm, axis=0)
    X_norm = X_norm / sigma
    return X_norm, mu, sigma


def reFeature(X_norm, mu, sigma):
    X_norm = X_norm * sigma
    X_norm = X_norm + mu
    return X_norm


def normalEqn(X, y):
    theta = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)
    return theta
