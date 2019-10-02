import numpy as np
import matplotlib.pylab as plt


def step_function(x):
    y = x > 0
    return y.astype(np.int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def identity_function(x):
    return x


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)  # オーバフロー対策
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


# 平均二乗誤差
def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)  # 0.5は最急降下法における計算を楽にするため


# 交差エントロピー誤差
def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))


def numerical_gradient(f, x):
    t = 1e-4
    grad = np.zeros_like(x)
    xt1 = np.copy(x)
    xt2 = np.copy(x)
    for idx in range(x.size):
        tmp1 = xt1[idx]
        tmp2 = xt2[idx]

        xt1[idx] = x[idx] + t
        xt2[idx] = x[idx] - t

        grad[idx] = (f(xt1) - f(xt2)) / (2*t)

        xt1[idx] = tmp1
        xt2[idx] = tmp2

    return grad
