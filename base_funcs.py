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
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y)) / batch_size


def numerical_gradient(f, x):
    t = 1e-4
    grad = np.zeros_like(x)
    xt1 = np.copy(x)
    xt2 = np.copy(x)
    for idx in range(x.shape[0]):
        tmp1 = xt1[idx]
        tmp2 = xt2[idx]

        xt1[idx] = x[idx] + t
        xt2[idx] = x[idx] - t

        grad[idx] = (f(xt1) - f(xt2)) / (2*t)

        xt1[idx] = tmp1
        xt2[idx] = tmp2

    return grad


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x

def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)