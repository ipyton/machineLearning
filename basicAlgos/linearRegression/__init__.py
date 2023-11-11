import numpy as np
import matplotlib.pyplot as plt

rate = 0.01 # 太大不好 越学越偏
x_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

w0 = np.random.normal()
w1 = np.random.normal()
print(w0)
print(w1)
err = 1


def h(ww0, ww1, x):
    return ww0 + ww1 * x


def train():
    global w0, w1, err
    m = len(x_train)
    while err > 0.00001:
        for x, y in zip(x_train, y_train):
            ww0 = w0
            ww1 = w1
            w0 = w0 - rate * (h(ww0, ww1, x) - y) * 1 / m
            w1 = w1 - rate * (h(ww0, ww1, x) - y) * x / m
        err = 0
        for x, y in zip(x_train, y_train):
            err += (y - h(w0, w1, x)) ** 2

        err = float(err / (2 * m))
        print(err)


train()
print(w0, w1)
