# coding: utf-8

import numpy as np
import csv
import matplotlib.pyplot as plt
import time

# store input position data
position = []
with open("position.csv", "r") as input_file:
    for row in input_file:
        pos = [float(s) for s in row[:-1].split(',')]
        position.append(np.array(pos))


def main():
    # initialize the variables
    x = np.array([0, 0])
    r = 0.3
    epsilon = 0.0001
    i = 0

    # initialize the penalty
    rho = 0.7

    # previous alpha (initial value)
    prev_alpha = 1.0

    drawfigure(x[0], x[1], r)
    start = time.time()
    #main roop
    while i < 100 and np.linalg.norm(nabla_f(x, r, rho)) >= epsilon:
        d = -nabla_f(x, r, rho)
        alpha = armijo(x, r, d, rho, prev_alpha)
        prev_alpha = alpha
        x = x + alpha * np.array([d[0], d[1]])
        r += alpha * d[2] #maron
        i += 1
    print str(time.time() - start) + "[sec]"
    drawfigure(x[0], x[1], r)


# draw figure
def drawfigure(x0, x1, r):
    X = []
    Y = []
    for pos in position:
        X.append(pos[0])
        Y.append(pos[1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    circle = plt.Circle((x0, x1), r, color="k", alpha=0.2)
    ax.add_patch(circle)
    plt.xlim([-1.5, 1.5])
    plt.ylim([-1.5, 1.5])
    ax.scatter(X, Y, color="k", marker=".")
    plt.show()


# prepare the second section for objective function
def sigma(x, r):
    sigma = 0
    for pos in position:
        sigma += max([0, np.linalg.norm(x - pos)**2 - r * r])
    return sigma


# objective function
def f(x, r, rho):
    return r * r + rho * sigma(x, r) + rho * max([0, -r])


# gradient vector
def nabla_f(x, r, rho):
    dx0 = 0
    dx1 = 0
    dr = 2 * r - rho if r < 0 else 2 * r
    for pos in position:
        if np.linalg.norm(x - pos)**2 - r * r > 0:
            dx0 += 2 * (x[0] - pos[0])
            dx1 += 2 * (x[1] - pos[1])
            dr -= 2 * r * rho
    return np.array([dx0, dx1, dr])


# Armijo method
def armijo(x, r, d, rho, prev_alpha):
    alpha = prev_alpha
    beta = 0.9
    tau = 0.0001
    dx = np.array([d[0], d[1]])
    dr = d[2]
    i = 0
    while i < 500 and f(x + alpha * dx, r + alpha * dr, rho) > f(x, r, rho) + tau * alpha * np.dot(nabla_f(x, r, rho), d): #maron
        alpha *= beta
        i += 1
    return alpha

main()
