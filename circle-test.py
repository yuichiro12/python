#coding: utf-8

import numpy as np
import csv
import matplotlib.pyplot as plt
import math
import random
import time


# store input radius data
radiuses = []
input_file = open("radius3.csv", "r")
for row in input_file:
    rad = float(row[:-1])
    radiuses.append(rad)


def main():
    #initialize the variables
    epsilon = 0.00001
    area = 0
    for ri in radiuses:
        area += ri * ri * math.pi
    r = math.sqrt(area * 3 / math.pi)
    x = np.array([(random.random() - 0.5) * r for i in radiuses])
    y = np.array([(random.random() - 0.5) * r for i in radiuses])

    # initialize the penalty
    rho = 1.0

    # previous alpha (initial value)
    prev_alpha = 1.0

    print r, f(x, y, r, rho)
    # main roop
    start = time.time()
    drawfigure(x, y, radiuses, r)
    d = nabla_f(x, y, r, rho)
    while i < 100:
        d = nabla_f(x, y, r, rho)
        d[0] = -1 * d[0]
        d[1] = -1 * d[1]
        d[2] *= -1
        if np.dot(d[0], d[0]) + np.dot(d[1], d[1]) + d[2]**2 >= epsilon:
            alpha = armijo(x, y, r, d, rho, prev_alpha)
            prev_alpha = alpha
            x = x + alpha * d[0]
            y = y + alpha * d[1]
            r = r + alpha * d[2]
            print r, f(x, y, r, rho)
            i += 1
        else:
            break
    drawfigure(x, y, radiuses, r)


def drawfigure(x, y, radiuses, r):
    i = 0
    fig = plt.figure()
    ax = fig.add_subplot(111)
    circle = plt.Circle((0, 0), r, color="k", alpha=0.3)
    ax.add_patch(circle)
    for i, ri in enumerate(radiuses):
        circle = plt.Circle((x[i], y[i]), ri, color="g", alpha=0.3)
        ax.add_patch(circle)
        i += 1
    plt.xlim([-10.5, 10.5])
    plt.ylim([-10.5, 10.5])
    plt.show()


def f(x, y, r, rho):
    sum1 = 0
    sum2 = 0
    max3 = 0
    for i, ri in enumerate(radiuses):
        for j, rj in enumerate(radiuses):
            if i == j:
                continue
            elif (rj + ri)**2 - (x[j] - x[i])**2 - (y[j] - y[i])**2 > 0:
                sum1 += (rj + ri)**2 - (x[j] - x[i])**2 - (y[j] - y[i])**2
    for i, ri in enumerate(radiuses):
        if x[i]**2 + y[i]**2 - (r - ri)**2 > 0:
            sum2 += x[i]**2 + y[i]**2 - (r - ri)**2
    if max(radiuses) > r:
        max3 = max(radiuses) - r
    return r + rho * sum1 + rho * sum2 + max3


def nabla_f(x, y, r, rho):
    dx = []
    dy = []
    dr = 0
    tmpdx = 0
    tmpdy = 0
    for i, ri in enumerate(radiuses):
        for j, rj in enumerate(radiuses):
            if i == j:
                continue
            elif (rj + ri)**2 - (x[j] - x[i])**2 - (y[j] - y[i])**2 > 0:
                tmpdx += -2 * (x[j] - x[i])
                tmpdy += -2 * (y[j] - y[i])
        if x[i]**2 + y[i]**2 - (r - ri)**2 > 0:
            tmpdx += 2 * x[i]
            tmpdy += 2 * y[i]
            dr += -2 * r
    dr *= rho
    if max(radiuses) > r:
        dr += max(radiuses) - r
    dx.append(tmpdx)
    dy.append(tmpdy)
    dr += 1
    return [rho * np.array(dx), rho * np.array(dy), dr]

# Armijo method
def armijo(x, y, r, d, rho, prev_alpha):
    alpha = prev_alpha
    beta = 0.90
    tau = 0.00001
    dx = d[0]
    dy = d[1]
    dr = d[2]
    i = 0
    while i < 1000:
        if  f(x + alpha * dx, y + alpha * dy, r + alpha * dr, rho) > f(x, y, r, rho) + tau * alpha * (np.dot(dx, -dx) + np.dot(dy, -dy) + np.dot(dr, -dr)):
            alpha *= beta
            i += 1
        else:
            break
    return alpha

main()
