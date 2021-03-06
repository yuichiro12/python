#coding: utf-8

import numpy as np
import csv
import matplotlib.pyplot as plt
import math
import random
import time


# store input radius data
radiuses = []
with open("radius.csv", "r") as input_file:
    for row in input_file:
        rad = float(row[:-1])
        radiuses.append(rad)


def main():
    #initialize the variables
    epsilon = 1e-5
    area = 0
    for ri in radiuses:
        area += ri * ri * math.pi
    r = math.sqrt(area * 3 / math.pi)
    # x = np.array([(random.random() - 0.5) * r for i in radiuses])
    # y = np.array([(random.random() - 0.5) * r for i in radiuses])

    # with open("newpoints.csv", "w") as fileobj:
    #     for i, ri in enumerate(radiuses):
    #         fileobj.write(str(x[i]) + ', ' + str(y[i]) + '\n')
    x = []
    y = []
    with open("points.csv", "r") as input_file2:
        for row in input_file2:
            pos = [float(s) for s in row[:-1].split(',')]
            x.append(pos[0])
            y.append(pos[1])
    
    i = 0

    # initialize the penalty
    rho = 1.3

    # previous alpha (initial value)
    prev_alpha = 1.0

    print r, f(x, y, r, rho)
    # main roop
    drawfigure(x, y, radiuses, r)
    start = time.time()
    while i < 1000:
        d = nabla_f(x, y, r, rho)
        d[0] *= -1
        d[1] *= -1
        d[2] *= -1
        if np.dot(d[0], d[0]) + np.dot(d[1], d[1]) + d[2]**2 >= epsilon**2:
            alpha = armijo(x, y, r, d, rho, prev_alpha)
            prev_alpha = alpha
            if f(x, y, r ,rho) - f(x + alpha * d[0], y + alpha * d[1], r + alpha * d[2], rho)  < 1e-20:
                break
            x = x + alpha * d[0]
            y = y + alpha * d[1]
            r = r + alpha * d[2]
            i += 1
        else:
            break
    print str(time.time() - start) + "[sec]"
    print r, f(x, y, r, rho)
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


# the second section for objective function
def sigma1(x, y):
    sum = 0
    for i, ri in enumerate(radiuses):
        for j, rj in enumerate(radiuses):
            if j <= i:
                continue
            elif (ri + rj)**2 - (x[i] - x[j])**2 - (y[i] - y[j])**2 > 0:
                sum += (ri + rj)**2 - (x[i] - x[j])**2 - (y[i] - y[j])**2
    return sum

# the third section for objective function
def sigma2(x, y, r):
    sum = 0
    for i, ri in enumerate(radiuses):
        if x[i]**2 + y[i]**2 - (r - ri)**2 > 0:
            sum += x[i]**2 + y[i]**2 - (r - ri)**2
    return sum

# the fourth section for objective function
def max3(r):
    return max(radiuses) - r if max(radiuses) > r else 0

# objective function
def f(x, y, r, rho):
    return r + rho * sigma1(x, y) + rho * sigma2(x, y, r) + rho * max3(r)


# gradient vector
def nabla_f(x, y, r, rho):
    dxarr = []
    dyarr = []
    dr = 0
    for i, ri in enumerate(radiuses):
        dx = 0
        dy = 0
        for j, rj in enumerate(radiuses):
            if j == i:
                continue
            elif (ri + rj)**2 - (x[i] - x[j])**2 - (y[i] - y[j])**2 > 0:
                dx += -2 * x[i] + 2 * x[j]
                dy += -2 * y[i] + 2 * y[j]
        if x[i]**2 + y[i]**2 - (r - ri)**2 > 0:
            dx += 2 * x[i]
            dy += 2 * y[i]
            dr += -2 * r + 2 * ri
        dxarr.append(dx)
        dyarr.append(dy)
    dr -= 1 if max(radiuses) > r else 0
    dr *= rho
    dr += 1
    return [rho * np.array(dxarr), rho * np.array(dyarr), dr]


# Armijo method
def armijo(x, y, r, d, rho, prev_alpha):
    alpha = prev_alpha
    beta = 0.9
    tau = 1e-5
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
