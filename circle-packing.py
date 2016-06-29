#coding: utf-8

import numpy as np
import csv
import matplotlib.pyplot as plt
import math
import random
import time
from pprint import pprint 


# store input radius data
radiuses = []
input_file = open("radius.csv", "r")
for row in input_file:
    rad = float(row[:-1])
    radiuses.append(rad)


def main():
    #initialize the variables
    epsilon = 0.0001
    r = 5.0
    x = np.array([random.random() for i in radiuses])
    y = np.array([random.random() for i in radiuses])

    # initialize the penalty
    rho = 100.0

    # previous alpha (initial value)
    prev_alpha = 1.0

    # main roop
    start = time.time()
    while i < 10:
        d = nabla_f(x, y, r, rho)
        d[0] *= -1
        d[1] *= -1
        d[2] *= -1
        if math.sqrt(np.dot(d[0], d[0]) + np.dot(d[1], d[1]) + d[2]**2) >= epsilon:
            alpha = armijo(x, y, r, d, rho, prev_alpha)
            prev_alpha = alpha
            x = x + alpha * d[0]
            y = y + alpha * d[1]
            r = r + alpha * d[2]
            i += 1
            
            pprint([x, y, r])
        else:
            break
    print str(time.time() - start) + "[sec]"
    # drawfigure(circles, r)


# the second section for objective function
def sigma1(x, y):
    sum = 0
    for i, ri in enumerate(radiuses):
        for j, rj in enumerate(radiuses):
            sum += max([0, (x[j] - x[i])**2 + (y[j] - y[i])**2 - (rj - ri)**2])
    return sum

# the third section for objective function
def sigma2(x, y, r):
    sum = 0
    for i, ri in enumerate(radiuses):
        sum += max([0, x[i]**2 + y[i]**2 - (r - ri)**2])
    return sum

# the fourth section for objective function
def sigma3(r):
    sum = 0
    for ri in radiuses:
        sum += max([0, ri - r])
    return sum


# objective function
def f(x, y, r, rho):
    return r * r + rho * sigma1(x, y) + rho * sigma2(x, y, r) + rho * sigma3(r)


# gradient vector
def nabla_f(x, y, r, rho):
    dxarr = []
    dyarr = []
    dr = 0
    for i, ri in enumerate(radiuses):
        dx = 0
        dy = 0
        for j, rj in enumerate(radiuses):
            if (x[j] - x[i])**2 + (y[j] - y[i])**2 - (rj - ri)**2 > 0:
                dx += 2 * (x[j] - x[i])
                dy += 2 * (y[j] - y[i])
        if x[i]**2 + y[i]**2 - (r - ri)**2 > 0:
            dx += 2 * x[i]
            dy += 2 * y[i]
            dr -= 2 * (r - ri)
        if ri - r > 0:
            dr -= 1
        dxarr.append(rho * dx)
        dyarr.append(rho * dy)
    dr = rho * dr + 2 * r
    return [np.array(dxarr), np.array(dyarr), dr]


# Armijo method
def armijo(x, y, r, d, rho, prev_alpha):
    alpha = prev_alpha
    beta = 0.9
    tau = 0.001
    dx = d[0]
    dy = d[1]
    dr = d[2]
    i = 0
    while i < 1000:
        if  f(x + alpha * dx, y + alpha * dy, r + alpha * dr, rho) > f(x, y, r, rho) + tau * alpha * (np.dot(dx, -dx) + np.dot(dy, -dy) + np.dot(dr, -dr)):
           alpha *= beta
           i += 1
           print i
        else:
            break
    return alpha

main()