# coding: utf-8

import numpy as np
import csv

# store input position data
position = []
input_file = open("position.csv", "r")
for row in input_file:
    pos = [float(s) for s in row[:-1].split(',')]
    position.append(np.array(pos))

#initialize the penalty
rho = 0.1

def main():
    # initialize the variables
    x = np.array([0, 0])
    r = 0.5
    epsilon = 0.01
    i = 0
    
    while i < 30 and np.linalg.norm(nabla_f(x, r, rho)) >= epsilon:
        d = -nabla_f(x, r, rho)
        alpha = armijo(x, r, d)
        # print alpha
        x = x + alpha * np.array([d[0], d[1]])
        r += alpha * d[2] #maron
        i += 1
        # print [x, r]
        # print x + np.array([d[0], d[1]])
        # print f(x, r, rho)


# prepare the second section for objective function
def sigma(x, r):
    sigma = 0
    for pos in position:
        sigma += max([0, np.linalg.norm(x - pos)**2 - r * r])
    return sigma


# objective function
def f(x, r, rho):
    return r * r + rho * sigma(x, r) + rho * max([0, -r])

# direction
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
def armijo(x, r, d):
    alpha = 1.0
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

