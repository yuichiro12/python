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
    r = 0.369
    epsilon = 0.01
    
    while np.linalg.norm(nabla_f(x, r, rho)) >= epsilon:
        d = -nabla_f(x, r, rho)
        print d
        alpha = almijo(x, r, d)
        x += alpha * np.array([d[0], d[1]])
        r = d[2]
        print [x, r]

    print [x, r]


# prepare the second section for objective function
def sigma(x, r):
    sigma = 0
    for pos in position:
        sigma += max([0, np.linalg.norm(x - pos)**2 - r * r])
    return sigma


# objective function
def f(x, r, rho):
    return r * r + sigma(x, r) + max([0, -r])

# direction
def nabla_f(x, r, rho):
    dx1 = 0
    dx2 = 0
    dr = 2 * r - rho if r < 0 else 2 * r
    for pos in position:
        if np.linalg.norm(x - pos)**2 - r * r > 0:
            dx1 += 2 * (x[0] - pos[0])
            dx2 += 2 * (x[1] - pos[1])
            dr -= 2 * r * rho
    return np.array([dx1, dx2, dr])

# Almijo method
def almijo(x, r, d):
    alpha = 1.0
    beta = 0.9
    dx = np.array([d[0], d[1]])
    dr = d[2]
    while f(x + alpha * dx, r + dr, rho) > f(x, r, rho) + beta * alpha * np.dot(nabla_f(x, r, rho), d):
        alpha *= beta
        print d
        print [x, r]
        print [x + alpha * dx, r + dr]
    return alpha

main()
