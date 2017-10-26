"""
Functions to calculate recursion coefficients.

Author:
    Moritz Lange

Date:
    10/24/2017
"""

import math
import orthpol as orth
import numpy as np
import time


def recursionCoefficients(n=2, g=1, lb=[-1], rb=[1], funcs=[lambda x: 1.], w=[0], x=[0]):
    """
    Calculate the recursion coefficients for a given dispersion relation J(omega).
    It is assumed that J(omega) is a sum of continuous functions on certain intervals and delta peaks.
    
    :param n: Number of recursion coefficients required
    :param g: Constant g, assuming that for J(omega) it is g(omega)=g*omega
    :param lb: List of left bounds of intervals
    :param rb: List of right bounds of intervals
    :param funcs: List of continuous functions defined on above intervals
    :param w: List of weights of delta peaks in J(omega)
    :param x: List of positions of delta peaks in J(omega)
    :return: alpha , beta Which are lists containing the n first recursion coefficients
    """
    # ToDo: check if lb, rb and funcs have the same length
    # ToDo: check if x and w have the same length
    # ToDo: reduce domain of h

    # convert continuous functions in J to h_squared, store those in place in the list funcs
    for count, function in enumerate(funcs):
        h_squared = _j_to_hsquared(function, g)
        funcs[count] = h_squared

    # convert delta peaks in J to h_squared, first convert weight w and then position x
    for count, weight in enumerate(w):
        w[count] = weight * g / math.pi
    for count, position in enumerate(x):
        x[count] = position / g

    p = orth.OrthogonalPolynomial(n,
                                  lb=lb, rb=rb,  # Domains
                                  funcs=funcs, ncap=100 * n,
                                  x=x, w=w)  # discrete points in weight function

    return p.alpha, p.beta


def _j_to_hsquared(function, g):
    """
    Transform J(omega) to h^2(omega)
    :param function: J(omega)
    :param g: factor
    :return: h^2
    """
    h_squared = lambda x: function(g * x) * g / math.pi
    return h_squared


# -----------------------------------------
# -----------------------------------------
# Testing code


functions = [lambda x: 1. / math.sqrt(2. * math.pi) * np.exp(-x ** 2 / 2.)]

# start_time=time.clock()
alphas, betas = recursionCoefficients(n=100, funcs=functions, lb=[-np.inf], rb=[np.inf])
# print(time.clock()-start_time, "s")
print(alphas)
print(betas)
