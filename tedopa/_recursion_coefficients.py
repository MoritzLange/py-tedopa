"""
Functions to calculate recursion coefficients.

Author:
    Moritz Lange

Date:
    10/24/2017
"""

import math
import orthpol as orth


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
    # ToDo: check if n<ncap, that is n<60000

    # convert continuous functions in J to h_squared, store those in place in the list funcs
    for count, function in enumerate(funcs):
        newlb, newrb, h_squared = _j_to_hsquared(function, lb[count], rb[count], g)
        lb[count] = newlb
        rb[count] = newrb
        funcs[count] = h_squared

    # convert delta peaks in J to h_squared, first convert weight w and then position x
    for count, weight in enumerate(w):
        w[count] = weight * g / math.pi
    for count, position in enumerate(x):
        x[count] = position / g

    # ncap=60000 seems to be highest ncap that works, higher values will break the algorithm.
    # For a speedup decrease ncap and sacrifice some accuracy (for small enough n the sacrifice is negligible)
    p = orth.OrthogonalPolynomial(n,
                                  lb=lb, rb=rb,  # Domains
                                  funcs=funcs, ncap=20000,
                                  x=x, w=w)  # discrete points in weight function

    return p.alpha, p.beta


def _j_to_hsquared(function, lb, rb, g):
    """
    Transform J(omega) to h^2(omega)
    :param function: J(omega)
    :param lb: left boundary
    :param rb: right boundary
    :param g: factor
    :return: lb, rb, h^2 Where lb and rb are the new left and right boundaries for h^2
    """
    h_squared = lambda x: function(g * x) * g / math.pi

    # change the boundaries of the intervals accordingly
    lb = lb / g
    rb = rb / g

    return lb, rb, h_squared
