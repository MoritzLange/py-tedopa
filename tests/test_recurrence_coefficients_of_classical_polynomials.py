"""
Test to see if the calculated recursion coefficients from
_recursion_coefficients.py for some classical polynomials
are actually the right ones.
"""

import math

import numpy as np

from tedopa import _recurrence_coefficients as rc

precision = 1e-5
ncap = 30000
# One could demand higher precision but then ncap might
# need to be higher which leads to longer computation times


def test_chebyshev():
    """
    Tests recursion coefficients of Chebyshev polynomials of first kind.
    """
    n = 100  # Number of coefficients to be checked

    lb, rb, g = -1, 1, 1

    def h_squared(x): return (1 - x ** 2) ** (-.5)

    # Calculate J from h^2 by adjusting the function and the boundaries
    j, lb, rb, g = convert_hsquared_to_j(h_squared, lb, rb, g)

    # Calculate the recursion coefficients, the alphas and betas
    alphas_n, betas_n = rc.recurrenceCoefficients(n=n - 1, g=g, j=j, lb=lb,
                                                  rb=rb, ncap=ncap)

    # Expected values
    alphas = [0] * n
    betas = generate_chebyshev_betas(n)

    assert np.allclose(alphas, alphas_n, rtol=precision)
    assert np.allclose(betas, betas_n, rtol=precision)


def test_legendre():
    """Tests if the recursion coefficients calculated for the legendre
    polynomials are the right ones.
    """
    n = 100  # Number of coefficients to be checked

    lb = -1
    rb = 1

    def h_squared(x): return 1

    g = 1

    j, lb, rb, g = convert_hsquared_to_j(h_squared, lb, rb, g)

    # Calculate the recursion coefficients, the alphas and betas
    alphas_n, betas_n = rc.recurrenceCoefficients(n=n - 1, g=g, j=j, lb=lb,
                                                  rb=rb, ncap=ncap)

    # The theoretical values
    alphas = [0] * n
    betas = generate_legendre_betas(n)

    assert np.allclose(alphas, alphas_n, rtol=precision)
    assert np.allclose(betas, betas_n, rtol=precision)


def test_hermite():
    """Tests if the recursion coefficients calculated for the hermite
    polynomials are the right ones.
    """
    n = 100  # Number of coefficients to be checked

    lb = -np.inf
    rb = np.inf

    def h_squared(x): return np.exp(- x ** 2)

    g = 1

    j, lb, rb, g = convert_hsquared_to_j(h_squared, lb, rb, g)

    # Calculate the recursion coefficients, the alphas and betas
    alphas_n, betas_n = rc.recurrenceCoefficients(n=n - 1, g=g, j=j, lb=lb,
                                                  rb=rb, ncap=ncap)

    # The theoretical values
    alphas = [0] * n
    betas = generate_hermite_betas(n)

    assert np.allclose(alphas, alphas_n, rtol=precision)
    assert np.allclose(betas, betas_n, rtol=precision)


def generate_hermite_betas(n=10):
    """
    Generate the first n beta coefficients for monic hermite polynomials
    Source for the recursion relation: calculated by hand
    :param n: Number of required coefficients
    :return: List of the first n coefficients
    """
    return [1.77245385] + list(np.arange(0.5, n * 0.5, 0.5))


def generate_legendre_betas(n=10):
    """
    Generate the first n beta coefficients for monic legendre polynomials
    Source for the recursion relation:
    http://people.math.gatech.edu/~jeanbel/6580/orthogPol13.pdf,
    accessed 11/07/17
    :param n: Number of required coefficients
    :return: List of the first n coefficients
    """
    coeffs = [2]
    for i in range(1, n):
        coeffs.append(i ** 2 / (4 * i ** 2 - 1))

    return coeffs


def generate_chebyshev_betas(n=10):
    """
    Generate the first n beta coefficients for monic chebyshev polynomials
    Source for the recursion relation:
    https://www3.nd.edu/~zxu2/acms40390F11/sec8-3.pdf, accessed 11/07/17
    :param n: Number of required coefficients, must be >2
    :return: List of the first n coefficients
    """
    return [3.14158433] + [0.5] + [0.25] * (n - 2)


def convert_hsquared_to_j(h_squared, lb, rb, g):
    """
    Convert parameters to the requires input form of recursionCoefficients

    Need to do this since recursionCoefficients does not take the weight function h^2 but J as an
    input
    """

    def j(x): return (math.pi / g) * h_squared(x / g)
    lb = lb * g
    rb = rb * g
    return(j, lb, rb, g)
