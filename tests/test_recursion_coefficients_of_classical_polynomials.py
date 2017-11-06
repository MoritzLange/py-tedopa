"""
Test to see if the calculated recursion coefficients from _recursion_coefficients.py for Chebyshev polynomials
are actually the right ones.

Author:
    Moritz Lange
"""

from tedopa import _recursion_coefficients
import numpy as np
import math


class TestCoefficients(object):
    # source for the recurrence relations used here: http://dlmf.nist.gov/18.9#E1 accessed at 10/26/2017

    precision = 1e-10 # One could demand higher precision but then ncap might need to be higher which leads to longer
                      # computation times

    def test_chebyshev(self):
        """Tests if the recursion coefficients calculated for the chebyshev polynomials of first kind
        are the right ones.
        """
        # First define boundaries and function of the weight function,
        # but keep in mind that recursionCoefficients does not take the weight function h^2 but J as an input
        lb = -1
        rb = 1
        h_squared = lambda x: (1 - x ** 2) ** (-.5)
        g = 1

        # Then calculate J from h^2 by adjusting the function and the boundaries
        j = lambda x: (math.pi / g) * h_squared(x / g)
        lb = lb * g
        rb = rb * g

        # Calculate the recursion coefficients, the alphas and betas
        alphas_numeric, betas_numeric = _recursion_coefficients.recursionCoefficients(n=10, g=g, funcs=[j], lb=[lb],
                                                                                      rb=[rb],
                                                                                      ncap=10000)

        # The theoretical values
        alphas = [0] * 10
        betas = [1.77, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]

        for k, alpha_k in enumerate(alphas):
            assert abs(alpha_k - alphas_numeric[k]) < TestCoefficients.precision
        for k, beta_k in enumerate(betas):
            assert abs(beta_k - betas_numeric[k]) < TestCoefficients.precision

    def test_legendre(self):
        """Tests if the recursion coefficients calculated for the legendre polynomials are the right ones.
        """
        # First define boundaries and function of the weight function,
        # but keep in mind that recursionCoefficients does not take the weight function h^2 but J as an input
        lb = -1
        rb = 1
        h_squared = lambda x: 1
        g = 1

        # Then calculate J from h^2 by adjusting the function and the boundaries
        j = lambda x: (math.pi / g) * h_squared(x / g)
        lb = lb * g
        rb = rb * g

        # Calculate the recursion coefficients, the alphas and betas
        alphas_numeric, betas_numeric = _recursion_coefficients.recursionCoefficients(n=10, g=g, funcs=[j], lb=[lb],
                                                                                      rb=[rb],
                                                                                      ncap=10000)

        # The theoretical values
        alphas = [0] * 10
        betas = [1.77, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]

        for k, alpha_k in enumerate(alphas):
            assert abs(alpha_k - alphas_numeric[k]) < TestCoefficients.precision
        for k, beta_k in enumerate(betas):
            assert abs(beta_k - betas_numeric[k]) < TestCoefficients.precision

    def test_hermite(self):
        """Tests if the recursion coefficients calculated for the hermite polynomials are the right ones.
        """
        # First define boundaries and function of the weight function,
        # but keep in mind that recursionCoefficients does not take the weight function h^2 but J as an input
        lb = -np.inf
        rb = np.inf
        h_squared = lambda x: np.exp(- x ** 2)
        g = 1

        # Then calculate J from h^2 by adjusting the function and the boundaries
        j = lambda x: (math.pi / g) * h_squared(x / g)
        lb = lb * g
        rb = rb * g

        # Calculate the recursion coefficients, the alphas and betas
        alphas_numeric, betas_numeric = _recursion_coefficients.recursionCoefficients(n=10, g=g, funcs=[j], lb=[lb],
                                                                                      rb=[rb],
                                                                                      ncap=10000)

        # The theoretical values
        alphas = [0] * 10
        betas = [1.77, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]

        for k, alpha_k in enumerate(alphas):
            assert abs(alpha_k - alphas_numeric[k]) < TestCoefficients.precision
        for k, beta_k in enumerate(betas):
            assert abs(beta_k - betas_numeric[k]) < TestCoefficients.precision
