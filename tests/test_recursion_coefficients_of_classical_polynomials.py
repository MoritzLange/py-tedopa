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

    precision = 1e-14  # should probably be machine precision (how to find out what that is?)

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
        As = [0, 2, 2, 2, 2, 2, 2, 2, 2, 2]
        Bs = [0] * 10
        Cs = [1] * 10

        for count, A_n in enumerate(As):
            assert abs(A_n - (1 / np.sqrt(betas_numeric[count + 1]))) < TestCoefficients.precision
        for count, B_n in enumerate(Bs):
            assert abs(B_n - (alphas_numeric[count] / np.sqrt(betas_numeric[count + 1]))) < TestCoefficients.precision
        for count, C_n in enumerate(Cs):
            assert abs(C_n - (np.sqrt(betas_numeric[count] / betas_numeric[count + 1]))) < TestCoefficients.precision

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
        As = [0] * 10
        Bs = [0] * 10
        Cs = [0] * 10

        for count in range(10):
            As[count] = (2 * count + 1) / (count + 1)
            Cs[count] = count / (count + 1)

        for count, A_n in enumerate(As):
            assert abs(A_n - (1 / np.sqrt(betas_numeric[count + 1]))) < TestCoefficients.precision
        for count, B_n in enumerate(Bs):
            assert abs(B_n - (alphas_numeric[count] / np.sqrt(betas_numeric[count + 1]))) < TestCoefficients.precision
        for count, C_n in enumerate(Cs):
            assert abs(C_n - (np.sqrt(betas_numeric[count] / betas_numeric[count + 1]))) < TestCoefficients.precision

    def test_hermite(self):
        """Tests if the recursion coefficients calculated for the legendre polynomials are the right ones.
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
        As = [2] * 10
        Bs = [0] * 10
        Cs = [0] * 10

        for count in range(10):
            Cs[count] = 2* count

        for count, A_n in enumerate(As):
            assert abs(A_n - (1 / np.sqrt(betas_numeric[count + 1]))) < TestCoefficients.precision
        for count, B_n in enumerate(Bs):
            assert abs(B_n - (alphas_numeric[count] / np.sqrt(betas_numeric[count + 1]))) < TestCoefficients.precision
        for count, C_n in enumerate(Cs):
            assert abs(C_n - (np.sqrt(betas_numeric[count] / betas_numeric[count + 1]))) < TestCoefficients.precision
