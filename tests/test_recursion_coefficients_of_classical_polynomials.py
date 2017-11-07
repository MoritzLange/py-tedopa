"""
Test to see if the calculated recursion coefficients from _recursion_coefficients.py for some classical polynomials
are actually the right ones.

Author:
    Moritz Lange
"""

from tedopa import _recursion_coefficients
import numpy as np
import math


class TestCoefficients(object):
    precision = 1e-4  # One could demand higher precision but then ncap might need to be higher which leads to longer computation times

    def test_chebyshev(self):
        """Tests if the recursion coefficients calculated for the chebyshev polynomials of first kind
        are the right ones.
        """
        n = 100  # Number of coefficients to be checked

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
        alphas_numeric, betas_numeric = _recursion_coefficients.recursionCoefficients(n=n, g=g, j=j, lb=lb,
                                                                                      rb=rb,
                                                                                      ncap=10000)

        # The theoretical values
        alphas = [0] * n
        betas = self.generate_chebyshev_betas(n)

        for k, alpha_k in enumerate(alphas):
            assert abs(alpha_k - alphas_numeric[k]) < TestCoefficients.precision
        for k, beta_k in enumerate(betas):
            assert abs(beta_k - betas_numeric[k]) < TestCoefficients.precision

    def test_legendre(self):
        """Tests if the recursion coefficients calculated for the legendre polynomials are the right ones.
        """
        n = 100  # Number of coefficients to be checked

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
        alphas_numeric, betas_numeric = _recursion_coefficients.recursionCoefficients(n=n, g=g, j=j, lb=lb,
                                                                                      rb=rb,
                                                                                      ncap=10000)

        # The theoretical values
        alphas = [0] * n
        betas = self.generate_legendre_betas(n)

        for k, alpha_k in enumerate(alphas):
            assert abs(alpha_k - alphas_numeric[k]) < TestCoefficients.precision
        for k, beta_k in enumerate(betas):
            assert abs(beta_k - betas_numeric[k]) < TestCoefficients.precision

    def test_hermite(self):
        """Tests if the recursion coefficients calculated for the hermite polynomials are the right ones.
        """
        n = 100  # Number of coefficients to be checked

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
        alphas_numeric, betas_numeric = _recursion_coefficients.recursionCoefficients(n=n, g=g, j=j, lb=lb,
                                                                                      rb=rb,
                                                                                      ncap=10000)

        # The theoretical values
        alphas = [0] * n
        betas = self.generate_hermite_betas(n)

        for k, alpha_k in enumerate(alphas):
            assert abs(alpha_k - alphas_numeric[k]) < TestCoefficients.precision
        for k, beta_k in enumerate(betas):
            assert abs(beta_k - betas_numeric[k]) < TestCoefficients.precision

    def generate_hermite_betas(self, n=10):
        """
        Generate the first n beta coefficients for monic hermite polynomials
        Source for the recursion relation: calculated by hand
        :param n: Number of required coefficients
        :return: List of the first n coefficients
        """
        return [1.77245385] + list(np.arange(0.5, n * 0.5, 0.5))

    def generate_legendre_betas(self, n=10):
        """
        Generate the first n beta coefficients for monic legendre polynomials
        Source for the recursion relation: http://people.math.gatech.edu/~jeanbel/6580/orthogPol13.pdf, accessed 11/07/17
        :param n: Number of required coefficients
        :return: List of the first n coefficients
        """
        coeffs = [2]
        for i in range(1, n):
            coeffs.append(i ** 2 / (4 * i ** 2 - 1))

        return coeffs

    def generate_chebyshev_betas(self, n=10):
        """
        Generate the first n beta coefficients for monic chebyshev polynomials
        Source for the recursion relation: https://www3.nd.edu/~zxu2/acms40390F11/sec8-3.pdf, accessed 11/07/17
        :param n: Number of required coefficients, must be >2
        :return: List of the first n coefficients
        """
        return [3.14158433] + [0.5] + [0.25] * (n - 2)
