"""
Test to see if the calculated recursion coefficients from _recursion_coefficients.py for Chebyshev polynomials
are actually the right ones.

Author:
    Moritz Lange
"""

import orthpol as orth
import numpy as np
import math


class TestCoefficients(object):
    # source for the recurrence relations used here: http://dlmf.nist.gov/18.9#E1 accessed at 10/26/2017

    #########################################################################
    #### DON'T FORGET THAT THE INPUT IS J AND GETS CONVERTED TO H^2!!!!! ####
    #########################################################################

    precision = 1e-14  # should probably be machine precision (how to find out what that is?)

    def test_chebyshev(self):
        # Tests if the recursion coefficients calculated for the chebyshev polynomials of first kind are the right ones.
        lb = [-1]
        rb = [1]
        wf = lambda x: (1 - x ** 2) ** (-.5)

        # somehow run the method to calculate 50 recursion coefficients, maybe ncap=10000 to make it faster?

        alphas = [2e-15]
        betas = [.5]
        for count, alphaN in enumerate(alphas):
            assert abs(alphaN - 0) < TestCoefficients.precision
        for count, betaN in enumerate(betas):
            assert abs(betaN - .5) < TestCoefficients.precision

    def test_legendre(self):
        # Tests if the recursion coefficients calculated for the legendre polynomials are the right ones.
        lb = [-1]
        rb = [1]
        wf = lambda x: 1

        # somehow run the method to calculate 50 recursion coefficients, maybe ncap=10000 to make it faster?

        alphas = [2e-15]
        betas = [.5]
        for count, alphaN in enumerate(alphas):
            assert abs(alphaN - 0) < TestCoefficients.precision
        for count, betaN in enumerate(betas):
            assert abs(betaN - (count / (2 * count + 1))) < TestCoefficients.precision

    def test_hermite(self):
        # Tests if the recursion coefficients calculated for the legendre polynomials are the right ones.
        lb = [-np.inf]
        rb = [np.inf]
        wf = lambda x: math.exp(- x ** 2)

        # somehow run the method to calculate 50 recursion coefficients, maybe ncap=10000 to make it faster?

        alphas = [2e-15]
        betas = [.5]
        for count, alphaN in enumerate(alphas):
            assert abs(alphaN - 0) < TestCoefficients.precision
        for count, betaN in enumerate(betas):
            assert abs(betaN - count) < TestCoefficients.precision
