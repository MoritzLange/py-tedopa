"""
Functions to calculate recursion coefficients.
"""
import math

import orthpol as orth


def recursionCoefficients(n=2, g=1, lb=-1, rb=1, j=lambda x: 1., ncap=60000):
    """
    Calculate the recursion coefficients for monic polynomials for a  given
    dispersion relation J(omega) as defined in the paper Journal of Mathematical
    Physics 51, 092109 (2010); doi: 10.1063/1.3490188 J(omega) must be a python
    lambda function.
    Args:
      n (int):
        Number of recursion coefficients required (Default value = 2)
      g (float):
        Constant g, assuming that for J(omega) it is g(omega)=g*omega (Default
        value = 1)
      lb (float):
        Left bound of interval on which J is defined (Default value = -1)
      rb (float):
        Right bound of interval on which J is defined (Default value = 1)
      j (float):
        J(omega) defined on above interval (Default value = lambda x: 1.)
      ncap (int):
        Number internally used by py-orthpol. Must be >n and <=60000. Between
        10000 and 60000 recommended, the higher the number the higher the
        accuracy and the longer the execution time. Defaults to 60000.
    Returns:
        list:
            A list with two items: (i) alphas, which is one half of the n first recursion coefficients and (ii) betas, which is other half of the n first recursion coefficients
    """
    # It would also be possible to give lists of J(omega) and intervals as input
    # if the py-orthpol package was changed accordingly, adding the quadrature
    # points obtained there. But that turned out to return coefficients which
    # were too inaccurate for our purposes.
    # The procedure does not work for ncap > 60000, it would return wrong values
    if ncap > 60000:
        return [0], [0]
    # n must be < ncap for orthpol to work
    if n > ncap:
        return [0], [0]
    # convert continuous functions in J to h_squared, store those in place in
    # the list j
    lb, rb, h_squared = _j_to_hsquared(func=j, lb=lb, rb=rb, g=g)
    p = orth.OrthogonalPolynomial(n,
                                  left=lb, right=rb,
                                  wf=h_squared, ncap=ncap)
    return p.alpha, p.beta


def _j_to_hsquared(func, lb, rb, g):
    """ Transform J(omega) to h^2(omega) which will be the weight function for
    the generated polynomials
    Args:
        func:
            J(omega)
        lb:
            left boundary
        rb:
            right boundary
        g: factor
    Returns:
        list:
            {lb, rb, h^2} where lb and rb are the new left and right boundaries
            for h^2
    """
    def h_squared(x): return func(g * x) * g / math.pi
    # change the boundaries of the interval accordingly
    lb = lb / g
    rb = rb / g
    return lb, rb, h_squared
