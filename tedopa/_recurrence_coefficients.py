"""
Functions to calculate recurrence coefficients, based on the package py-orthpol.
"""
import math

import orthpol as orth


def recurrenceCoefficients(n, lb, rb, j, g, ncap=60000):
    """
    Calculate recurrence coefficients for given spectral density

    Recurrence coeffcients for an arbitrary measure are defined as follows.
    Given some measure :math:`d\mu(x)`, which defines the set
    :math:`\\{\\pi_n(x)\\in \\mathbb{P}_n,n=0,1,2,\\ldots\\}` of monic
    orthogonal polynomials with respect to the measure, the following recurrence
    relation holds.

    .. math::

       \\pi_{k+1}(x)=(x- \\alpha_k)\\pi_k(x)-\\beta_k\\pi_{k-1}(x), \\quad k=0,1,2...

    where :math:`\\pi_{-1}(x)\\equiv 0`, and the recurrence coefficients are:

    .. math::

       \\alpha_k=\\frac{\\langle x \\pi_k,\pi_k\\rangle}{\\langle \\pi_k,\\pi_k\\rangle}, \\quad \\beta_k=\\frac{\\langle \\pi_k,\\pi_k\\rangle}{\\langle \\pi_{k-1},\\pi_{k-1}\\rangle}.

    The TEDOPA mapping for a given spectral density relies on calculating the
    recurrence coefficients with respect to the measure :math:`d\mu(x)=h^2(x)dx`,
    where :math:`J(\\omega)=\\pi h^2[g^{-1}(\\omega)]
    \\frac{dg^{-1}(\\omega)}{d\\omega}` and :math:`g(x) = gx`. Thus, this function
    first calculates the function :math:`h^2(x)` from :math:`J(\\omega)` and then
    calls py-orthpol package to find the recurrence coefficients.

    For more details, see Journal of Mathematical Physics 51, 092109 (2010);
    doi: 10.1063/1.3490188 and ACM Trans. Math. Softw. 20, 21-62 (1994);
    doi: 10.1145/174603.174605

    Note that the input `j` must be a python lambda function representing the
    spectral density :math:`J(\omega)`

    Args:
      n (int):
        Number of recurrence coefficients required.
      g (float):
        Constant g, assuming that for J(omega) it is g(omega)=g*omega.
      lb (float):
        Left bound of interval on which J is defined.
      rb (float):
        Right bound of interval on which J is defined.
      j (types.LambdaType):
        :math:`J(\omega)` defined on the interval (lb, rb)
      ncap (int):
        Number internally used by py-orthpol to determine the accuracy with
        which the recurrence coefficients are calculated. Must be >n and <=60000.
        Between 10000 and 60000 recommended, the higher the number the higher
        the accuracy and the longer the execution time. (Default value = 60000)

    Returns:
        list[list[float], list[float]]:
            A list of two lists, each with the respective recurrence coefficients
            :math:`\{\\alpha_i: i = 1,2,\\dots,n\}` and
            :math:`\{\\beta_i: i = 1,2,\\dots,n\}` defined above.

    """
    # It would also be possible to give lists of J(omega) and intervals as input
    # if the py-orthpol package was changed accordingly, adding the quadrature
    # points obtained there. But that turned out to return coefficients which
    # were too inaccurate for our purposes.
    # Also the procedure does not work for ncap > 60000, it would return wrong
    # values. n must be < ncap for orthpol to work

    # ToDo: Check if ncap <= 60000 is system dependent or holds everywhere

    if ncap > 60000:
        raise ValueError("ncap <= 60000 is not fulfilled")

    if n > ncap:
        raise ValueError("n must be smaller than ncap.")

    lb, rb, h_squared = _j_to_hsquared(func=j, lb=lb, rb=rb, g=g)

    p = orth.OrthogonalPolynomial(n, left=lb, right=rb, wf=h_squared, ncap=ncap)

    return p.alpha, p.beta


def _j_to_hsquared(func, lb, rb, g):
    """
    Transform spectral density J to square of coupling function h(x)

    This transforms the given spectral density :math:`J(\\omega)` to the square :math:`h^2(x)` of the coupling function assuming a linear dispersion relation :math:`g(x)=gx`.

    Args:
        func (types.LambdaType):
            J(omega)
        lb (float):
            Left boundary
        rb (float):
            Right boundary
        g (float):
            Constant slope of the linear dispersion relations

    Returns:
        list[float, float, types.LambdaType]:
            lb, rb, :math:`h^2`, where lb and rb are the new left and right boundaries
            for :math:`h^2`.

    """

    def h_squared(x): return func(g * x) * g / math.pi

    # change the boundaries of the interval accordingly
    lb = lb / g
    rb = rb / g
    return lb, rb, h_squared
