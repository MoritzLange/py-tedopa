"""
Implementation of the TEDOPA mapping as described in
Journal of Mathematical Physics 51, 092109 (2010); doi: 10.1063/1.3490188
"""

import numpy as np
import mpnum as mp
from tedopa import _recursion_coefficients as rc
from tedopa import tmps


def tedopa(h_loc, a, chain, j, domain, g=1):
    """
    Mapping the Hamiltonian of a system linearly coupled to a reservoir of bosonic modes to a 1D chain and performing time evolution
    Args:
        h_loc (numpy.ndarray): Local Hamiltonian
        a (numpy.ndarray): Interaction operator defined as A_hat in the paper
        chain (mpnum.MPArray): The chain on which the hamiltonian is to be applied
        j (types.LambdaType): spectral function J(omega) as defined in the paper
        domain (list[float]): Domain on which j is defined, for example [0, np.inf]
        g (float): Constant g, assuming that for J(omega) it is g(omega)=g*omega

    Returns:
        mpnum.MPArray: The full system Hamiltonian mapped to a chain
    """

    # ToDo: Check if different g actually make a difference

    if len(domain) != 2:
        raise ValueError("Domain needs to be of the form [x1, x2]")
    if len(a) != chain.shape[0][0]:
        raise ValueError("Dimension of 'a' must be the same as that of the first site of the chain.")

    singlesite_ops, twosite_ops = _get_operators(a, chain, j, domain,g)


def _get_operators(a, chain, j, domain, g):
    """
    Get the operators acting on the interaction and environment part of the chain
    Args:
        a (numpy.ndarray): Interaction operator defined as A_hat in the paper
        chain (mpnum.MPArray): The chain on which the hamiltonian is to be applied
        j (types.LambdaType): spectral function J(omega) as defined in the paper
        domain (list[float]): Domain on which j is defined, for example [0, np.inf]
        g (float): Constant g, assuming that for J(omega) it is g(omega)=g*omega

    Returns:
        list[list[numpy.ndarray]]: Lists of single-site and adjacent-site operators
    """
    params = _get_parameters(len_chain=len(chain), j=j, domain=domain, g=g)
    dims_chain = [i[0] for i in chain.shape]
    bs = [_get_annihilation_op(dim) for dim in dims_chain]
    b_daggers = [b.T for b in bs]
    return _get_singlesite_ops(a, params, bs, b_daggers), _get_twosite_ops(params, bs, b_daggers)


def _get_singlesite_ops(a, params, bs, b_daggers):
    """
        Function to generate a list of the operators acting on every two adjacent sites
        Args:
            a (numpy.ndarray): Interaction operator provided by the user
            params (list): Parameters as returned by _get_parameters()
            bs (list): The list of annihilation operators acting on each site of the chain
            b_daggers (list): The list of creation operators acting on each site of the chain

        Returns:
            list: List of operators acting on every two adjacent sites
        """
    omegas, ts, c0 = params
    singlesite_ops = [omegas[i] * b_daggers[i].dot(bs[i]) for i in range(len(bs))]
    singlesite_ops[0] = singlesite_ops[0] + c0 * a.dot(bs[0] + b_daggers[0])
    return singlesite_ops


def _get_twosite_ops(params, bs, b_daggers):
    """
    Function to generate a list of the operators acting on every two adjacent sites
    Args:
        params (list): Parameters as returned by _get_parameters()
        bs (list): The list of annihilation operators acting on each site of the chain
        b_daggers (list): The list of creation operators acting on each site of the chain

    Returns:
        list: List of operators acting on every two adjacent sites
    """
    omegas, ts, c0 = params
    twosite_ops = [ts[i] * (
        np.kron(bs[i], b_daggers[i + 1]) + np.kron(b_daggers[i], bs[i + 1])) for i in range(len(bs) - 1)]
    return twosite_ops


def _get_parameters(len_chain, j, domain, g):
    """
    Calculate the parameters needed for mapping the Hamiltonian to 1D chain
    Args:
        len_chain (int): Length of the chain = number of recursion coefficients required
        j (types.LambdaType): spectral function J(omega) as defined in the paper
        domain (list[float]): Domain on which j is defined, for example [0, np.inf]
        g (float): Constant g, assuming that for J(omega) it is g(omega)=g*omega

    Returns:
        list[list[float], list[float], float]: omegas, ts, c0 as defined in the paper
    """
    alphas, betas = rc.recursionCoefficients(len_chain, lb=domain[0], rb=domain[1], j=j, g=g)
    omegas = g * np.array(alphas)[:-1:]
    ts = g * np.sqrt(np.array(betas)[1::])
    c0 = np.sqrt(betas[0])
    return omegas, ts, c0


def _get_annihilation_op(dim):
    """
    Creates the annihilation operator
    Args:
        dim (int): Dimension of the site it should act on

    Returns:
        numpy.ndarray: The annihilation operator
    """
    op = np.zeros((dim, dim))
    for i in range(dim - 1):
        op[i, i + 1] = np.sqrt(i + 1)
    return op
