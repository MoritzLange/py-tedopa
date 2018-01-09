"""
Implementation of the TEDOPA mapping as described in
Journal of Mathematical Physics 51, 092109 (2010); doi: 10.1063/1.3490188
"""

import numpy as np

import mpnum as mp
from tedopa import _recursion_coefficients as rc
from tedopa import tmps


def get_hamiltonian(h_loc, a, chain, j, domain, g=1):
    """
    Mapping the Hamiltonian of a system linearly coupled to a reservoir of bosonic modes to a 1D chain
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

    h_loc = tmps.matrix_to_mpo(h_loc, [[len(h_loc)] * 2])
    dims_chain = [i[0] for i in chain.shape]
    sum = _get_sum(a, chain, j, domain, g)
    hamiltonian = mp.chain([h_loc, mp.eye(len(dims_chain), dims_chain)]) + mp.chain([mp.eye(h_loc.shape[0][0]), sum])
    return tmps.compress_losslessly(hamiltonian, 'mpo')


def _get_sum(a, chain, j, domain, g):
    """
    Get the second and last term, i.e. everything except the local Hamiltonian, from eq.(16) in the paper
    Args:
        a (numpy.ndarray): Interaction operator defined as A_hat in the paper
        chain (mpnum.MPArray): The chain on which the hamiltonian is to be applied
        j (types.LambdaType): spectral function J(omega) as defined in the paper
        domain (list[float]): Domain on which j is defined, for example [0, np.inf]
        g (float): Constant g, assuming that for J(omega) it is g(omega)=g*omega

    Returns:
        mpnum.MPArray: The sum, i.e. the last term of eq. (16) in the paper
    """
    a = tmps.matrix_to_mpo(a, [[len(a)] * 2])
    omegas, ts, c0 = _get_parameters(len_chain=len(chain), j=j, domain=domain, g=g)
    dims_chain = [i[0] for i in chain.shape]
    bs = [_get_annihilation_op(dim) for dim in dims_chain]
    b_daggers = [b.T for b in bs]
    singlesite_ops = [omegas[i] * mp.dot(b_daggers[i], bs[i]) for i in range(len(bs))]
    singlesite_ops[0] = singlesite_ops[0] + c0 * mp.dot(a, bs[0] + b_daggers[0])
    twosite_ops = [ts[i] * (
        mp.chain([bs[i], b_daggers[i + 1]]) + mp.chain([b_daggers[i], bs[i + 1]])) for i in range(len(bs) - 1)]

    return mp.local_sum(singlesite_ops) + mp.local_sum(twosite_ops)


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
        mpnum.MPArray: The creation operator
    """
    op = np.zeros((dim, dim))
    for i in range(dim - 1):
        op[i, i + 1] = np.sqrt(i + 1)
    return tmps.matrix_to_mpo(op, [[dim]] * 2)
