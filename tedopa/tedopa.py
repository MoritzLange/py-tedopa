"""
Implementation of the TEDOPA mapping as described in
Journal of Mathematical Physics 51, 092109 (2010); doi: 10.1063/1.3490188
"""

import numpy as np
import mpnum as mp
from tedopa import _recursion_coefficients as rc
from tedopa import tmps


# ToDo: Check if different g actually make a difference
# ToDo: Let the user provide the required compression

def tedopa1(h_loc, a, state, method, trotter_compr, compr, j, domain, ts, g,
            trotter_order=2, num_trotter_slices=100, ncap=60000, v=False):
    """
    Mapping the Hamiltonian of a system composed of one site, linearly coupled
    to a reservoir of bosonic modes, to a 1D chain and performing time evolution
    Args:
        h_loc (numpy.ndarray): Local Hamiltonian of the one site system
        a (numpy.ndarray): Interaction operator defined as A_hat in the paper
        state (mpnum.MPArray): The state of the system which is to be
            evolved. In PMPS form.
        method (str): The form of the state, determining the method used
            in the calculations. Either 'mpo' or 'pmps'.
        trotter_compr (dict):
            Compression parameters used in the iterations of Trotter (in the
            form used by
            mpnum.compress())
        compr (dict): Parameters for the compression which is executed on every
            MPA during the calculations, except for the Trotter calculation,
            where trotter_compr is used.
            compr = dict(method='svd', rank=10) would for example ensure that
            the ranks of any MPA never exceed 10 during all of the calculations.
        j (types.LambdaType): spectral function J(omega) as defined in the paper
        domain (list[float]): Domain on which j is defined,
            for example [0, np.inf]
        ts (list of float): The times in seconds for which the evolution should
            be computed.
        g (float): Constant g, assuming that for J(omega) it is g(omega)=g*omega
        trotter_order (int):
            Order of trotter to be used. Currently only 2 and 4
            are implemented
        num_trotter_slices (int): Number of Trotter slices to be used for the
            largest t in ts.
            If ts=[10, 25, 30] and num_trotter_slices=100,
            then the programme would use 100/30*10=33, 100/30*25=83 and
            100/30*30=100 Trotter slices to calculate the time evolution for the
            three times.
        ncap (int):
            Number internally used by py-orthpol. Must be <= 60000,
            the higher the longer the calculation of the coefficients takes
            and the more accurate it becomes.
        v (bool): Verbose or not verbose (will print what is going on vs.
            won't print anything)

    Returns:
        list[list[float], list[mpnum.MPArray]]: An array of times and an array
        of the corresponding evolved states
    """
    state_shape = state.shape
    if len(domain) != 2:
        raise ValueError("Domain needs to be of the form [x1, x2]")
    if len(a) != state_shape[0][0]:
        raise ValueError(
            "Dimension of 'a' must be the same as that of the \
            first site of the chain.")
    if len(state_shape) < 2:
        raise ValueError("The provided state has no chain representing "
                         "the mapped environment")

    if v: print("Calculating the TEDOPA mapping...")
    singlesite_ops, twosite_ops = _get_operators(h_loc, a, state_shape,
                                                 j, domain, g, ncap)
    if v: print("Proceeding to tmps...")
    # put max ranks for compression
    times, states, compr_errors, trot_errors = tmps.evolve(
        state=state, hamiltonians=[singlesite_ops, twosite_ops], ts=ts,
        num_trotter_slices=num_trotter_slices, method=method,
        trotter_compr=trotter_compr, trotter_order=trotter_order,
        compr=compr, v=v)
    return times, states


def tedopa2(h_loc, a, state, method, sys_position, trotter_compr, compr, js,
            domains, ts, gs=(1, 1), trotter_order=2, num_trotter_slices=100,
            ncap=60000, v=False):
    """
    Mapping the Hamiltonian of a system composed of two sites, each linearly
    coupled to a reservoir of bosonic modes, to a 1D chain and performing
    time evolution.
    The first elements in the lists js, domains, etc. always refer to the
    first (left) site and the second elements in the lists refer to the
    second (right) site of the system
    Args:
        h_loc (numpy.ndarray): Local Hamiltonian of the two site system
        a list[numpy.ndarray]: The two interaction operators defined as
            A_hat in the paper
        state (mpnum.MPArray): The state of the system which is to be
            evolved. In PMPS form.
        method (str): The form of the state, determining the method used
            in the calculations. Either 'mpo' or 'pmps'.
        sys_position (int): Which index, in the chain representing the state, is
            the position of the first site of the system (first would be 1,
            not 0). E.g. 3 if the chain sites are
            environment-environment-system-system-environment-environment
        trotter_compr (dict):
            Compression parameters used in the iterations of Trotter (in the
            form used by mpnum.compress())
        compr (dict): Parameters for the compression which is executed on every
            MPA during the calculations, except for the Trotter calculation,
            where trotter_compr is used.
            compr = dict(method='svd', rank=10) would for example ensure that
            the ranks of any MPA never exceed 10 during all of the calculations.
        js list[types.LambdaType]: spectral functions J(omega) for both
            environments as defined in the paper
        domains list[list[float]]: Domains on which the js are defined,
            for example [[0, np.inf], [0,1]]
        ts (list of float): The times in seconds for which the evolution should
            be computed.
        gs list[float]: Constant g, assuming that for J(omega) it is
            g(omega) = g * omega
        trotter_order (int):
            Order of trotter to be used. Currently only 2 and 4
            are implemented
        num_trotter_slices (int): Number of Trotter slices to be used for the
            largest t in ts.
            If ts=[10, 25, 30] and num_trotter_slices=100,
            then the programme would use 100/30*10=33, 100/30*25=83 and
            100/30*30=100 Trotter slices to calculate the time evolution for the
            three times.
        ncap (int):
            Number internally used by py-orthpol. Must be <= 60000,
            the higher the longer the calculation of the coefficients takes
            and the more accurate it becomes.
        v (bool): Verbose or not verbose (will print what is going on vs.
            won't print anything)

    list[list[float], list[mpnum.MPArray]]: An array of times and an array
        of the corresponding evolved states
    """
    state_shape = state.shape
    # ToDo: Implement some checks, like above
    if v: print("Calculating the TEDOPA mapping...")
    left_ops = _get_operators(np.zeros([state_shape[sys_position - 1][0]] * 2),
                              a[0], list(reversed(state_shape[:sys_position:])),
                              js[0], domains[0], gs[0], ncap)
    singlesite_ops_left, twosite_ops_left = (list(reversed(i)) for i in
                                             left_ops)
    singlesite_ops_right, twosite_ops_right = \
        _get_operators(np.zeros([state_shape[sys_position][0]] * 2), a[1],
                       state_shape[sys_position::], js[1], domains[1], gs[1],
                       ncap)
    singlesite_ops = singlesite_ops_left + singlesite_ops_right
    twosite_ops = twosite_ops_left + [h_loc] + twosite_ops_right

    if v: print("Proceeding to tmps...")
    # put max ranks for compression
    times, states, compr_errors, trot_errors = tmps.evolve(
        state=state, hamiltonians=[singlesite_ops, twosite_ops], ts=ts,
        num_trotter_slices=num_trotter_slices, method=method,
        trotter_compr=trotter_compr, trotter_order=trotter_order,
        compr=compr, v=v)
    return times, states


def _get_operators(h_loc, a, state_shape, j, domain, g, ncap):
    """
    Get the operators acting on the chain
    Args:
        h_loc (numpy.ndarray): Local Hamiltonian
        a (numpy.ndarray): Interaction operator defined as A_hat in the paper
        state_shape list[list[int]]: The shape of the chain on which the
            hamiltonian is to be applied
        j (types.LambdaType): spectral function J(omega) as defined in the paper
        domain (list[float]): Domain on which j is defined,
            for example [0, np.inf]
        g (float): Constant g, assuming that for J(omega) it is g(omega)=g*omega
        ncap (int):
            Number internally used by py-orthpol.

    Returns:
        list[list[numpy.ndarray]]: Lists of single-site and
            adjacent-site operators
    """
    params = _get_parameters(
        n=len(state_shape), j=j, domain=domain, g=g, ncap=ncap)
    dims_chain = [i[0] for i in state_shape]
    bs = [_get_annihilation_op(dim) for dim in dims_chain[1::]]
    b_daggers = [b.T for b in bs]
    return _get_singlesite_ops(h_loc, params, bs, b_daggers), \
           _get_twosite_ops(a, params, bs, b_daggers)


def _get_singlesite_ops(h_loc, params, bs, b_daggers):
    """
    Function to generate a list of the operators acting on every
    single site
    Args:
        h_loc (numpy.ndarray): Local Hamiltonian
        params (list): Parameters as returned by _get_parameters()
        bs (list): The list of annihilation operators acting on each site
            of the chain
        b_daggers (list): The list of creation operators acting on each site
            of the chain

    Returns:
        list: List of operators acting on every single site
    """
    omegas, ts, c0 = params
    singlesite_ops = [omegas[i]
                      * b_daggers[i].dot(bs[i]) for i in range(len(bs))]
    singlesite_ops = [h_loc] + singlesite_ops

    return singlesite_ops


def _get_twosite_ops(a, params, bs, b_daggers):
    """
    Function to generate a list of the operators acting on every
    two adjacent sites
    Args:
        a (numpy.ndarray): Interaction operator provided by the user
        params (list): Parameters as returned by _get_parameters()
        bs (list): The list of annihilation operators acting on each site
            of the chain
        b_daggers (list): The list of creation operators acting on each site
            of the chain

    Returns:
        list: List of operators acting on every two adjacent sites
    """
    omegas, ts, c0 = params
    twosite_ops = [ts[i] * (
        np.kron(bs[i], b_daggers[i + 1]) + np.kron(b_daggers[i], bs[i + 1])) for
                   i in range(len(bs) - 1)]
    twosite_ops = [c0 * np.kron(a, bs[0] + b_daggers[0])] + twosite_ops

    return twosite_ops


def _get_parameters(n, j, domain, g, ncap):
    """
    Calculate the parameters needed for mapping the Hamiltonian to a 1D chain
    Args:
        n (int): Number of recursion coefficients required
            (rc.recursionCoefficients() actually returns one more)
        j (types.LambdaType): spectral function J(omega) as defined in the paper
        domain (list[float]): Domain on which j is defined,
            for example [0, np.inf]
        g (float): Constant g, assuming that for J(omega) it is g(omega)=g*omega
        ncap (int):
            Number internally used by py-orthpol.

    Returns:
        list[list[float], list[float], float]: omegas, ts, c0
            as defined in the paper
    """
    alphas, betas = rc.recursionCoefficients(n, lb=domain[0],
                                             rb=domain[1], j=j, g=g, ncap=ncap)
    omegas = g * np.array(alphas)
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
