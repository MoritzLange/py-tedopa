"""
Functions to calculate the time evolution of an operator.

Author:
    Moritz Lange
"""

import mpnum as mp
import numpy as np
from scipy.linalg import expm
from itertools import repeat


def evolve(state, hamiltonians, t, num_trotter_slices, method, compr):
    """
    Evolve a state using tMPS.

    Args:
        state (mpnum.MPArray): The state to be evolved in time (the density matrix, not wave function).
            It is assumed, that every site has two legs and all legs of the state are of the same physical dimension.
            The state has to be an MPO or PMPS, depending on which method is chosen
        hamiltonians (list): Either a list containing the Hamiltonian acting on every single site
            and the Hamiltonian acting on every two adjacents sites, like [H_i, H_ij],
            or a list containing a list of Hamiltonians acting on the single sites
            and a list of Hamiltonians acting on each two adjacent sites, like
            [[h1, h2, h3, ...], [h12, h23, h34, ...]]
        t (float): The time for which the evolution should be computed
        num_trotter_slices (int): The number of time steps or Trotter slices for the time evolution
        trotter_order (int): Which order of trotter should be used. Currently implemented are only 1 and 2
        method (str): Which method to use. Either 'mpo' or 'pmps', leading to computations based on matrix product
            operators for density matrices or purified states for density matrices. PMPS is faster.
        compr: Compression parameters for the Trotter steps

    Returns:
        mpnum.MPArray: The evolved state's density matrix as MPO or PMPS, depending on chosen method
    """
    # ToDo: See if normalization is in fact in place
    # ToDo: Implement possibility to store time evolution at different times while calculating, instead of just at the end
    # ToDo: Implement opportunity to omit the single-site-hamiltonians to decrease the error
    # ToDo: Maybe add support for hamiltonians depending on time (if not too complicated)
    # ToDo: Make sure the hamiltonians are of the right dimension
    # ToDo: Implement tracking of errors

    _compress_losslessly(state, method)

    if len(state) < 3: raise ValueError("State has too few sites")
    if t == 0: return state

    u = _trotter(hamiltonians=hamiltonians, t=t, num_trotter_slices=num_trotter_slices, num_sites=len(state))
    _compress_losslessly(u, method)
    if method == 'mpo':
        u_dagger = u.T.conj()

    # evolve in time
    for i in range(num_trotter_slices):
        state = mp.dot(u, state)
        if method == 'mpo': state = mp.dot(state, u_dagger)
        _normalize(state, method)
        overlap = state.compress(**compr)
        # implement something to store the error here

    _normalize(state, method)

    return state


def _trotter(hamiltonians, t, num_trotter_slices, num_sites):
    """
    Calculate the time evolution operators u and u_dagger, each comprising even and odd terms, for one Trotter slice.

    Args:
        hamiltonians (list): List of two lists of Hamiltonians, the Hamiltonians in the first
            acting on every single site, the Hamiltonians in the second acting on every pair of two adjacent sites
        t (float): The time for which the evolution should be computed
        num_trotter_slices (int): The number of time steps or Trotter slices for the time evolution
        num_sites (int): Number of sites of the state to be evolved

    Returns:
        mpnum.MPArray: The time evolution operator u for one Trotter slice
    """
    h_single, h_adjacent = _get_h_list(hs=hamiltonians, num_sites=num_sites)
    u_single, u_odd, u_even = _get_u_list(h_single, h_adjacent, t=t, num_trotter_slices=num_trotter_slices)
    mpo_single, mpo_odd, mpo_even = _u_list_to_mpo(u_single, u_odd, u_even)
    u_two_sites = mp.dot(mp.dot(mpo_odd, mpo_even), mpo_odd)
    u = mp.dot(mp.dot(mpo_single, u_two_sites), mpo_single)

    return u


def _get_h_list(hs, num_sites):
    """
    If only one Hamiltonian acting on every single site and one acting on every two adjacent sites is given,
    transform it into the form returned. If not, check whether the lengths of the lists match the number of sites.
    Args:
        hs (list): Hamiltonians as in evolve()
        num_sites (int): Number of sites of the state to be evolved

    Returns:
         list: A list of Hamiltonians acting on the single sites
            and a list of Hamiltonians acting on each two adjacent sites, like
            [h1, h2, h3, ...], [h12, h23, h34, ...]

    """
    if type(hs[0]) is not list:
        hs = [list(repeat(hs[0], num_sites)), list(repeat(hs[1], num_sites - 1))]
    elif (len(hs[0]) != num_sites) or (len(hs[1]) != num_sites - 1):
        raise ValueError("Number of given Hamiltonians does not match number of sites")

    return hs[0], hs[1]


def _get_u_list(h_single, h_adjacent, t, num_trotter_slices):
    """
    Calculate time evolution operators from Hamiltonians. The time evolution operators acting on single and on odd sites
    contain a factor .5 for the second order Trotter.
    Args:
        h_single (list): The Hamiltonians acting on every single site
        h_adjacent (list): The Hamiltonians acting on every two adjacent sites
        t (float): The time for which the evolution should be computed
        num_trotter_slices (int): The number of time steps or Trotter slices for the time evolution

    Returns:
        list: Lists of time evolution operators.
            The first of those lists contains operators acting single sites. The second one the operators
            acting on odd adjacent sites and the third one the ones acting on even adjacent sites,
            like [u1, u2, ...] [u12, u34, ...], [u23, u45, ...]

    """
    u_single = list(expm(-1j * t / num_trotter_slices / 2 * h) for h in h_single)
    u_odd = list(expm(-1j * t / num_trotter_slices / 2 * h) for h in h_adjacent[::2])
    u_even = list(expm(-1j * t / num_trotter_slices * h) for h in h_adjacent[1::2])

    return u_single, u_odd, u_even


def _u_list_to_mpo(u_single, u_odd, u_even):
    """
    Transform the matrices for time evolution to operators acting on the full state
    Args:
        u_single (list): List of time evolution operators acting on single sites
        u_odd (list): List of time evolution operators acting on odd adjacent sites
        u_even (list): List of time evolution operators acting on even adjacent sites

    Returns:
        mpnum.MPArray: The time evolution MPOs for the full state acting on individual sites,
            acting on odd adjacent sites, acting on even adjacent sites.

    """
    dims = [len(u_single[i]) for i in range(len(u_single))]
    odd = mp.chain(
        matrix_to_mpo(u, [[dims[2 * i]] * 2, [dims[2 * i + 1]] * 2]) for i, u in enumerate(u_odd))
    even = mp.chain(
        matrix_to_mpo(u, [[dims[2 * i + 1]] * 2, [dims[2 * i + 2]] * 2]) for i, u in enumerate(u_even))
    even = mp.chain([mp.eye(1, dims[0]), even])
    if len(u_odd) > len(u_even):
        even = mp.chain([even, mp.eye(1, dims[-1])])
    elif len(u_even) == len(u_odd):
        odd = mp.chain([odd, mp.eye(1, dims[-1])])
    single = mp.chain(matrix_to_mpo(u, [[dims[i]] * 2]) for i, u in enumerate(u_single))

    return single, odd, even


def matrix_to_mpo(matrix, shape):
    """
    Generates a MPO from a NxN matrix in global form.
    The number of legs per site must be the same for all sites.
    Args:
        matrix (numpy.ndarray): The matrix to be transformed to an MPO
        shape (list): The shape the single sites of the resulting MPO should have, as used in mpnum.
            For example two sites with two legs each might look like [[3, 3], [2, 2]]

    Returns:
        mpnum.MPArray: The MPO representing the matrix
    """
    num_legs = len(shape[0])
    if not (np.array([len(shape[i]) for i in range(len(shape))]) == num_legs).all():
        raise ValueError("Not all sites have the same number of physical legs")
    newShape = []
    for i in range(num_legs):
        for j in range(len(shape)):
            newShape = newShape + [shape[j][i]]
    matrix = matrix.reshape(newShape)
    mpo = mp.MPArray.from_array_global(matrix, ndims=num_legs)
    _compress_losslessly(mpo, 'mpo')

    return mpo


# Does this work, is state mutable and this operation in place?
def _compress_losslessly(state, method):
    """
    Compress and normalize a state with a very high relative error. This is meant to get unnecessary ranks out of
    a state without losing information.
    Args:
        state (mpnum.MPArray): The state to be compressed
        method (str): Whether the state is MPO or PMPS

    Returns:

    """
    state.compress(relerr=1e-20)
    state = _normalize(state, method)


# Does this work, is state mutable and this operation in place?
def _normalize(state, method):
    """
    Normalize a state (hopefully in place)
    Args:
        state (mpnum.MPArray): The state to be normalized
        method (str): Whether it is a MPO or PMPS state

    Returns:

    """
    if method == 'pmps': state = state / mp.norm(state)
    if method == 'mpo': state = state / mp.trace(state)
