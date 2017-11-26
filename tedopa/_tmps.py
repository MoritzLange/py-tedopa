"""
Functions to calculate recursion coefficients.

Author:
    Moritz Lange

Date:
    26/07/2017
"""

import mpnum as mp
from scipy.linalg import expm
import numpy as np
from itertools import repeat


def evolve(state, hamiltonians, t, num_time_steps, trotter_order=1, method='mpo'):
    """
    Assumptions:    All sites have two physical legs
                    and all legs on all sites have the same dimension
                    (for each site on its own this is to be expected since this is a density matrix)
    Generalization to arbitrary leg dimensions per site should maybe be added later
    :param state: Density matrix (MPO) of the state that is to be evolved. Contains N sites.
    :param hamiltonians: list of hamiltonians, first acting on each single site, second acting on every pair of adjacent sites
    :param trotter_order: For now only 1 or 2
    :param t: time
    :param num_time_steps: number of steps for trotter
    :param method: mpo or pmps
    :return: evolved state's density matrix as mpo
    """

    if method == 'mpo':
        evolved_state = evolve_mpo(state, hamiltonians, trotter_order, t, num_time_steps)
    elif method == 'pmps':
        evolved_state = evolve_pmps(state, hamiltonians, trotter_order, t, num_time_steps)
    else:
        return state

    return evolved_state


def evolve_mpo(state, hamiltonians, trotter_order, t, num_time_steps):
    # Later: first evolve hamiltonians acting on only one site
    # then evolve hamiltonians acting on two adjacent sites

    # error acceptable for compression
    relerr = 1e-4
    # max ranks acceptable
    maxRanks = 20

    h1 = hamiltonians[0]
    h2 = hamiltonians[1]

    # find the time evolution operator for a small time step for h1
    element = expm(-1j * t * 1 / num_time_steps * h1)
    element_dagger = expm(1j * t * 1 / num_time_steps * h1)
    mpo_elem = matrix_to_mpo(matrix=element, num_sites=1, site_shape=(2, 2))
    mpo_elem_dag = matrix_to_mpo(matrix=element_dagger, num_sites=1, site_shape=(2, 2))
    u1 = mp.chain(repeat(mpo_elem, len(state)))
    u1_dagger = mp.chain(repeat(mpo_elem_dag, len(state)))

    # find the time evolution operator for a small time step for h2
    if trotter_order == 1:
        u2, u2_dagger = trotter_1(hamiltonian=h2, num_sites=len(state), t=t, num_time_steps=num_time_steps)
    else:  # trotter_order == 2:
        u2, u2_dagger = trotter_2(hamiltonian=h2, num_sites=len(state), t=t, num_time_steps=num_time_steps)

    for i in range(num_time_steps):
        state1 = mp.dot(mp.dot(u1, state), u1_dagger)
        state2 = mp.dot(mp.dot(u2, state), u2_dagger)
        state = state1 + state2
        state.compress(method='svd', relerr=relerr)
        # implement something to store the error here
        if max(state.ranks) > maxRanks:
            break

    return state


def evolve_pmps(state, hamiltonians, trotter_order, t, num_time_steps):
    # To be implemented, for now just redirect at evolve_mpo
    return evolve_mpo(state, hamiltonians, trotter_order, t, num_time_steps)


def trotter_1(hamiltonian, num_sites, t, num_time_steps):
    """
        Trotter of first order
        :param hamiltonian:
        :param num_sites:
        :param t:
        :param num_time_steps:
        :return: u, u_dagger as MPOs
        """
    dim = int(np.sqrt(len(hamiltonian)))  # The dimension of the physical legs of the state

    # generate time evolution operator "element" acting on two adjacent sites for a small time step from given hamiltonian
    element = expm(-1j * t * 1 / num_time_steps * hamiltonian)
    element_dagger = expm(1j * t * 1 / num_time_steps * hamiltonian)

    mpo_elem = matrix_to_mpo(matrix=element, num_sites=2, site_shape=(dim, dim))
    mpo_elem_dag = matrix_to_mpo(matrix=element_dagger, num_sites=2, site_shape=(dim, dim))

    # get operators acting on odd sites
    u_odd = mpo_on_odd(mpo_elem=mpo_elem, num_sites=num_sites, dim=dim)
    u_odd_dag = mpo_on_odd(mpo_elem=mpo_elem_dag, num_sites=num_sites, dim=dim)

    # get operators acting on even sites
    u_even = mpo_on_even(mpo_elem=mpo_elem, num_sites=num_sites, dim=dim)
    u_even_dag = mpo_on_even(mpo_elem=mpo_elem_dag, num_sites=num_sites, dim=dim)

    # construct the complete time evolution operator for the state for a small time step
    u = mp.dot(u_even, u_odd)
    u_dagger = mp.dot(u_odd_dag, u_even_dag)

    return u, u_dagger


def trotter_2(hamiltonian, num_sites, t, num_time_steps):
    """
        Trotter of first order
        :param hamiltonian:
        :param num_sites:
        :param t:
        :param num_time_steps:
        :return: u, u_dagger as MPOs
        """
    dim = int(np.sqrt(len(hamiltonian)))  # The dimension of the physical legs of the state

    # generate time evolution operator "element" acting on two adjacent sites for a small time step from given hamiltonian
    element_even = expm(-1j * t * 1 / num_time_steps * hamiltonian)
    element_even_dagger = expm(1j * t * 1 / num_time_steps * hamiltonian)
    element_odd = expm(-1j * t * 1 / 2 * 1 / num_time_steps * hamiltonian)
    element_odd_dagger = expm(1j * t * 1 / 2 * 1 / num_time_steps * hamiltonian)

    mpo_elem_even = matrix_to_mpo(matrix=element_even, num_sites=2, site_shape=(dim, dim))
    mpo_elem_even_dag = matrix_to_mpo(matrix=element_even_dagger, num_sites=2, site_shape=(dim, dim))
    mpo_elem_odd = matrix_to_mpo(matrix=element_odd, num_sites=2, site_shape=(dim, dim))
    mpo_elem_odd_dag = matrix_to_mpo(matrix=element_odd_dagger, num_sites=2, site_shape=(dim, dim))

    # get operators acting on odd sites
    u_odd = mpo_on_odd(mpo_elem=mpo_elem_odd, num_sites=num_sites, dim=dim)
    u_odd_dag = mpo_on_odd(mpo_elem=mpo_elem_odd_dag, num_sites=num_sites, dim=dim)

    # get operators acting on even sites
    u_even = mpo_on_even(mpo_elem=mpo_elem_even, num_sites=num_sites, dim=dim)
    u_even_dag = mpo_on_even(mpo_elem=mpo_elem_even_dag, num_sites=num_sites, dim=dim)

    # construct the complete time evolution operator for the state for a small time step
    u = mp.dot(mp.dot(u_odd, u_even), u_odd)
    u_dagger = mp.dot(mp.dot(u_odd_dag, u_even_dag), u_odd_dag)

    return u, u_dagger


def mpo_on_odd(mpo_elem, num_sites, dim):
    """
    Creates the full mpo for the whole state from small mpos acting on two adjacent sites, only for those acting on odd-even sites
    :param mpo_elem: mpo acting on adjacent states
    :param num_sites: number of sites of the state
    :param dim: dimension of each physical leg of each site, integer
    :return: full mpo
    """
    odd = mp.chain(repeat(mpo_elem, int(num_sites / 2)))
    if (num_sites % 2 == 1):
        odd = mp.chain([odd, mpo_identity(dim=dim)])
    return odd


def mpo_on_even(mpo_elem, num_sites, dim):
    """
    Creates the full mpo for the whole state from small mpos acting on two adjacent sites, only for those acting on even-odd sites
    :param mpo_elem: mpo acting on adjacent states
    :param num_sites: number of sites of the state
    :param dim: dimension of each physical leg of each site, integer
    :return: full mpo
    """
    even = mp.chain([mpo_identity(dim=dim)] + list(repeat(mpo_elem, int(num_sites / 2 - .5))))
    if (num_sites % 2 == 0):
        even = mp.chain([even, mpo_identity(dim=dim)])
    return even


def mpo_identity(dim):
    """
    Generate an MPO of the unity operator acting on one site
    :param dim: physical dimension of the unity operator
    :return: MPO of the operator
    """
    id = np.identity(dim)
    mpo_id = mp.MPArray.from_array_global(id, ndims=2)

    return mpo_id


def matrix_to_mpo(matrix, num_sites, site_shape):
    """
    Generate an mpo from a MxN matrix in global form.
    :param matrix: The matrix to be converted
    :param num_sites: Number of sites represented by the matrix
    :param site_shape: Shape of a single site of the resulting mpo as used in mpnum, e.g. (2,2) for 2 legs each of physical dimension 2
    :return: the desired mpo
    """
    newShape = []
    for j in sorted(site_shape):
        newShape = newShape + [j] * num_sites
    matrix = matrix.reshape(newShape)
    mpo = mp.MPArray.from_array_global(matrix, ndims=len(site_shape))
    return mpo
