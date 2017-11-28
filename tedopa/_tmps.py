"""
Functions to calculate recursion coefficients.

Author:
    Moritz Lange

Date:
    28/07/2017
"""

import mpnum as mp
from scipy.linalg import expm
import numpy as np
from itertools import repeat


def evolve(state, hamiltonians, t, num_time_steps, trotter_order, method):
    """
    Evolve a state using tMPS.

    Args:
        state (mpnum.MPArray): The state to be evolved in time (the density matrix, not wave function).
            It is assumed, that every site has two legs and all legs of the state are of the same physical dimension
        hamiltonians (list of numpy.ndarray): List of two Hamiltonians, the first acting on every single site, the
            second acting on every pair of two adjacent sites
        t (int): The time for which the evolution should be computed
        num_time_steps (int): The number of time steps or Trotter slices for the time evolution
        trotter_order (int): Which order of trotter should be used. Currently implemented are only 1 and 2
        method (str): Which method to use. Either 'mpo' or 'pmps', leading to computations based on matrix product
            operators for density matrices or purified states for density matrices

    Returns:
        mpnum.MPArray: The evolved state's density matrix
    """
    # ToDo: Maybe add support for states with different physical dimensions at each site (if not too complicated)
    # ToDo: Make sure the hamiltonians are of the right dimension
    # ToDo: Implement evolve_pmps()

    if method == 'mpo':
        evolved_state = evolve_mpo(state, hamiltonians, trotter_order, t, num_time_steps)
    elif method == 'pmps':
        evolved_state = evolve_pmps(state, hamiltonians, trotter_order, t, num_time_steps)
    else:
        return state

    return evolved_state


def evolve_mpo(state, hamiltonians, t, num_time_steps, trotter_order):
    """
    Evolve the state using MPO representation instead of purified states, for further documentation see evolve().

    Args:
        See evolve()

    Returns:
        mpnum.MPArray: The evolved state's density matrix
    """

    # error acceptable for compression
    relerr = 1e-4
    # max ranks acceptable
    maxRanks = 100

    u, u_dagger = trotter(hamiltonians, trotter_order=trotter_order, num_sites=len(state), t=t,
                          num_time_steps=num_time_steps)

    for i in range(num_time_steps):
        state = mp.dot(mp.dot(u, state), u_dagger)
        state.compress(method='svd', relerr=relerr)
        # implement something to store the error here
        if max(state.ranks) > maxRanks:
            break

    return state


def evolve_pmps(state, hamiltonians, t, num_time_steps, trotter_order):
    """
        Evolve the state using purified state representation instead of MPOs, for further documentation see evolve().

        Args:
            See evolve()

        Returns:
            mpnum.MPArray: The evolved state's density matrix
        """
    # To be implemented, for now just redirect to evolve_mpo()
    return evolve_mpo(state, hamiltonians, trotter_order, t, num_time_steps)


def trotter(hamiltonians, t, num_time_steps, trotter_order, num_sites):
    """
    Calculate the time evolution operators u and u_dagger, each comprising even and odd terms, for one Trotter slice.

    Args:
        hamiltonians (list of numpy.ndarray): List of two Hamiltonians, the first acting on every single site, the
            second acting on every pair of two adjacent sites
        t (int): The time for which the evolution should be computed
        num_time_steps (int): The number of time steps or Trotter slices for the time evolution
        trotter_order (int): Which order of trotter should be used. Currently implemented are only 1 and 2
        num_sites (int): Number of sites of the state to be evolved

    Returns:
        mpnum.MPArray: The time evolution operators u and u_dagger for one Trotter slice
    """
    hamiltonian1 = hamiltonians[0]  # the hamiltonian acting on every site
    hamiltonian2 = hamiltonians[1]  # the hamiltonian acting on every two adjacent sites

    hamiltonian1_2 = np.kron(hamiltonian1, np.identity(len(hamiltonian1))) + np.kron(np.identity(len(hamiltonian1)),
                                                                                     hamiltonian1)  # hamiltonian1 acting on two sites

    dim = int(len(hamiltonian1))  # The dimension of the physical legs of the state

    trotter_factor_even = 1

    if trotter_order == 1:
        trotter_factor_odd = 1
    else:  # trotter_order == 2
        trotter_factor_odd = .5

    # from given hamiltonian generate time evolution operator "element" acting on two adjacent sites for a small time step
    # for the odd sites also add the hamiltonians acting on every single site individually (but only for the odd ones, otherwise
    # they'd be applied twice
    element_even = expm(-1j * t * 1 / num_time_steps * trotter_factor_even * hamiltonian2)
    element_even_dagger = expm(1j * t * 1 / num_time_steps * trotter_factor_even * hamiltonian2)
    element_odd = expm(-1j * t * 1 / num_time_steps * trotter_factor_odd * (hamiltonian1_2 + hamiltonian2))
    element_odd_dagger = expm(1j * t * 1 / num_time_steps * trotter_factor_odd * (hamiltonian1_2 + hamiltonian2))

    mpo_elem_even = matrix_to_mpo(matrix=element_even, num_sites=2, site_shape=(dim, dim))
    mpo_elem_even_dag = matrix_to_mpo(matrix=element_even_dagger, num_sites=2, site_shape=(dim, dim))
    mpo_elem_odd = matrix_to_mpo(matrix=element_odd, num_sites=2, site_shape=(dim, dim))
    mpo_elem_odd_dag = matrix_to_mpo(matrix=element_odd_dagger, num_sites=2, site_shape=(dim, dim))

    # generate time evolution operator "fill" acting on the last site if num_sites is odd (and therefore this last site
    # is not covered by mpo_elem_odd)
    fill = expm(-1j * t * 1 / num_time_steps * trotter_factor_odd * hamiltonian1)
    fill_dagger = expm(1j * t * 1 / num_time_steps * trotter_factor_odd * hamiltonian1)
    mpo_fill = matrix_to_mpo(matrix=fill, num_sites=1, site_shape=(dim, dim))
    mpo_fill_dag = matrix_to_mpo(matrix=fill_dagger, num_sites=1, site_shape=(dim, dim))

    # get operators acting on odd sites
    u_odd = mpo_on_odd(mpo_elem=mpo_elem_odd, mpo_fill=mpo_fill, num_sites=num_sites, dim=dim)
    u_odd_dag = mpo_on_odd(mpo_elem=mpo_elem_odd_dag, mpo_fill=mpo_fill_dag, num_sites=num_sites, dim=dim)

    # get operators acting on even sites
    u_even = mpo_on_even(mpo_elem=mpo_elem_even, num_sites=num_sites, dim=dim)
    u_even_dag = mpo_on_even(mpo_elem=mpo_elem_even_dag, num_sites=num_sites, dim=dim)

    # construct the complete time evolution operator for the state for a small time step
    if trotter_order == 1:
        u = mp.dot(u_even, u_odd)
        u_dagger = mp.dot(u_odd_dag, u_even_dag)
    else:  # trotter_order == 2
        u = mp.dot(mp.dot(u_odd, u_even), u_odd)
        u_dagger = mp.dot(mp.dot(u_odd_dag, u_even_dag), u_odd_dag)

    return u, u_dagger


def mpo_on_odd(mpo_elem, mpo_fill, num_sites, dim):
    """
    Creates the full MPO for the whole state from small MPOs acting on two adjacent sites,
        only for those acting on odd sites

    Args:
        mpo_elem (mpnum.MPArray): Elementary MPO acting on every pair of two adjacent sites
        mpo_fill (mpnum.MPArray): The MPO acting on the last site if it is not covered by an mpo_elem (for states
            with uneven numbers of sites)
        num_sites (int): The number of sites of the state to be evolved
        dim (int): The physical dimension of each of the legs of the state

    Returns:
        mpnum.MPArray: The full MPO acting on the whole state
    """

    odd = mp.chain(repeat(mpo_elem, int(num_sites / 2)))
    if (num_sites % 2 == 1):
        odd = mp.chain([odd, mpo_fill])
    return odd


def mpo_on_even(mpo_elem, num_sites, dim):
    """
    Creates the full MPO for the whole state from small MPOs acting on two adjacent sites,
        only for those acting on even sites

    Args:
        mpo_elem (mpnum.MPArray): Elementary MPO acting on every pair of two adjacent sites
        num_sites (int): The number of sites of the state to be evolved
        dim (int): The physical dimension of each of the legs of the state

    Returns:
        mpnum.MPArray: The full MPO acting on the whole state
    """

    even = mp.chain([mpo_identity(dim=dim)] + list(repeat(mpo_elem, int(num_sites / 2 - .5))))
    if (num_sites % 2 == 0):
        even = mp.chain([even, mpo_identity(dim=dim)])
    return even


def mpo_identity(dim):
    """
    Creates an identity operator acting on a single site, thereby not changing it

    Args:
        dim (int): Dimension of the physical legs

    Returns:
        mpnum.MPArray: Identity operator
    """

    id = np.identity(dim)
    mpo_id = mp.MPArray.from_array_global(id, ndims=2)

    return mpo_id


def matrix_to_mpo(matrix, num_sites, site_shape):
    """
    Generates a MPO from a MxN matrix in global form (not tested if this really works on non-square matrices).
    Args:
        matrix (numpy.ndarray): The matrix to be transformed to an MPO
        num_sites (int): The number of sites the MPO should have
        site_shape (list of int): The shape every single site of the resulting MPO should have, as used in mpnum.
            For example a site with two legs each of dimension two would look like (2,2)

    Returns:
        mpnum.MPArray: The MPO representing the matrix
    """

    newShape = []
    for j in sorted(site_shape):
        newShape = newShape + [j] * num_sites
    matrix = matrix.reshape(newShape)
    mpo = mp.MPArray.from_array_global(matrix, ndims=len(site_shape))
    return mpo
