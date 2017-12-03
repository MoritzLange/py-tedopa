"""
Functions to calculate recursion coefficients.

Author:
    Moritz Lange

Date:
    28/07/2017
"""

import mpnum as mp
import numpy as np
from scipy.linalg import expm
from itertools import repeat


def evolve(state, hamiltonians, t, num_trotter_slices, trotter_order, method):
    """
    Evolve a state using tMPS.

    Args:
        state (mpnum.MPArray): The state to be evolved in time (the density matrix, not wave function).
            It is assumed, that every site has two legs and all legs of the state are of the same physical dimension
        hamiltonians (list of numpy.ndarray): List of two Hamiltonians, the first acting on every single site, the
            second acting on every pair of two adjacent sites
        t (float): The time for which the evolution should be computed
        num_trotter_slices (int): The number of time steps or Trotter slices for the time evolution
        trotter_order (int): Which order of trotter should be used. Currently implemented are only 1 and 2
        method (str): Which method to use. Either 'mpo' or 'pmps', leading to computations based on matrix product
            operators for density matrices or purified states for density matrices

    Returns:
        mpnum.MPArray: The evolved state's density matrix
    """
    # ToDo: Maybe add support for states with different physical dimensions at each site (if not too complicated)
    # ToDo: Maybe add support for hamiltonians depending on time (if not too complicated)
    # ToDo: Make sure the hamiltonians are of the right dimension
    # ToDo: Implement evolve_pmps()

    # for speedup compress everything as much as possible
    state.compress(relerr=1e-15)

    if len(state) < 3:
        raise ValueError("State has too few sites")

    if t == 0:
        return state

    if method == 'mpo':
        evolved_state = evolve_mpo(state=state, hamiltonians=hamiltonians, t=t, num_trotter_slices=num_trotter_slices,
                                   trotter_order=trotter_order)
    elif method == 'pmps':
        evolved_state = evolve_pmps(state=state, hamiltonians=hamiltonians, t=t, num_trotter_slices=num_trotter_slices,
                                    trotter_order=trotter_order)
    else:
        return state

    return evolved_state


def evolve_mpo(state, hamiltonians, t, num_trotter_slices, trotter_order):
    """
    Evolve the state using MPO representation instead of purified states, for further documentation see evolve().

    Args:
        See evolve()

    Returns:
        mpnum.MPArray: The evolved state's density matrix
    """

    initialState = state.copy()

    # error acceptable for compression during each Trotter step
    relerr = 1e-8
    # max ranks acceptable
    maxRanks = 100

    u = trotter(hamiltonians=hamiltonians, t=t, num_trotter_slices=num_trotter_slices, trotter_order=trotter_order,
                num_sites=len(state))
    u_dagger = u.adj()

    for i in range(num_trotter_slices):
        state = mp.dot(mp.dot(u, state), u_dagger)
        # in contrast to every other compression in this file, this one is not for speedup but part of the trotter algorithm
        state.compress(method='svd', relerr=relerr)
        # implement something to store the error here
        if max(state.ranks) > maxRanks:
            raise RecursionError("Max ranks for MPO exceeded")

    return state


def evolve_pmps(state, hamiltonians, t, num_trotter_slices, trotter_order):
    """
        Evolve the state using purified state representation instead of MPOs, for further documentation see evolve().

        Args:
            See evolve()

        Returns:
            mpnum.MPArray: The evolved state's density matrix
        """
    # To be implemented, for now just redirect to evolve_mpo()
    return evolve_mpo(state=state, hamiltonians=hamiltonians, t=t, num_trotter_slices=num_trotter_slices,
                      trotter_order=trotter_order)


def trotter(hamiltonians, t, num_trotter_slices, trotter_order, num_sites):
    """
    Calculate the time evolution operators u and u_dagger, each comprising even and odd terms, for one Trotter slice.

    Args:
        hamiltonians (list of numpy.ndarray): List of two Hamiltonians, the first acting on every single site, the
            second acting on every pair of two adjacent sites
        t (float): The time for which the evolution should be computed
        num_trotter_slices (int): The number of time steps or Trotter slices for the time evolution
        trotter_order (int): Which order of trotter should be used. Currently implemented are only 1 and 2
        num_sites (int): Number of sites of the state to be evolved

    Returns:
        mpnum.MPArray: The time evolution operators u and u_dagger for one Trotter slice
    """
    hamiltonian1 = hamiltonians[0]  # the hamiltonian acting on every site
    hamiltonian2 = hamiltonians[1]  # the hamiltonian acting on every two adjacent sites

    dim = int(len(hamiltonian1))  # The dimension of the physical legs of the state

    # hamiltonian1 acting on two sites
    hamiltonian1_2 = np.kron(hamiltonian1, np.identity(dim)) + np.kron(np.identity(dim), hamiltonian1)

    trotter_factor_even = 1

    if trotter_order == 1:
        trotter_factor_odd = 1
    else:  # trotter_order == 2
        trotter_factor_odd = .5

    # from given hamiltonian generate time evolution operator "element" acting on two adjacent sites for a small time step
    # for the odd sites also add the hamiltonians acting on every single site individually (but only for the odd ones, otherwise
    # they'd be applied twice
    element_even = expm(-1j * t * 1 / num_trotter_slices * trotter_factor_even * hamiltonian2)
    element_odd = expm(-1j * t * 1 / num_trotter_slices * trotter_factor_odd * (hamiltonian1_2 + hamiltonian2))

    mpo_elem_even = matrix_to_mpo(matrix=element_even, num_sites=2, site_shape=(dim, dim))
    mpo_elem_odd = matrix_to_mpo(matrix=element_odd, num_sites=2, site_shape=(dim, dim))

    # generate time evolution operator "fill" acting on the last site if num_sites is odd (and therefore this last site
    # is not covered by mpo_elem_odd)
    fill = expm(-1j * t * 1 / num_trotter_slices * trotter_factor_odd * hamiltonian1)
    mpo_fill = matrix_to_mpo(matrix=fill, num_sites=1, site_shape=(dim, dim))

    # get operators acting on odd sites
    u_odd = mpo_on_odd(mpo_elem=mpo_elem_odd, mpo_fill=mpo_fill, num_sites=num_sites)

    # get operators acting on even sites
    u_even = mpo_on_even(mpo_elem=mpo_elem_even, num_sites=num_sites, dim=dim)

    # construct the complete time evolution operator for the state for a small time step
    if trotter_order == 1:
        u = mp.dot(u_even, u_odd)
    else:  # trotter_order == 2
        u = mp.dot(mp.dot(u_odd, u_even), u_odd)

    # and compress the product
    u.compress(relerr=1e-15)

    return u


def mpo_on_odd(mpo_elem, mpo_fill, num_sites):
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
    odd.compress(relerr=1e-15)

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
    even.compress(relerr=1e-15)

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
    mpo_id.compress(relerr=1e-15)

    return mpo_id


def matrix_to_mpo(matrix, num_sites, site_shape):
    """
    Generates a MPO from a MxN matrix in global form (not tested if this really works on non-square matrices).
    Args:
        matrix (numpy.ndarray): The matrix to be transformed to an MPO
        num_sites (int): The number of sites the MPO should have
        site_shape (tuple of int): The shape every single site of the resulting MPO should have, as used in mpnum.
            For example a site with two legs each of dimension two would look like (2,2)

    Returns:
        mpnum.MPArray: The MPO representing the matrix
    """

    newShape = []
    for j in sorted(site_shape):
        newShape = newShape + [j] * num_sites
    matrix = matrix.reshape(newShape)
    mpo = mp.MPArray.from_array_global(matrix, ndims=len(site_shape))
    mpo.compress(relerr=1e-15)

    return mpo
