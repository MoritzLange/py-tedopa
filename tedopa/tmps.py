"""
Functions to calculate the time evolution of an operator in MPO or PMPS form
from Hamiltonians acting on every single and every two adjacent sites.
"""
from itertools import repeat
from collections import Counter

import numpy as np
from scipy.linalg import expm

import mpnum as mp


def _times_to_steps(times, subsystems,
                    num_trotter_slices):
    """
    Calculate the respective Trotter steps for the given times for which
    evolution should be computed.
    If times=[10, 25, 30] and num_trotter_slices was 100, then the result
    would be times=[33, 83, 100]

    Args:
        times (list[float]):
            The times for which the evolution should be computed and the
            state of the full system or a subsystem returned (i.e. it's reduced
            density matrix (for now only works with method='mpo'. If the
            method is not mpo, omit subsystems)). The algorithm will
            calculate the evolution using the given number of Trotter steps
            for the largest number in ts. On the way there it will store the
            evolved states for smaller times.
            NB: Beware of memory overload since len(t)
            mpnum.MPArrays will be stored
        subsystems (list):
            A list defining for which subsystem the reduced density matrix or
            whether the full state should be returned for a time in ts.
            This can be a list of the length of ts looking like
            [[a1, b1], [a2, b2], ...] or just a list like [a, b]. In the first
            case the respective subsystem for every entry in ts
            will be returned, in the second case the same subsystem will be
            returned for all entries in ts.
            [a, b] will lead to a return of the reduced density matrix of the
            sites from a up to, but not including, b. For example [0, 3] if the
            reduced density matrix of the first three sites shall be returned.
            A time can occur twice in ts and then different subsystems to
            be returned can be defined for that same time.
            If this parameter is omitted, the full system will be returned for
            every time in ts.
        num_trotter_slices (int): Number of Trotter slices to be used for the
            largest t in ts.

    Returns:
        tuple[list[int], list[list[int]], float]: times, sites for which the
        subsystem should be returned at the respective time,
        tau = maximal t / num_trotter_slices
    """
    if type(subsystems[0]) != list:
        subsystems = [subsystems] * len(times)
    subsystems = [x for _, x in sorted(zip(times, subsystems))]
    times.sort()
    tau = times[-1] / num_trotter_slices
    ts = [int(round(t / tau)) for t in times]
    return ts, subsystems, tau


def _trotter_slice(hamiltonians, tau, num_sites, trotter_order, compr):
    """
    Calculate the time evolution operator u for the respective trotter order for
    one trotter slice.

    Args:
        hamiltonians (list):
            List of two lists of Hamiltonians, the Hamiltonians in the first
            acting on every single site, the Hamiltonians in the second acting
            on every pair of two adjacent sites
        tau (float):
            As defined in _times_to_steps()
        num_sites (int):
            Number of sites of the state to be evolved
        trotter_order (int):
            Order of trotter to be used
        compr (dict): Parameters for the compression which is executed on every
            MPA during the calculations, except for the Trotter calculation
            where trotter_compr is used

    Returns:
        list[mpnum.MPArray]:
            The time evolution operator parts, which, applied one after
            another, give one Trotter slice
    """
    if trotter_order == 2:
        return _trotter_two(hamiltonians, tau, num_sites, compr)
    if trotter_order == 4:
        return _trotter_four(hamiltonians, tau, num_sites, compr)
    else:
        raise ValueError("Trotter order " + str(trotter_order) +
                         " is currently not implemented.")


def _trotter_two(hamiltonians, tau, num_sites, compr):
    """
    Calculate the time evolution operator u, comprising even and odd terms, for
    one Trotter slice and Trotter of order 2.

    Args:
        hamiltonians (list):
            List of two lists of Hamiltonians, the Hamiltonians in the first
            acting on every single site, the Hamiltonians in the second acting
            on every pair of two  adjacent sites
        tau (float):
            As defined in _times_to_steps()
        num_sites (int):
            Number of sites of the state to be evolved
        compr (dict): Parameters for the compression which is executed on every
            MPA during the calculations, except for the Trotter calculation
            where trotter_compr is used

    Returns:
        list[mpnum.MPArray]:
            The time evolution operator parts, which, applied one after
            another, give one Trotter slice
    """
    h_single, h_adjacent = _get_h_list(hs=hamiltonians, num_sites=num_sites)
    dims = [len(h_single[i]) for i in range(len(h_single))]
    u_odd_list = _get_u_list_odd(dims, h_single, h_adjacent, tau=tau / 2)
    u_even_list = _get_u_list_even(dims, h_single, h_adjacent, tau=tau)
    u_odd = _u_list_to_mpo_odd(dims, u_odd_list, compr)
    u_even = _u_list_to_mpo_even(dims, u_even_list, compr)
    return [u_odd, u_even, u_odd]


def _trotter_four(hamiltonians, tau, num_sites, compr):
    """
    Calculate the time evolution operator u, comprising even and odd terms, for
    one Trotter slice and Trotter of order 4.

    Args:
        hamiltonians (list):
            List of two lists of Hamiltonians, the Hamiltonians in the first
            acting on every single site, the Hamiltonians in the second acting
            on every pair of two adjacent sites
        tau (float): As defined in _times_to_steps()
        num_sites (int): Number of sites of the state to be evolved
        compr (dict): Parameters for the compression which is executed on every
            MPA during the calculations, except for the Trotter calculation
            where trotter_compr is used

    Returns:
        list[mpnum.MPArray]:
            The time evolution operator parts, which, applied one after
            another, give one Trotter slice
    """
    taus_for_odd = [tau * .5 / (4 - 4 ** (1 / 3)),
                    tau / (4 - 4 ** (1 / 3)),
                    tau * .5 * (1 - 3 / (4 - 4 ** (1 / 3)))]
    taus_for_even = [tau / (4 - 4 ** (1 / 3)),
                     tau * (1 - 4 / (4 - 4 ** (1 / 3)))]
    h_single, h_adjacent = _get_h_list(hs=hamiltonians, num_sites=num_sites)
    dims = [len(h_single[i]) for i in range(len(h_single))]
    u_odd_lists = [_get_u_list_odd(dims, h_single, h_adjacent, t) for t in
                   taus_for_odd]
    u_even_lists = [_get_u_list_even(dims, h_single, h_adjacent, t) for t in
                    taus_for_even]
    multiplication_order = [0, 0, 1, 0, 2, 1, 2, 0, 1, 0, 0]
    us = []
    for i in range(11):
        if i % 2 == 1:
            us = us + [_u_list_to_mpo_even(dims,
                                           u_even_lists[
                                               multiplication_order[i]],
                                           compr)]
        else:
            us = us + [_u_list_to_mpo_odd(
                dims, u_odd_lists[multiplication_order[i]], compr)]
    return us


def _get_h_list(hs, num_sites):
    """
    If only one Hamiltonian acting on every single site and one acting on every
    two adjacent sites is given, transform it into the form returned. If not,
    check whether the lengths of the lists match the number of sites.

    Args:
        hs (list):
            Hamiltonians as in evolve()
        num_sites (int):
            Number of sites of the state to be evolved

    Returns:
        list[list[numpy.ndarray], list[numpy.ndarray]]:
            A list with two items: The first is a list of Hamiltonians acting
            on the single sites, like [h1, h2, h3, ...] and the second is a list
            of Hamiltonians acting on each two adjacent sites, like [h12, h23,
            h34, ...]
    """
    if type(hs[0]) is not list:
        hs = [list(repeat(hs[0], num_sites)),
              list(repeat(hs[1], num_sites - 1))]
    elif (len(hs[0]) != num_sites) or (len(hs[1]) != num_sites - 1):
        raise ValueError(
            "Number of given Hamiltonians does not match number of sites")
    return hs[0], hs[1]


def _get_u_list_odd(dims, h_single, h_adjacent, tau):
    """
    Calculates time evolution operators for adjacent odd sites from
    Hamiltonians.

    Args:
        dims (list):
            The dimensions of the single sites of the state U will be applied to
        h_single (list):
            The Hamiltonians acting on every single site
        h_adjacent (list):
            The Hamiltonians acting on every two adjacent sites
        tau (float):
            The time step for the time evolution of U

    Returns:
        list[numpy.ndarray]:
            List of operators acting on odd adjacent sites, like [u12, u34, ...]
    """
    h_2sites = [
        1 / 2 * (np.kron(h_single[i], np.identity(dims[i + 1])) +
                 np.kron(np.identity(dims[i]), h_single[i + 1]))
        for i in range(0, len(h_single) - 1, 2)]
    u_odd = list(expm(-1j * tau * (h + h_2sites[i]))
                 for i, h in enumerate(h_adjacent[::2]))
    if len(dims) % 2 == 1:
        u_odd = u_odd + [expm(-1j * tau * h_single[-1] / 2)]
    return u_odd


def _get_u_list_even(dims, h_single, h_adjacent, tau):
    """
    Calculates time evolution operators for adjacent even sites from
    Hamiltonians.

    Args:
        dims (list):
            The dimensions of the single sites of the state U will be applied to
        h_single (list):
            The Hamiltonians acting on every single site
        h_adjacent (list):
            The Hamiltonians acting on every two adjacent sites
        tau (float):
            The time step for the time evolution of U

    Returns:
        list[numpy.ndarray]:
            List of operators acting on even adjacent sites, like
            [u23, u45, ...]
    """
    h_2sites = [
        1 / 2 * (np.kron(h_single[i], np.identity(dims[i + 1])) +
                 np.kron(np.identity(dims[i]), h_single[i + 1]))
        for i in range(1, len(h_single) - 1, 2)]
    u_even = list(expm(-1j * tau * (h + h_2sites[i])) for i, h in
                  enumerate(h_adjacent[1::2]))
    u_even = [expm(-1j * tau * h_single[0] / 2)] + u_even
    if len(dims) % 2 == 0:
        u_even = u_even + [expm(-1j * tau * h_single[-1] / 2)]
    return u_even


def _u_list_to_mpo_odd(dims, u_odd, compr):
    """
    Transforms a list of matrices for time evolution on odd sites to operators
    acting on the full state.

    Args:
        dims (list):
            List of dimensions of each site
        u_odd (list):
            List of time evolution operators acting on odd adjacent sites
        compr (dict): Parameters for the compression which is executed on every
            MPA during the calculations, except for the Trotter calculation
            where trotter_compr is used

    Returns:
        mpnum.MPArray:
            The time evolution MPO for the full state acting on odd adjacent
            sites
    """
    if len(dims) % 2 == 1:
        last_h = u_odd[-1]
        u_odd = u_odd[:-1]
    odd = mp.chain(matrix_to_mpo(
        u, [[dims[2 * i]] * 2, [dims[2 * i + 1]] * 2], compr)
                   for i, u in enumerate(u_odd))
    if len(dims) % 2 == 1:
        odd = mp.chain([odd, matrix_to_mpo(last_h, [[dims[-1]] * 2], compr)])
    return odd


def _u_list_to_mpo_even(dims, u_even, compr):
    """
    Transforms a list of matrices for time evolution on even sites to operators
    acting on the full state.

    Args:
        dims (list):
            List of dimensions of each site
        u_even (list):
            List of time evolution operators acting on even adjacent sites
        compr (dict): Parameters for the compression which is executed on every
            MPA during the calculations, except for the Trotter calculation
            where trotter_compr is used

    Returns:
        mpnum.MPArray:
            The time evolution MPO for the full state acting on even adjacent
            sites
    """
    if len(dims) % 2 == 0:
        last_h = u_even[-1]
        u_even = u_even[:-1]
    even = mp.chain(matrix_to_mpo(
        u, [[dims[2 * i + 1]] * 2, [dims[2 * i + 2]] * 2], compr)
                    for i, u in enumerate(u_even[1::]))
    even = mp.chain([matrix_to_mpo(u_even[0], [[dims[0]] * 2], compr), even])
    if len(dims) % 2 == 0:
        even = mp.chain([even, matrix_to_mpo(last_h, [[dims[-1]] * 2], compr)])
    return even


def matrix_to_mpo(matrix, shape, compr=None):
    """
    Generates a MPO from a NxN matrix in global form (probably also works for
    MxN). The number of legs per site must be the same for all sites.

    Args:
        matrix (numpy.ndarray):
            The matrix to be transformed to an MPO
        shape (list):
            The shape the single sites of the resulting MPO should have, as used
            in mpnum. For example three sites with two legs each might look like
            [[3, 3], [2, 2], [2, 2]]
        compr (dict): Parameters for the compression which is executed on every
            MPA during the calculations, except for the Trotter calculation
            where trotter_compr is used

    Returns:
        mpnum.MPArray:
            The MPO representing the matrix
    """
    if compr == None:
        compr = dict(method='svd', relerr=1e-6)
    num_legs = len(shape[0])
    if not (np.array([len(shape[i]) for i in
                      range(len(shape))]) == num_legs).all():
        raise ValueError("Not all sites have the same number of physical legs")
    newShape = []
    for i in range(num_legs):
        for j in range(len(shape)):
            newShape = newShape + [shape[j][i]]
    matrix = matrix.reshape(newShape)
    mpo = mp.MPArray.from_array_global(matrix, ndims=num_legs)
    mpo.compress(**compr)
    return mpo


# Does this work, is state mutable and this operation in place?
def normalize(state, method):
    """
    Normalize a state (hopefully in place)

    Args:
        state (mpnum.MPArray): The state to be normalized
        method (str): Whether it is a MPS, MPO or PMPS state

    Returns:
        mpnum.MPArray: The normalized state
    """
    if method == 'pmps' or method == 'mps':
        state = state / mp.norm(state)
    if method == 'mpo':
        state = state / mp.trace(state)
    return state


def evolve(state, hamiltonians, num_trotter_slices, method, trotter_compr,
           trotter_order, compr, ts, subsystems=None, v=False):
    """
    Evolve a state using tMPS.

    Args:
        state (mpnum.MPArray):
            The state to be evolved in time(the density matrix, not state
            vector). The state has to be an MPS, MPO or PMPS, depending on which
            method is chosen
        hamiltonians (list):
            Either a list containing the Hamiltonian acting on every single site
            and the Hamiltonian acting on every two adjacents sites, like[H_i,
            H_ij], or a list containing a list of Hamiltonians acting on the
            single sites and a list of Hamiltonians acting on each two adjacent
            sites, like [[h1, h2, h3, ...], [h12, h23, h34, ...]]
        num_trotter_slices (int): Number of Trotter slices to be used for the
            largest t in ts.
        method (str):
            Which method to use. Either 'mps', 'mpo' or 'pmps'.
        trotter_compr (dict):
            Compression parameters used in the iterations of Trotter.
            Startmpa will be set by the algorithm, does not need to be
            specified.
        trotter_order (int):
            Order of trotter to be used. Currently only 2 and 4
            are implemented
        compr (dict): Parameters for the compression which is executed on every
            MPA during the calculations, except for the Trotter calculation
            where trotter_compr is used
        ts (list[float]):
            The times for which the evolution should be computed and the
            state of the full system or a subsystem returned (i.e. it's reduced
            density matrix (for now only works with method='mpo'. If the
            method is not mpo, omit subsystems)). The algorithm will
            calculate the evolution using the given number of Trotter steps
            for the largest number in ts. On the way there it will store the
            evolved states for smaller times.
            NB: Beware of memory overload since len(t)
            mpnum.MPArrays will be stored
        subsystems (list):
            A list defining for which subsystem the reduced density matrix or
            whether the full state should be returned for a time in ts.
            This can be a list of the length of ts looking like
            [[a1, b1], [a2, b2], ...] or just a list like [a, b]. In the first
            case the respective subsystem for every entry in ts
            will be returned, in the second case the same subsystem will be
            returned for all entries in ts.
            [a, b] will lead to a return of the reduced density matrix of the
            sites from a up to, but not including, b. For example [0, 3] if the
            reduced density matrix of the first three sites shall be returned.
            A time can occur twice in ts and then different subsystems to
            be returned can be defined for that same time.
            If this parameter is omitted, the full system will be returned for
            every time in ts.
        v (bool): Verbose or not verbose (will print what is going on vs.
            won't print anything)

    Returns:
        list[list[float], list[list[int]], list[mpnum.MPArray], list[float], list[float]]:
            A list with five items: (i)The list of times for which the density
            matrices have been computed (ii) The list indicating which
            subsystems of the system are returned at the respective
            time of the first list (iii) The list of density matrices as MPO
            or PMPS as mpnum.MPArray, depending on the input "method". If
            that was MPS, the full states will still be MPSs, the reduced
            ones will be MPOs. (iv) The errors due to compression during the
            procedure (v) The order of errors due to application of Trotter
            during the procedure
    """
    # ToDo: Maybe add support for hamiltonians depending on time
    # ToDo:     (if not too complicated)
    # ToDo: Make sure the hamiltonians are of the right dimension
    # ToDo: Implement tracking of errors properly
    state.compress(**compr)
    state = normalize(state, method)
    if len(state) < 3:
        raise ValueError("State has too few sites")
    if (np.array(ts) == 0).all():
        raise ValueError(
            "No time evolution requested by the user. Check your input 't'")
    if subsystems == None:
        subsystems = [0, len(state)]
    ts, subsystems, tau = _times_to_steps(ts, subsystems, num_trotter_slices)
    us = _trotter_slice(hamiltonians=hamiltonians, tau=tau,
                        num_sites=len(state), trotter_order=trotter_order,
                        compr=compr)
    if v:
        print("Time evolution operator for Trotter slice calculated, "
              "starting "
              "Trotter iterations...")
    return _time_evolution(state, us, ts, subsystems, tau, method,
                           trotter_compr, v)


def _time_evolution(state, us, ts, subsystems, tau, method, trotter_compr, v):
    """
    Do the actual time evolution

    Args:
        state (mpnum.MPArray):
            The state to be evolved in time
        us (list[mpnum.MPArray]):
            The time evolution operator parts, which, applied one after
            another, give one Trotter slice
        ts (list[int]):
            List of time steps as generated by _times_to_steps()
        subsystems (list[list[int]]):
            Sites for which the subsystem should be returned at the respective
            time
        tau (float):
            As defined in _times_to_steps()
        method (str):
            Method to use as defined in evolve()
        trotter_compr (dict):
            Compression parameters used in the iterations of Trotter
        v (bool): Verbose or not verbose (will print what is going on vs.
            won't print anything)

    Returns:
        list[list[float], list[list[int]], list[mpnum.MPArray], list[float], list[float]]:
            A list with five items: (i)The list of times for which the density
            matrices have been computed (ii) The list indicating which
            subsystems of the system are returned at the respective
            time of the first list (iii) The list of density matrices as MPO
            or PMPS as mpnum.MPArray, depending on the input "method". If
            that was MPS, the full states will still be MPSs, the reduced
            ones will be MPOs. (iv) The errors due to compression during the
            procedure (v) The order of errors due to application of Trotter
            during the procedure
    """
    c = Counter(ts)

    times = []
    states = []
    compr_errors = []
    trot_errors = []

    var_compression = False
    if trotter_compr['method'] == 'var':
        var_compression = True
    accumulated_overlap = 1
    accumulated_trotter_error = 0

    for i in range(ts[-1] + 1):
        for j in range(c[i]):
            _append(times, states, compr_errors, trot_errors, tau, i, j, ts,
                    subsystems, state, accumulated_overlap,
                    accumulated_trotter_error, method)
        for u in us:
            if var_compression:
                trotter_compr['startmpa'] = mp.MPArray.copy(state)
            state = mp.dot(u, state)
            accumulated_overlap *= state.compress(**trotter_compr)
        if method == 'mpo':
            for u in us:
                if var_compression:
                    trotter_compr['startmpa'] = mp.MPArray.copy(state)
                state = mp.dot(state, u.T.conj())
                accumulated_overlap *= state.compress(**trotter_compr)
        state = normalize(state, method)
        accumulated_trotter_error += tau ** 3
        if v and np.sqrt(i) % 1 == 0 and i != 0:
            print(str(i) + " Trotter iterations finished...")
    if v:
        print("Done with time evolution")
    return times, subsystems, states, compr_errors, trot_errors


def _append(times, states, compr_errors, trot_errors, tau, i, j, ts,
            subsystems, state, accumulated_overlap,
            accumulated_trotter_error, method):
    """
    Function to append the time evolved state and related information to the
    output lists of _time_evolution()

    Args:
        times (list[float]): List containing the times to which the states are
            evolved
        states (list[mpnum.MPArray]): List containing the evolved states
        compr_errors (list[float]): List containing the respective compression
            errors
        trot_errors (list[float]): List containing the respective Trotter errors
        tau (float): The time of one Trotter slice
        i (int): Number indicating which is the current Trotter slice
        j (int): Number indicating how many times a state related to the
            current i has been appended already
        ts (list[int]): List containing the time steps
        subsystems (list[list[int]]): List of sites for which the subsystem
            should be returned at the respective time
        state (mpnum.MPArray): The current state
        accumulated_overlap (float): The accumulated overlap error
        accumulated_trotter_error (float): The accumulated Trotter error
        method (str): Method to use as defined in evolve()

    Returns:
        None: Nothing, changes happen in place

    """
    times.append(tau * i)
    sites = [x for t, x in zip(ts, subsystems) if t == i][j]
    if sites == [0, len(state)]:
        states.append(state.copy())
    elif method == 'mpo':
        states.append(next(
            mp.reductions_mpo(state, sites[1] - sites[0], [sites[0]])))
    elif method == 'pmps':
        states.append(next(
            mp.reductions_pmps(state, sites[1] - sites[0], [sites[0]])))
    elif method == 'mps':
        states.append(next(
            mp.reductions_mps_as_mpo(state, sites[1] - sites[0], [sites[0]])))
    compr_errors.append(accumulated_overlap)
    trot_errors.append(accumulated_trotter_error)
