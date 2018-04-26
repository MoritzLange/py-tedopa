"""
Time evolution of state given as MPS, MPO or PMPS via tMPS algorithm.

Functions to calculate the time evolution of an operator in MPS, MPO or PMPS form
from Hamiltonians acting on every single and every two adjacent sites.

"""
from collections import Counter
from itertools import repeat

import numpy as np
from scipy.linalg import expm

import mpnum as mp


def _sort_subsystems(subsystems, step_numbers):
    """
    List of subsystems for which the state should be returned at each time.
    This function just brings subsystems in the right form and sorts it.

    .. todo::
       Add doctest with example of the two types of allowed inputs.

    .. todo::
       Raise exception if length of the subsystem list (of first type) differs
       from the length of [step_numbers]

    Args:
        subsystems (list):
            Same as that described in :func:`evolve`
        step_numbers (list):
            The Trotter slices at which the subsystem is saved. Same as the
            first element from the output of _times_to_steps
    Returns:
        list[list[int]]:
            Sites for which the subsystem should be returned at the
            respective time,
    """
    if type(subsystems[0]) != list:
        subsystems = [subsystems] * len(step_numbers)
    subsystems = [x for _, x in sorted(zip(step_numbers, subsystems))]
    return subsystems


def _times_to_steps(ts, num_trotter_slices):
    """
    Calculate Trotter step numbers at which subsystem states should be saved

    If ts=[10, 25, 30] and num_trotter_slices was 100, then the result
    would be step_numbers=[33, 83, 100]

    .. todo::
       Convert this example into a doctest.

    Args:
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
        num_trotter_slices (int): Number of Trotter slices to be used for the
            largest t in ts.

    Returns:
        tuple[list[int], float]: step numbers, tau = maximal t /
        num_trotter_slices
    """
    ts.sort()
    tau = ts[-1] / num_trotter_slices
    step_numbers = [int(round(t / tau)) for t in ts]
    return step_numbers, tau


def _trotter_slice(hamiltonians, tau, num_sites, trotter_order, compr):
    """
    List of ordered operator exponentials for one Trotter slice

    The Trotter-Suzuki formula approximates the time-evlution during a single
    Trotter slice

    .. math::
       U(\\tau) =  \\text{e}^{\\mathrm{i}\\sum_{j=1}^m H_j \\tau},

    with

    .. math::
       {U}^\\prime(\\tau) =\\prod_{p=1}^N U_p,

    which is a product of :math:`N` operator exponentials

    .. math::
       U_p := {\\text{e}^{H_{j_p}\\tau_p}}

    of :math:`H_j`. Here :math:`\\{\\tau_p\\}` is a sequence of real numbers
    such that :math:`\\sum_p \\tau_p = \\tau`. This function returns the list of
    operators :math:`U_p` as MPOs.

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
            Order of Trotter-Suzuki decomposition to be used
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
    List of ordered operator exponentials for one second-order Trotter slice

    .. todo::
       Write more about second-order Trotter-Suzuki decomposition here. Explicit
       values of :math:`\\tau_p` from the above defition of trotter.

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
    h_single, h_adjacent = _get_h_list(hamiltonians=hamiltonians,
                                       num_sites=num_sites)
    dims = [len(h_single[i]) for i in range(len(h_single))]
    u_odd_list = _get_u_list_odd(dims, h_single, h_adjacent, tau=tau / 2)
    u_even_list = _get_u_list_even(dims, h_single, h_adjacent, tau=tau)
    u_odd = _u_list_to_mpo_odd(dims, u_odd_list, compr)
    u_even = _u_list_to_mpo_even(dims, u_even_list, compr)
    return [u_odd, u_even, u_odd]


def _trotter_four(hamiltonians, tau, num_sites, compr):
    """
    List of ordered operator exponentials for one fourth-order Trotter slice

    .. todo::
       Write more about fourth-order Trotter-Suzuki decomposition here. Explicit
       values of :math:`\\tau_p` from the above defition of Trotter.


    Args:
        hamiltonians (list):
            List of two lists of Hamiltonians, the Hamiltonians in the first
            acting on every single site, the Hamiltonians in the second acting
            on every pair of two adjacent sites
        tau (float):
            As defined in _times_to_steps()
        num_sites (int):
            Number of sites of the state to be evolved
        compr (dict):
            Parameters for the compression which is executed on every MPA during
            the calculations, except for the Trotter calculation where
            trotter_compr is used

    Returns:
        list[mpnum.MPArray]:
            The time evolution operator parts, which, applied one after another,
            give one Trotter slice
    """
    taus_for_odd = [tau * .5 / (4 - 4 ** (1 / 3)),
                    tau / (4 - 4 ** (1 / 3)),
                    tau * .5 * (1 - 3 / (4 - 4 ** (1 / 3)))]
    taus_for_even = [tau / (4 - 4 ** (1 / 3)),
                     tau * (1 - 4 / (4 - 4 ** (1 / 3)))]
    h_single, h_adjacent = _get_h_list(hamiltonians=hamiltonians,
                                       num_sites=num_sites)
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


def _get_h_list(hamiltonians, num_sites):
    """
    Convert given list of Hamiltonians into form suitable for exponentiation

    If only one Hamiltonian acting on every single site and one acting on every
    two adjacent sites is given, transform it into the form returned. If not,
    check whether the lengths of the lists match the number of sites.

    Args:
        hamiltonians (list):
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
    if type(hamiltonians[0]) is not list:
        hamiltonians = [list(repeat(hamiltonians[0], num_sites)),
                        list(repeat(hamiltonians[1], num_sites - 1))]
    elif (len(hamiltonians[0]) != num_sites) or (
            len(hamiltonians[1]) != num_sites - 1):
        raise ValueError(
            "Number of given Hamiltonians does not match number of sites")
    return hamiltonians[0], hamiltonians[1]


def _get_u_list_odd(dims, h_single, h_adjacent, tau):
    """
    Calculates individual operator exponentials of adjacent odd-even sites

    .. todo::
       Mention the explicit expressions of the unitaries in terms of the
       Hamiltonian terms.

    .. todo::
       Add doctest that gives an example of the inputs and outputs. Use four
       sites, ``H_single = Z, h_adj  = XX``

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
    Calculates individual operator exponentials of adjacent even-off sites

    .. todo::
       Mention the explicit expressions of the unitaries in terms of the
       Hamiltonian terms.

    .. todo::
       Add doctest that gives an example of the inputs and outputs. Use four
       sites, ``H_single = Z, h_adj  = XX``

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
    Transforms list of matrices on odd-even sites to MPOs acting on full stateself.

    .. todo::
       Give explicit form of the final MPO (tensor product of input matrices)

    Args:
        dims (list):
            List of dimensions of each site
        u_odd (list):
            List of time evolution operators acting on odd adjacent sites
        compr (dict): 
            Parameters for the compression which is executed on every MPA during
            the calculations, except for the Trotter calculation where
            trotter_compr is used

    Returns:
        mpnum.MPArray:
            The MPO for the full state acting on odd-even adjacent sites
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
    Transforms list of matrices on odd-even sites to MPOs acting on full state

    .. todo::
       Give explicit form of the final MPO (tensor product of input matrices)


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
            The MPO for the full state acting on even-odd adjacent sites
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
    Convert matrix to MPO

    Converts given NxN matrix in global form (probably also works for MxN) into
    an MPO with the given shape. The number of legs per site must be the same
    for all sites.

    .. todo::
       Check if it works for MxN and clarify the docstring above.

    .. todo::
       Add doctest. Can use to_array on the output MPO to display results.

    Args:
        matrix (numpy.ndarray):
            The matrix to be transformed to an MPO
        shape (list):
            The shape the single sites of the resulting MPO should have, as used
            in mpnum. For example three sites with two legs each might look like
            ``[[3, 3], [2, 2], [2, 2]]``. Format same as ``numpy.ndarray.shape``
        compr (dict):
            Parameters for the compression which is executed on every MPA during
            the calculations, except for the Trotter calculation where
            trotter_compr is used

    Returns:
        mpnum.MPArray:
            The MPO with shape ``shape`` representing the matrix
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


def normalize(state, method):
    """
    Normalize a state (hopefully in place)

    .. todo::
       Check if state is mutable and this operation in place. Then clear docstring

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
    Evolve a state using tMPS under given Hamiltonians with given parameters

    .. todo::
       Add some description about tmps here and in the module docstring. Add reference.

    .. todo:
       Raise exception if hamiltonians are not of the right dimension

    .. todo::
       Implement tracking of compression errors.

    Args:
        state (mpnum.MPArray):
            The state to be evolved in time. The state has to be an MPS, MPO or
            PMPS, depending on which method is chosen
        hamiltonians (list):
            Either a list containing the Hamiltonian acting on every single site
            and the Hamiltonian acting on every two adjacents sites, like
            ``[H_i, H_ij]``, or a list containing a list of Hamiltonians acting
            on the single sites and a list of Hamiltonians acting on each two
            adjacent sites, like ``[[h1, h2, h3, ...], [h12, h23, h34, ...]]``
        num_trotter_slices (int):
            Number of Trotter slices to be used for evolution over time equal to
            the largest t in ts.
        method (str):
            Which method to use. Either 'mps', 'mpo' or 'pmps'.
        trotter_compr (dict):
            Compression parameters used in the iterations of Trotter. If using
            variational compression, then ``Startmpa`` will be set by the
            algorithm, does not need to be specified.
        trotter_order (int):
            Order of Trotter-Suzuki decomposition to be used. Currently only 2
            and 4 are implemented
        compr (dict):
            Parameters for the compression which is executed on every MPA during
            the calculations, except for the Trotter calculation where
            trotter_compr is used
        ts (list[float]):
            The times for which the evolution should be computed and the state
            of the full system or a subsystem returned (i.e. it's reduced
            density matrix). The algorithm will calculate the
            evolution using the given number of Trotter steps for the largest
            number in ts. On the way there it will store the evolved states for
            smaller times. NB: Beware of memory overload since len(t) number of
            mpnum.MPArrays will be stored
        subsystems (list):
            A list defining for which subsystem the reduced density matrix or
            whether the full state should be returned for a time in ``ts``.
            This can be a list of the length same as that of ``ts`` looking
            like ``[[a1, b1], [a2, b2], ...]`` or just a list like ``[a, b]``.
            In the first case the respective subsystem for every entry in ts
            will be returned, in the second case the same subsystem will be
            returned for all entries in ``ts``. ``[a, b]`` will lead to a
            return of the reduced density matrix of the sites from ``a`` up to,
            but not including, ``b``. For example ``[0, 3]`` if the reduced
            density matrix of the first three sites shall be returned. A time
            can occur twice in ``ts`` and then different subsystems to be returned
            can be defined for that same time. If this parameter is omitted, the
            full system will be returned for every time in ``ts``.
        v (bool):
            Verbose or not verbose (will print what is going on vs. won't print
            anything)

    Returns:
        list[list[float], list[list[int]], list[mpnum.MPArray], list[float], list[float]]:
            A list with five items: (i) The list of times for which the density
            matrices have been computed (ii) The list indicating which
            subsystems of the system are returned at the respective time of the
            first list (iii) The list of density matrices as MPO or PMPS as
            mpnum.MPArray, depending on the input "method". If that was MPS, the
            full states will still be MPSs, the reduced ones will be MPOs. (iv)
            The errors due to compression during the procedure (v) The order of
            errors due to application of Trotter-Suzuki decomposition during the
            evolution.

    """
    state.compress(**compr)
    state = normalize(state, method)
    if len(state) < 3:
        raise ValueError("State has too few sites")
    if (np.array(ts) == 0).all():
        raise ValueError(
            "No time evolution requested by the user. Check your input 't'")
    if subsystems == None:
        subsystems = [0, len(state)]
    step_numbers, tau = _times_to_steps(ts, num_trotter_slices)
    subsystems = _sort_subsystems(subsystems, step_numbers)

    us = _trotter_slice(hamiltonians=hamiltonians, tau=tau,
                        num_sites=len(state), trotter_order=trotter_order,
                        compr=compr)
    if v:
        print("Time evolution operator for Trotter slice calculated, "
              "starting "
              "Trotter iterations...")
    return _time_evolution(state, us, step_numbers, subsystems, tau, method,
                           trotter_compr, v)


def _time_evolution(state, us, step_numbers, subsystems, tau, method,
                    trotter_compr, v):
    """
    Implements time-evolution via Trotter-Suzuki decomposition

    Args:
        state (mpnum.MPArray):
            The state to be evolved in time
        us (list[mpnum.MPArray]):
            List of ordered operator exponentials for a single Trotter slice
        step_numbers (list[int]):
            List of time steps as generated by _times_to_steps()
        subsystems (list[list[int]]):
            Sites for which the subsystem states should be returned at the
            respective times
        tau (float):
            Duration of one Trotter slice. As defined in _times_to_steps()
        method (str):
            Which method to use. Either 'mps', 'mpo' or 'pmps'.
        trotter_compr (dict):
            Compression parameters used in the iterations of Trotter-Suzuki
            decomposition.
        v (bool):
            Verbose (Yes) or not verbose (No). Will print what is going on vs.
            will not print anything.

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
    c = Counter(step_numbers)

    times = []
    states = []
    compr_errors = []
    trot_errors = []

    var_compression = False
    if trotter_compr['method'] == 'var':
        var_compression = True
    accumulated_overlap = 1
    accumulated_trotter_error = 0

    for i in range(step_numbers[-1] + 1):
        for j in range(c[i]):
            _append(times, states, compr_errors, trot_errors, tau, i, j,
                    step_numbers, subsystems, state, accumulated_overlap,
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


def _append(times, states, compr_errors, trot_errors, tau, i, j, step_numbers,
            subsystems, state, accumulated_overlap,
            accumulated_trotter_error, method):
    """
    Function to append time evolved state etc to output of _time_evolution()

    Args:
        times (list[float]):
            List containing the times to which the states are evolved
        states (list[mpnum.MPArray]):
            List containing the evolved states
        compr_errors (list[float]):
            List containing the respective compression errors
        trot_errors (list[float]):
            List containing the respective Trotter errors
        tau (float):
            The time of one Trotter slice
        i (int):
            Number indicating which is the current Trotter slice
        j (int):
            Number indicating how many times a state related to the
            current i has been appended already
        step_numbers (list[int]):
            List containing the time steps
        subsystems (list[list[int]]):
            List of sites for which the subsystem should be returned at the
            respective time
        state (mpnum.MPArray):
            The current state
        accumulated_overlap (float):
            The accumulated overlap error
        accumulated_trotter_error (float):
            The accumulated Trotter error
        method (str):
            Method to use as defined in evolve()

    Returns:
        None: Nothing, changes happen in place

    """
    times.append(tau * i)
    sites = [x for t, x in zip(step_numbers, subsystems) if t == i][j]
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


if __name__ == "__main__":
    import doctest

    doctest.testmod()
