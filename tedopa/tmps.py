"""
This module provides functions for time evolution of a state given as MPS,
MPO or PMPS via the tMPS algorithm.

This is based on functions which calculate the time evolution of an operator in
MPS, MPO or PMPS form from Hamiltonians acting on every single and every two
adjacent sites.

tMPS is a method to evolve a one dimensional quantum state, represented
as an MPS, MPO or PMPS, in time. It requires that the Hamiltonian is only
comprised of terms acting on single sites or pairs of adjacent sites of
the state. This allows the Hamiltonian :math:`H` to be written as

.. math::
    H = \\sum_j h_{j, j+1},

where :math:`j` is the index of the respective site in the state.
These components can be grouped into those acting on even and those acting
on odd sites, leading to a time evolution operator

.. math::
    U(\\tau) = \\text{e}^{\mathrm{i}(H_{\\text{even}}+H_{\\text{
    odd}})\\tau},

with

.. math::
    H_{\\text{even}} = \\sum_{j\\text{ even}} h_{j, j+1}
.. math::
    H_{\\text{odd}} = \\sum_{j\\text{ odd}} h_{j, j+1}

This allows to perform Trotter-Suzuki decompositions of :math:`U`,
for example of second order:

.. math::
    U(\\tau) = \\text{e}^{\mathrm{i} H_{\\text{odd}} \\tau/2} \\text{e}^{\mathrm{i}
    H_{\\text{even}} \\tau} \\text{e}^{\mathrm{i} H_{\\text{odd}} \\tau/2}
    + \\mathcal{O}(\\tau^3).

These decompositions provide the advantage that :math:`U` does not need to
be calculated as a whole matrix, which could potentially become way too
big. Since the elements within :math:`H_{\\text{even}}` and those within
:math:`H_{\\text{odd}}` commute, :math:`U` can be broken up into smaller pieces
which a computer can handle even for very large systems.

For more information, see chapter 7 in Schollwöck’s paper Annals of
Physics 326, 96-192 (2011); doi: 10.1016/j.aop.2010.09.012

In this file, ``evolve()`` is the main function to be called to evolve a
state in time. It will itself call ``_trotter_slice()`` which will call
``_trotter_two()`` or ``_trotter_four()`` to calculate the :math:`U(\\tau)`
representing one Trotter slice. When that is done, ``evolve()`` will take it
and pass it on to ``_time_evolution()`` which will then go through the
Trotter iterations, thus actually evolving the state in time, and store the
requested results on the way.
"""
from collections import Counter
from itertools import repeat

import numpy as np
from scipy.linalg import expm

import mpnum as mp


def _get_subsystems_list(subsystems, len_step_numbers):
    """
    This function just brings subsystems, which indicates which subsystem
    should be returned at the respective step number, in the right form.

    Args:
        subsystems (list):
            Same as that described in :func:`evolve`
        len_step_numbers (int):
            The length of the array containing the step numbers for which the
            evolved state is to be stored.
    Returns:
        list[list[int]]:
            Sites for which the subsystem should be returned at the
            respective time,
    """
    if type(subsystems[0]) != list:
        if len(subsystems) != 2:
            raise ValueError("Subsystems must be of the form [a, b] or [[a1, "
                             "a2, ...], [b1, b2, ...]].")
        subsystems = [subsystems] * len_step_numbers
    return subsystems


def _times_to_steps(ts, num_trotter_slices):
    """
    Based on the requested times `ts`, calculate Trotter step numbers at which
    (subsystems of) evolved states need to be saved.

    Doctests:
    >>> _times_to_steps([10, 25, 30], 100)
    ([33, 83, 100], 0.3)
    >>> _times_to_steps([8, 26, 19], 80)
    ([25, 80, 58], 0.325)

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
    tau = max(ts) / num_trotter_slices
    step_numbers = [int(round(t / tau)) for t in ts]
    return step_numbers, tau


def _trotter_slice(hamiltonians, tau, num_sites, trotter_order, compr):
    """
    Get a list of ordered operator exponentials for one Trotter slice.

    The Trotter-Suzuki formula approximates the time-evolution during a single
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
    such that :math:`\\sum_p \\tau_p = \\tau`. The :math:`H_{j_p}` for a
    certain :math:`p` are all elements of the Hamiltonian acting either on even
    or on odd pairs of adjacent sites. This ensures that within one :math:`U_p`
    all terms in the exponential commute.

    This function returns the list of operators :math:`U_p` as MPOs.

    For more information on Trotter-Suzuki, see chapter 7 in Schollwöck's paper
    Annals of Physics 326, 96-192 (2011); doi: 10.1016/j.aop.2010.09.012.

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
    Get a list of ordered operator exponentials for one second-order Trotter
    slice.

    Based on the description in the documentation of _trotter_slice() and on
    the paper by Schollwöck, :math:`N` = 3, with :math:`\\tau_1 =\\tau_3 =
    \\tau/2` and :math:`\\tau_2=\\tau`.

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
    Get a list of ordered operator exponentials for one fourth-order Trotter
    slice.

    Based on the description in the documentation of _trotter_slice() and on
    the paper by Schollwöck, :math:`N` = 11, with

    .. math::
        \\tau_1 = \\tau_{11} = \\frac{\\tau}{2(4 - 4^{1/3})},

    .. math::
        \\tau_2 = \\tau_3 = \\tau_4 = \\tau_8 = \\tau_9 = \\tau_{10} =
        \\frac{\\tau}{4 - 4^{1 / 3}},

    .. math::
        \\tau_5 = \\tau_7 = \\frac{\\tau (1 - 3)}{2(4 - 4^{1 / 3})},

    and

    .. math::
        \\tau_6 = \\frac{\\tau (1 - 4)}{4 - 4^{1 / 3}}.

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
    Convert given list of Hamiltonians into form suitable for exponentiation.

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
    Calculates individual operator exponentials of adjacent odd-even sites,
    i.e. transforms :math:`\\{h_{j, j+1} : j \\text{ odd}\\}` into :math:`\\{
    \\text{e}^{\\mathrm{i} h_{j,j+1} \\tau} : j \\text{ odd}\\}`

    Doctest:
    >>> dims = [2, 2, 2, 2]
    >>> sx = np.array([[0, 1], [1, 0]])
    >>> sz = np.array([[1, 0], [0, -1]])
    >>> tau = 1
    >>> actual_result = _get_u_list_odd(dims, [sx] * 4, [np.kron(sz, sz)] * 3, \
        tau)
    >>> expected_result = [expm(-1j * tau * (np.kron(sz,sz) + \
        .5 * (np.kron(sx, np.identity(2)) + np.kron(np.identity(2), sx))))] * 2
    >>> print(np.array_equal(expected_result, actual_result))
    True

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
    Calculates individual operator exponentials of adjacent even-odd sites,
    i.e. transforms :math:`\\{h_{j,j+1} : j \\text{ even}\\}` into :math:`\\{
    \\text{e}^{\\mathrm{i} h_{j,j+1} \\tau} : j \\text{ even}\\}`

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
    Transforms list of matrices on odd-even sites to MPO acting on full
    state. So the list of u_odd :math:`\\{u_{j,j+1} : j \\text{ odd}\\}`,
    which are ``numpy.ndarrays``, is transformed into :math:`\\bigotimes_j
    u_{j,j+1} : j \\text{ odd}` of the type ``mpnum.MPArray``.

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
    Transforms list of matrices on even-odd sites to MPO acting on full
    state. So the list of u_even :math:`\\{u_{j,j+1} : j \\text{ even}\\}`,
    which are ``numpy.ndarrays``, is transformed into :math:`\\bigotimes_j
    u_{j,j+1} : j \\text{ even}` of the type ``mpnum.MPArray``.

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

    Converts given :math:`M \\times N` matrix in global form into an MPO with
    the given shape. The number of legs per site must be the same for all sites.

    Doctest:
    >>> matrix = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
    >>> mpo = matrix_to_mpo(matrix, [[3, 3]])
    >>> print(mpo.to_array_global())
    [[1 0 0]
     [0 0 0]
     [0 0 0]]

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
    Normalize a state.

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


def _set_compr_params():
    """
    A function to set default compression parameters if none were provided.
    They will suffice in many cases, but might well lead to
    problems described in the Pitfalls section of the introduction notebook.
    If that is the case, try to find and provide your own suitable
    compression parameters, which is also recommended to have more control
    over the calculations and their precision. For more information on this,
    read the introduction notebook and make use of the verbose output option to
    monitor bond dimensions during calculations.

    Returns:
        list[dict]:
            Some default compression and Trotter compression parameters
    """
    return dict(method='svd', relerr=1e-10), dict(method='svd', relerr=1e-4,
                                                  rank=30)


def evolve(state, hamiltonians, num_trotter_slices, method, trotter_order,
           ts, trotter_compr=None, compr=None, subsystems=None,
           v=0):
    """
    Evolve a one dimensional MPS, MPO or PMPS state using tMPS as described in
    the module's documentation.

    The initial state, Hamiltonians and certain parameters are required. The
    output is a list of times and a list of the evolved states at these times.
    Those states might be subsystems of the whole evolved system,
    which allows for the user to keep memory consumption small by
    focusing on the subsystems of interest.

    .. todo::
       Raise exception if hamiltonians are not of the right dimension

    .. todo::
       Implement tracking of compression errors.

    .. todo::
        Get variable compression to work (might involve changing mpnum).

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
        trotter_order (int):
            Order of Trotter-Suzuki decomposition to be used. Currently only 2
            and 4 are implemented
        ts (list[float]):
            The times for which the evolution should be computed and the state
            of the full system or a subsystem returned (i.e. it's reduced
            density matrix). The algorithm will calculate the
            evolution using the given number of Trotter steps for the largest
            number in ts. On the way there it will store the evolved states for
            smaller times. NB: Beware of memory overload since len(t) number of
            mpnum.MPArrays will be stored
        trotter_compr (dict):
            Compression parameters used in the iterations of Trotter (in the
            form required by mpnum.MPArray.compress(). If unsure, look at
            https://github.com/dseuss/mpnum/blob/master/examples/mpnum_intro.ipynb .)
            If omitted, some default compression will be used that will
            probably work but might lead to problems. See _set_compr_params()
            for more information.
        compr (dict):
            Parameters for the compression which is executed on every MPA during
            the calculations, except for the Trotter calculation, where
            trotter_compr is used. compr = dict(method='svd', rank=10) would for
            example ensure that the ranks of any MPA never exceed 10 during all
            of the calculations. An accepted relative error for the
            compression can be provided in addition to or instead of ranks,
            which would lead to e.g.
            compr = dict(method='svd', rank=10, relerr=1e-12).
            If omitted, some default compression will be used that will
            probably work but might lead to problems. See _set_compr_params()
            for more information.
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
            density matrix of the first three sites should be returned. A time
            can occur twice in ``ts`` and then different subsystems to be
            returned can be defined for that same time. If this parameter is
            omitted, the full system will be returned for every time in ``ts``.
        v (int):
            Level of verbose output. 0 means no output, 1 means that some
            basic output showing the progress of calculations is produced. 2
            will in addition show the bond dimensions of the state after every
            couple of iterations, 3 will show bond dimensions after every
            Trotter iteration.

    Returns:
        list[list[float], list[list[int]], list[mpnum.MPArray]]:
            A list with five items: (i) The list of times for which the density
            matrices have been computed (ii) The list indicating which
            subsystems of the system are returned at the respective time of the
            first list (iii) The list of density matrices as MPO or PMPS as
            mpnum.MPArray, depending on the input "method". If that was MPS, the
            full states will still be MPSs, the reduced ones will be MPOs.

    """
    if compr is None: compr, _ = _set_compr_params()
    if trotter_compr is None: _, trotter_compr = _set_compr_params()
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
    subsystems = _get_subsystems_list(subsystems, len(step_numbers))

    us = _trotter_slice(hamiltonians=hamiltonians, tau=tau,
                        num_sites=len(state), trotter_order=trotter_order,
                        compr=compr)
    if v != 0:
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
        v (int):
            Level of verbose output. 0 means no output, 1 means that some
            basic output showing the progress of calculations is produced. 2
            will in addition show the bond dimensions of the state after every
            couple of iterations, 3 will show bond dimensions after every
            Trotter iteration.

    Returns:
        list[list[float], list[list[int]], list[mpnum.MPArray]:
            A list with five items: (i) The list of times for which the density
            matrices have been computed (ii) The list indicating which
            subsystems of the system are returned at the respective time of the
            first list (iii) The list of density matrices as MPO or PMPS as
            mpnum.MPArray, depending on the input "method". If that was MPS, the
            full states will still be MPSs, the reduced ones will be MPOs.

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

    for i in range(max(step_numbers) + 1):
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
        if (v == 1 or v == 2) and np.sqrt(i + 1) % 1 == 0 and i < \
                step_numbers[-1]:
            print(str(i + 1) + " Trotter iterations finished...")
            if v == 2:
                print("Ranks: " + str(state.ranks))
        if v == 3 and i < step_numbers[-1]:
            print(str(i + 1) + " Trotter iterations finished...")
            print("Ranks: " + str(state.ranks))

    if v != 0:
        print("Done with time evolution")
    return times, subsystems, states  # , compr_errors, trot_errors


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
