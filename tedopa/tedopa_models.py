"""
This module contains functions which enable the user to run TEDOPA for
certain settings without having to know ``mpnum``. This is a wrapper for the
tedopa/tedopa.py module.

TEDOPA proceeds in two steps: (i) Map the Hamiltonian from a system,
composed of one or two sites that are each linearly coupled to a
distinct reservoir of bosonic modes with a given spectral function, to the same
site/sites coupled to a 1D chain of oscillators (one chain per system site) and
(ii) perform time evolution.

The performed mapping is based on an algorithm introduced by Chin et al. in
Journal of Mathematical Physics 51, 092109 (2010); doi: 10.1063/1.3490188.

If something is unclear in the functions here, please look at
tedopa/tedopa.py or of course the example notebooks for deeper understanding.

.. todo::
   Might be good to have a thermal state of a given Hamiltonian. => make a
   function ``create_bosonic_thermal_state()``

"""

from tedopa import tedopa as td
from tedopa import tmps
import numpy as np
import mpnum as mp


def create_bosonic_vacuum_state(system_site_states, len_chain, dim_oscillators):
    """
    This function will create the state of a system, comprised of one or two
    sites, each connected to a (possibly distinct) environment of bosonic
    modes in vacuum state. It is assumed that the state is not entangled and
    the single sites of the system are each in a pure state. It will be
    represented as an MPS rather than an MPO.

    Args:
        system_site_states (list[numpy.ndarray]): List of the vectors
            describing the state of the system site or sites,
            i.e. [:math:`|\\psi_1 \\rangle`]
            or [:math:`|\\psi_1 \\rangle`, :math:`|\\psi_2 \\rangle`].
        len_chain (list[int]): List specifying the length of the chains
            representing the environment of the respective system site.
        dim_oscillators (list[int]): List specifying the dimensions of the
            oscillators in the respective chain.

    Returns:
        mpnum.MPArray: The vacuum state in MPS form.
    """
    if not (len(system_site_states) == len(len_chain) == len(dim_oscillators)):
        raise ValueError("The system must either have one site coupled to one"
                         " chain or two sites coupled to two chains.")
    chains = [[np.array([1] + [0] * (oscillator_dimension - 1))] * chain_length
              for oscillator_dimension, chain_length in
              zip(dim_oscillators, len_chain)]
    if len(system_site_states) == 2:
        return mp.MPArray.from_kron(chains[0] + system_site_states + chains[1])
    else:
        return mp.MPArray.from_kron(system_site_states + chains[0])


def tedopa1_for_bosonic_vacuum_state(system_site_state, len_chain,
                                     dim_oscillators, h_loc, a, j, domain, ts,
                                     observable, g=1, trotter_order=2,
                                     num_trotter_slices=100, v=1):
    """
    This function will take information about a system of only one site,
    the coupling to its environment and some simulation parameters, and
    performs TEDOPA on it. The initial state of the environment will be set
    to a bosonic vacuum state. The function also takes an observable and will
    return the expectation values for that observable with respect to the
    reduced density matrices of the evolved one-site system.

    Args:
        system_site_state (numpy.ndarray): The vector :math:`|\\psi \\rangle`
            describing the state of the one-site system.
        len_chain (int): The length of the chain representing the environment of
            the system site.
        dim_oscillators (int): The dimension of the oscillators in the
            chain. When choosing e.g. 6, each of the ``len_chain`` number of
            oscillators in the chain will have a 6x6 density matrix
        h_loc (numpy.ndarray):
            Matrix representation of the local Hamiltonian of the one-site
            system.
        a (numpy.ndarray):
            Interaction operator. This is the site-part of the tensor product
            that comprises the interaction Hamiltonian and is defined as
            :math:`\\hat{A}` in Chin et al.
        j (types.LambdaType):
            Spectral function :math:`J(\\omega)` as defined in Chin et al.
            doi: 10.1063 / 1.3490188
        domain (list[float]):
            Domain on which :math:`J(\\omega)` is defined, for example [0,
            np.inf]
        ts (list[float]):
            The times for which the evolved states, and based on that the
            expectation values of the observable, should be computed.
        observable (numpy.ndarray): The observable used to determine the
            expectation values at the times in ``ts`` for the reduced density
            matrices of the evolved one-site system. It must be of the same
            dimension as the density matrix of the system in question.
        g (float):
            Cutoff :math:`g`, assuming that for :math:`J(\\omega)` it is
            :math:`g(\\omega)=g\\omega`.
        trotter_order (int):
            Order of Trotter - Suzuki decomposition to be used. Currently only 2
            and 4 are implemented
        num_trotter_slices (int):
            Number of Trotter slices to be used for the largest t in ts. If
            ts=[10, 25, 30] and num_trotter_slices=100, then the program
            would use 100/30*10=33, 100/30*25=83 and 100/30*30=100 Trotter
            slices to calculate the time evolution for the three times.
        v (int):
            Level of verbose output. 0 means no output, 1 will show the
            progress of the calculations. 2 and 3 will also indicate bond
            dimensions in the evolved state. For more information see
            :func:`tedopa.tedopa1`

    Returns:
        tuple[list[float], list[mpnum.MPArray.dtype]]:
            A list of times corresponding to the respective expectation
            values and a list of the expectation values of the provided
            observable for evolved states
    """
    initial_state = create_bosonic_vacuum_state([system_site_state],
                                                [len_chain], [dim_oscillators])
    times, evolved_states = td.tedopa1(h_loc=h_loc, a=a, state=initial_state,
                                       method='mps', j=j,
                                       domain=domain,
                                       ts_full=[], ts_system=ts, g=g,
                                       trotter_order=trotter_order,
                                       num_trotter_slices=num_trotter_slices,
                                       ncap=40000, v=v)
    expectation_values = calculate_expectation_values(evolved_states,
                                                      observable)
    return times, expectation_values


def tedopa2_for_bosonic_vacuum_state(system_site_state, len_chain,
                                     dim_oscillators, h_loc, a_twosite, js,
                                     domains, ts,
                                     observable, gs=1, trotter_order=2,
                                     num_trotter_slices=100, v=0):
    """
    This function will take information about a system comprised of two not
    entangled sites, the coupling to their two environments and some
    simulation parameters, and performs TEDOPA on it. The initial state of
    both environments will be set to a bosonic vacuum state. The function
    also takes an observable and will return the expectation values for that
    observable with respect to the reduced density matrices of the evolved
    two-site system.

    Args:
        system_site_state (list[numpy.ndarray]): List of the vectors describing
            the states of the system sites, i.e. [:math:`|\\psi_1
            \\rangle`, :math:`|\\psi_2 \\rangle`].
        len_chain (list[int]): The lengths of the two chains representing the
            environments of the two system sites.
        dim_oscillators (list[int]): The dimensions of the oscillators in the two
            chains. When choosing e.g. [6, 8], each of the ``len_chain[0]``
            number of oscillators in the first chain will have a 6x6 density
            matrix, and each of the oscillators in the second chain 8x8
            density matrices.
        h_loc (numpy.ndarray):
            Matrix representation of the local Hamiltonian of the two-site
            system.
        a_twosite (list[numpy.ndarray]):
            List of two matrices, each of which represents the site-part of the
            tensor product interaction Hamiltonian for the two sites. See
            :math:`\\hat{A}` in Chin et al. doi: 10.1063 / 1.3490188
        js (list[types.LambdaType]):
            Spectral functions :math:`J(\\omega)` for the two environments as
            defined by Chin et al.
        domains (list[list[float]]):
            Domains on which the :math:`J(\\omega)` are defined. Can be
            different for the two sites, for example, [[0, np.inf], [0,1]]
        ts (list[float]):
            The times for which the evolved states, and based on that the
            expectation values of the observable, should be computed.
        observable (numpy.ndarray): The observable used to determine the
            expectation values at the times in ``ts`` for the reduced density
            matrices of the evolved two-site system. It must be of the same
            dimension as the density matrix of the system in question.
        gs (list[float]):
            List of cutoffs :math:`g`, assuming that for :math:`J(\\omega)`
            it is :math:`g(\\omega)=g\\omega`.
        trotter_order (int):
            Order of Trotter - Suzuki decomposition to be used. Currently only 2
            and 4 are implemented
        num_trotter_slices (int):
            Number of Trotter slices to be used for the largest t in ts. If
            ts=[10, 25, 30] and num_trotter_slices=100, then the program
            would use 100/30*10=33, 100/30*25=83 and 100/30*30=100 Trotter
            slices to calculate the time evolution for the three times.
        v (int):
            Level of verbose output. 0 means no output, 1 will show the
            progress of the calculations. 2 and 3 will also indicate bond
            dimensions in the evolved state. For more information see
            :func:`tedopa.tedopa2`

    Returns:
        tuple[list[float], list[mpnum.MPArray.dtype]]:
            A list of times corresponding to the respective expectation
            values and a list of the expectation values of the provided
            observable for evolved states
    """
    initial_state = create_bosonic_vacuum_state(system_site_state,
                                                len_chain, dim_oscillators)
    times, evolved_states = td.tedopa2(h_loc=h_loc, a_twosite=a_twosite,
                                       state=initial_state,
                                       method='mps', sys_position=len_chain[0],
                                       js=js, domains=domains,
                                       ts_full=[], ts_system=ts, gs=gs,
                                       trotter_order=trotter_order,
                                       num_trotter_slices=num_trotter_slices,
                                       ncap=40000, v=v)
    expectation_values = calculate_expectation_values(evolved_states,
                                                      observable)
    return times, expectation_values


def calculate_expectation_values(states, observable):
    """
    Calculates the expectation values :math:`\\langle M \\rangle_i` of the the
    ``observable`` :math:`M` with respect to the ``states``
    :math:`\\{\\rho\\}`

    .. math::
        \\langle M \\rangle_i = \\text{tr}(\\rho_i M)

    For this function to work, the observable has to have the same dimension
    as the density matrix which represents the state, i.e. all states must
    have the same dimensions. This function is meant to work with a list of
    evolved states of the same system at different times.

    Args:
        states (list[mpnum.MPArray]): List of states :math:`\\{\\rho\\}`.
            They are assumed to be MPOs and already normalized
        observable (numpy.ndarray): The matrix representing the observable
            :math:`M` in global form (as opposed to local form. Global form
            is just the usual form to write matrices in QM. For more
            information, see the ``mpnum`` documentation.)

    Returns:
        list[mpnum.MPArray.dtype]: List of expectation values for the states
    """
    if not all(state.shape == states[0].shape for state in states):
        raise ValueError("The states in the provided list are not all of the "
                         "same shape.")
    if len(observable) != np.prod(np.array([_[0] for _ in states[0].shape])):
        raise ValueError("Observable dimensions and state dimensions do not "
                         "fit")
    expct_values = [mp.trace(mp.dot(state,
                                    tmps.matrix_to_mpo(observable,
                                                       [[_[0]] * 2 for _ in
                                                        state.shape])))
                    for state in states]
    return expct_values
