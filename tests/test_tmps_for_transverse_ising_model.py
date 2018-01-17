"""
Test to check the implemented functions in _tmps.py for the
transverse Ising model
"""

from scipy.linalg import expm
from scipy.linalg import sqrtm
import numpy as np
import mpnum as mp
from tedopa import tmps


class TestTMPS(object):
    precision = 1e-7  # required precision of the tMPS results

    def test_mpo_trotter2(self):
        n = 4  # number of sites

        state = self.state(n=n)
        J = 1
        B = 1
        times = [1, 2]
        hamiltonian = self.hamiltonian(n=n, J=J, B=B)

        mpo_state = tmps.matrix_to_mpo(state, [[2, 2]] * n)

        num_trotter_slices = 100

        times, evolved_states, errors1, errors2 = \
            tmps.evolve(mpo_state, hamiltonians=[B * self.sx(),
                                                 J * np.kron(self.sz(),
                                                             self.sz())],
                        ts=times, num_trotter_slices=num_trotter_slices,
                        method='mpo', compr=dict(method='svd', relerr=1e-20),
                        trotter_order=2)

        rho_t_arr_1 = self.exp(state=state, hamiltonian=hamiltonian, t=times[0])
        rho_t_arr_2 = self.exp(state=state, hamiltonian=hamiltonian, t=times[1])

        rho_t_mpo_1 = evolved_states[0].to_array_global().reshape(
            [2 ** n, 2 ** n])
        rho_t_mpo_2 = evolved_states[1].to_array_global().reshape(
            [2 ** n, 2 ** n])

        fidelity_1 = np.trace(
            sqrtm(sqrtm(rho_t_arr_1).dot(rho_t_mpo_1).dot(sqrtm(rho_t_arr_1))))
        fidelity_2 = np.trace(
            sqrtm(sqrtm(rho_t_arr_2).dot(rho_t_mpo_2).dot(sqrtm(rho_t_arr_2))))

        assert np.isclose(1, fidelity_1, rtol=self.precision)
        assert np.isclose(1, fidelity_2, rtol=self.precision)

    def test_pmps_trotter2(self):
        n = 4  # number of sites

        state = self.state(n=n)
        J = 1
        B = 1
        times = [1, 2]
        hamiltonian = self.hamiltonian(n=n, J=J, B=B)

        mpo_state = tmps.matrix_to_mpo(state, [[2, 2]] * n)
        pmps_state = mp.mpo_to_pmps(mpo_state)

        num_trotter_slices = 100

        times, evolved_states, errors1, errors2 = \
            tmps.evolve(pmps_state, hamiltonians=[B * self.sx(),
                                                  J * np.kron(self.sz(),
                                                              self.sz())],
                        ts=times, num_trotter_slices=num_trotter_slices,
                        method='pmps', compr=dict(method='svd', relerr=1e-20),
                        trotter_order=2)

        rho_t_arr_1 = self.exp(state=state, hamiltonian=hamiltonian, t=times[0])
        rho_t_arr_2 = self.exp(state=state, hamiltonian=hamiltonian, t=times[1])

        rho_t_pmps_1 = mp.pmps_to_mpo(
            evolved_states[0]).to_array_global().reshape([2 ** n, 2 ** n])
        rho_t_pmps_2 = mp.pmps_to_mpo(
            evolved_states[1]).to_array_global().reshape([2 ** n, 2 ** n])

        fidelity_1 = np.trace(
            sqrtm(sqrtm(rho_t_arr_1).dot(rho_t_pmps_1).dot(sqrtm(rho_t_arr_1))))
        fidelity_2 = np.trace(
            sqrtm(sqrtm(rho_t_arr_2).dot(rho_t_pmps_2).dot(sqrtm(rho_t_arr_2))))

        assert np.isclose(1, fidelity_1, rtol=self.precision)
        assert np.isclose(1, fidelity_2, rtol=self.precision)

    def test_pmps_trotter4(self):
        n = 4  # number of sites

        state = self.state(n=n)
        J = 1
        B = 1
        times = [1, 2]
        hamiltonian = self.hamiltonian(n=n, J=J, B=B)

        mpo_state = tmps.matrix_to_mpo(state, [[2, 2]] * n)
        pmps_state = mp.mpo_to_pmps(mpo_state)

        num_trotter_slices = 100

        times, evolved_states, errors1, errors2 = \
            tmps.evolve(pmps_state, hamiltonians=[B * self.sx(),
                                                  J * np.kron(self.sz(),
                                                              self.sz())],
                        ts=times, num_trotter_slices=num_trotter_slices,
                        method='pmps', compr=dict(method='svd', relerr=1e-20),
                        trotter_order=2)

        rho_t_arr_1 = self.exp(state=state, hamiltonian=hamiltonian, t=times[0])
        rho_t_arr_2 = self.exp(state=state, hamiltonian=hamiltonian, t=times[1])

        rho_t_pmps_1 = mp.pmps_to_mpo(
            evolved_states[0]).to_array_global().reshape([2 ** n, 2 ** n])
        rho_t_pmps_2 = mp.pmps_to_mpo(
            evolved_states[1]).to_array_global().reshape([2 ** n, 2 ** n])

        fidelity_1 = np.trace(
            sqrtm(sqrtm(rho_t_arr_1).dot(rho_t_pmps_1).dot(sqrtm(rho_t_arr_1))))
        fidelity_2 = np.trace(
            sqrtm(sqrtm(rho_t_arr_2).dot(rho_t_pmps_2).dot(sqrtm(rho_t_arr_2))))

        assert np.isclose(1, fidelity_1, rtol=self.precision)
        assert np.isclose(1, fidelity_2, rtol=self.precision)

    ############# Pauli matrices ################
    def sx(self):
        return np.array([[0, 1], [1, 0]])

    def sy(self):
        return np.array([[0, -1j], [1j, 0]])

    def sz(self):
        return np.array([[1, 0], [0, -1]])

    ############ Other matrices #################
    def state(self, n):
        """
        Generates a density matrix for a state in the transverse Ising model
        with n sites.

        Args:
            n (int): Number of sites

        Returns:
            numpy.ndarray: A density matrix for that state
        """
        state = np.zeros((2 ** n, 2 ** n))
        state[0, 0] = 1
        return state

    def hamiltonian(self, n=5, J=1, B=1):
        """
        Generates the full Hamiltonian for the transverse Ising model,
        as defined in the respective Wikipedia article as of 28/11/2017

        Args:
            n (int): Number of sites
            J (int): Strength of interaction within every pair of
                two adjacent sites
            B (int): Strength of the magnetic field applied

        Returns:
            numpy.ndarray: The full Hamiltonian
        """

        if n < 2:
            n = 2

        hamiltonian = np.zeros((2 ** n, 2 ** n))

        for i in range(1, n):
            # calculate the outer products for the sites left of site i and i+1
            # in the sum of the Hamiltonian
            if i > 1:
                left = np.identity(2 ** (i - 1))
                part1 = np.kron(np.kron(left, self.sz()), self.sz())
                part2 = np.kron(left, self.sx())
            if i == 1:
                part1 = np.kron(self.sz(), self.sz())
                part2 = self.sx()
            # calculate the outer products for the sites right of site i and i+1
            # in the sum of the Hamiltonian
            if i < n - 1:
                right = np.identity(2 ** (n - 1 - i))
                part1 = np.kron(part1, right)
                part2 = np.kron(np.kron(part2, right), np.identity(2))
            if i == n - 1:
                part2 = np.kron(part2, np.identity(2))
            # add everything to the sum
            hamiltonian = hamiltonian + J * part1 + B * part2

        # finally add the Sx for the last site which was not
        # taken care of in above loop
        hamiltonian = hamiltonian + B * np.kron(np.identity(2 ** (n - 1)),
                                                self.sx())

        return hamiltonian

    ############## Time evolution ################
    def exp(self, state, hamiltonian, t=1):
        """
        Evolve the state in time by using the classical approach of
        exponentiating the full Hamiltonian and applying
        the result to the density matrix.

        Args:
            state (numpy.ndarray): The state to be evolved
            hamiltonian (numpy.ndarray): The Hamiltonian of the system
            t (float): The time for the evolution

        Returns:
            numpy.ndarray: The evolved state
        """

        U = expm(-1j * t * hamiltonian)
        U_dagger = expm(1j * t * hamiltonian)
        newState = U.dot(state).dot(U_dagger)
        return newState
