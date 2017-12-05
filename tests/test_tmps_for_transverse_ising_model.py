"""
Test to check the implemented functions in _tmps.py for the transverse Ising model

Author:
    Moritz Lange
"""

from scipy.linalg import expm
import numpy as np
import mpnum as mp
from tedopa import _tmps

# NOT WORKING YET

class TestTMPS(object):
    precision = 1e-5  # required precision of the tMPS results

    def test_mpo_approach(self):

        n = 4  # number of sites

        values_matrix = []  # to store the results of the conventional matrix approach
        values_mpo_trot1 = []  # to store the results of the calculation made using MPOs and Trotter of order 1
        values_mpo_trot2 = []  # to store the results of the calculation made using MPOs and Trotter of order 2

        state = self.state(n=n)
        J = 1
        B = 1
        hamiltonian = self.hamiltonian(n=n, J=J, B=B)
        kroneckerSum = self.kroneckerSum(n=n)

        # Convert the state density matrix into an MPO
        reshaped_state = state.reshape([2] * 2 * n)
        mpo_state = mp.MPArray.from_array_global(reshaped_state, ndims=2)

        num_trotter_steps = 100

        # Calculate expected total spin of the system for 3 time steps
        for t in np.linspace(0, 1, 2):
            # Calculate the expectation using the conventional way with full matrix first
            rho_t = self.exp(state=state, hamiltonian=hamiltonian, t=t)
            values_matrix = values_matrix + [np.trace(rho_t.dot(kroneckerSum))]

            # Then using MPOs and Trotter of order 1
            evolved_state = _tmps.evolve(mpo_state, hamiltonians=[B * self.sx(), J * np.kron(self.sz(), self.sz())],
                                         t=t, num_time_steps=num_trotter_steps, trotter_order=1, method='mpo')
            rho_t = evolved_state.to_array_global()
            rho_t = rho_t.reshape([2 ** n, 2 ** n])
            values_mpo_trot1 = values_mpo_trot1 + [np.trace(rho_t.dot(kroneckerSum))]

            # Then using MPOs and Trotter of order 2
            evolved_state = _tmps.evolve(mpo_state, hamiltonians=[B * self.sx(), J * np.kron(self.sz(), self.sz())],
                                         t=t, num_time_steps=num_trotter_steps, trotter_order=2, method='mpo')
            rho_t = evolved_state.to_array_global()
            rho_t = rho_t.reshape([2 ** n, 2 ** n])
            values_mpo_trot2 = values_mpo_trot2 + [np.trace(rho_t.dot(kroneckerSum))]

        # Now compare the results
        assert np.allclose(values_matrix, values_mpo_trot1, atol=self.precision)
        assert np.allclose(values_matrix, values_mpo_trot2, atol=self.precision)

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
        Generates a density matrix for a state in the transverse Ising model with n sites.

        Args:
            n (int): Number of sites

        Returns:
            numpy.ndarray: A density matrix for that state
        """
        state = np.zeros((2 ** n, 2 ** n))
        state[0, 0] = 1
        return state

    def kroneckerSum(self, n):
        """
        Generates the Kronecker sum of the sZ Pauli matrices for n sites

        Args:
            n (int): Number of sites

        Returns:
            numpy.ndarray: The Kronecker sum
        """

        pol_op = np.zeros((2 ** n, 2 ** n))
        for i in range(n):
            I1 = np.identity(2 ** i)
            I2 = np.identity(2 ** (n - i - 1))
            pol_op = pol_op + np.kron(np.kron(I1, self.sz()), I2)
        return pol_op

    def hamiltonian(self, n=5, J=1, B=1):
        """
        Generates the full Hamiltonian for the transverse Ising model, as defined in the respective Wikipedia article
        as of 28/11/2017

        Args:
            n (int): Number of sites
            J (int): Strength of interaction within every pair of two adjacent sites
            B (int): Strength of the magnetic field applied

        Returns:
            numpy.ndarray: The full Hamiltonian
        """

        if n < 2:
            n = 2

        hamiltonian = np.zeros((2 ** n, 2 ** n))

        for i in range(1, n):
            # calculate the outer products for the sites left of site i and i+1 in the sum of the Hamiltonian
            if i > 1:
                left = np.identity(2 ** (i - 1))
                part1 = np.kron(np.kron(left, self.sz()), self.sz())
                part2 = np.kron(left, self.sx())
            if i == 1:
                part1 = np.kron(self.sz(), self.sz())
                part2 = self.sx()
            # calculate the outer products for the sites right of site i and i+1 in the sum of the Hamiltonian
            if i < n - 1:
                right = np.identity(2 ** (n - 1 - i))
                part1 = np.kron(part1, right)
                part2 = np.kron(np.kron(part2, right), np.identity(2))
            if i == n - 1:
                part2 = np.kron(part2, np.identity(2))
            # add everything to the sum
            hamiltonian = hamiltonian + J * part1 + B * part2

        # finally add the Sx for the last site which was not taken care of in above loop
        hamiltonian = hamiltonian + B * np.kron(np.identity(2 ** (n - 1)), self.sx())

        return hamiltonian

    ############## Time evolution ################
    def exp(self, state, hamiltonian, t=1):
        """
        Evolve the state in time by using the classical approach of exponentiating the full Hamiltonian and applying
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
