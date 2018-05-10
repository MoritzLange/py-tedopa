"""
Tests for the basic functions in py-tedopa.tmps.py
To check if the whole time evolution works (i.e. the more advanced functions
orchestrating the basic functions) see test_tmps_for_transverse_ising_model.py
"""

import pytest as pt
import numpy as np
from numpy.testing import assert_array_almost_equal

from tedopa import tmps


@pt.mark.parametrize('subsystems, len_step_numbers',
                     [([0, 1], 4), ([[0, 1], [2, 3], [4, 8]], 3)])
def test_get_subsystems_list(subsystems, len_step_numbers):
    subsystems = tmps._get_subsystems_list(subsystems, len_step_numbers)
    assert len(subsystems) == len_step_numbers


@pt.mark.parametrize('matrix, mpo_shape',
                     [(np.array([[0, 1], [1, 0]]), [[2, 2]]),
                      (np.array([[3, 5, 4], [6, 8, 3]]), [[1, 2], [3, 1]])])
def test_matrix_to_mpo(matrix, mpo_shape):
    assert_array_almost_equal(
        matrix, tmps.matrix_to_mpo(matrix, mpo_shape).to_array_global().reshape(
            matrix.shape))
