"""
Tests for the functions in tedopa/tedopa_models.py
"""

import pytest as pt
import numpy as np
from numpy.testing import assert_almost_equal
import mpnum as mp

from tedopa import tedopa_models as tm


@pt.mark.parametrize('system_site_states, len_chain, dim_oscillators',
                     [([np.array([0, 1])], [3], [4]), (
                             [np.array([0, 1, 0]), np.array([0, 0, 1])], [2, 5],
                             [8, 3])])
def test_create_bosonic_vacuum_state(system_site_states, len_chain,
                                     dim_oscillators):
    state = tm.create_bosonic_vacuum_state(system_site_states, len_chain,
                                           dim_oscillators)
    assert len(state) == len(system_site_states) + sum(len_chain)


@pt.mark.parametrize('sites, dim', [(1, 3), (2, 2)])
def test_calculate_expectation_values(sites, dim):
    mpo = mp.random_mpa(sites=sites, ldim=(dim, dim), rank=1)
    obsvbl = np.random.random([dim ** sites] * 2)
    expct_value1 = np.trace(np.dot(mpo.to_array_global().reshape([dim ** sites]
                                                                 * 2), obsvbl))
    expct_value2 = tm.calculate_expectation_values([mpo], obsvbl)[0]
    assert_almost_equal(expct_value1, expct_value2)
