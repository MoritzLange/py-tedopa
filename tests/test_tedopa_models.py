"""
Tests for the functions in tedopa/tedopa_models.py
"""

import pytest as pt
import numpy as np

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
