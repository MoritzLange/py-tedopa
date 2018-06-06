"""
Tests for the functions in tedopa/tedopa.py
"""

from random import randint
import pytest as pt
import numpy as np

from tedopa import tedopa as td


@pt.mark.parametrize('dim', [2, 3, 6, 8])
def test_get_annihilation_op(dim):
    arr = td._get_annihilation_op(dim)
    i = randint(0, dim - 2)
    assert arr[i, i + 1] == np.sqrt(i + 1)
