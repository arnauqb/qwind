"""
Tests the radiation class
"""

import numpy as np
import pytest
from numpy import testing
from qwind import aux_numba

def test_qwind_integral_kernel():
    expected = np.array([2./20**2, 1./20**2])
    result = aux_numba._qwind_integral_kernel(1, 0, 3, 4)
    testing.assert_almost_equal(result, expected)

def test_integration_quad_r_phid():

    expected = 1/10**2.
    result = aux_numba.integration_quad_r_phid_test(0, 1, 2, 3)
    testing.assert_almost_equal(result, expected)

    expected = 10./134**2.
    result = aux_numba.integration_quad_r_phid_test(np.pi/2., 3, 10, 5)
    testing.assert_almost_equal(result, expected)

def test_integration_quad_z_phid():

    expected = 1/10**2.
    result = aux_numba.integration_quad_z_phid_test(0, 1, 2, 3)
    testing.assert_almost_equal(result, expected)

    expected = 1./134**2.
    result = aux_numba.integration_quad_z_phid_test(np.pi/2., 3, 10, 5)
    testing.assert_almost_equal(result, expected)

