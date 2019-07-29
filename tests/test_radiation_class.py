"""
Tests the radiation class
"""

import numpy as np
import pytest
from numpy import testing
from qwind import wind

radiation = wind.Qwind().radiation



# fm tests #
def test_force_multiplier_k():
    """
    Tests k interpolation
    """
    xi_values = 10**np.array([ -4, -3, -2.26, -2.00, -1.50, -1.00,
              -0.42, 0.00, 0.22, 0.50, 1.0,
              1.5, 1.8, 2.0, 2.18, 2.39,
              2.76, 3.0, 3.29, 3.51, 3.68, 4.0 ])
    k_values = [ 0.411, 0.411, 0.400, 0.395, 0.363, 0.300,
               0.200, 0.132, 0.100, 0.068, 0.042,
               0.034, 0.033, 0.021, 0.013, 0.048,
               0.046, 0.042, 0.044, 0.045, 0.032,
               0.013]

    for xi, k_truth in zip(xi_values, k_values):
        k = radiation.force_multiplier_k(xi)
        testing.assert_almost_equal(k, k_truth)

def test_force_multiplier_etamax():
    """
    Tests eta interpolation
    """
    xi_values = 10**np.array([ -3, -2.5, -2.00, -1.50, -1.00,
             -0.5, -0.23, 0.0, 0.32, 0.50,
              1.0, 1.18,1.50, 1.68, 2.0,
              2.02, 2.16, 2.25, 2.39, 2.79,
              3.0, 3.32, 3.50, 3.75, 4.00 ])

    etamax_values = [ 6.95, 6.95, 6.98, 7.05, 7.26,
               7.56, 7.84, 8.00, 8.55, 8.95,
               8.47, 8.00, 6.84, 6.00, 4.32,
               4.00, 3.05, 2.74, 3.00, 3.10,
               2.73, 2.00, 1.58, 1.20, 0.78 ]   

    for xi, etamax_truth in zip(xi_values, etamax_values):
        etamax = radiation.force_multiplier_eta_max(xi)
        testing.assert_almost_equal(np.log10(etamax), etamax_truth)

def test_fm():
    """
    Tests force multiplier
    """
    xi_values = np.array([1e-4, 1e-4, 1e-4, 5e2]) 
    t_values = [1e-6, 1e-4, 1,  1e-4]
    fm_values = [1025, 1e2, 0.4,  1.2]
    
    for i in range(0,len(fm_values)):
        t = t_values[i]
        xi = xi_values[i]
        fm_truth = fm_values[i]
        fm = radiation.force_multiplier(t, xi)
        testing.assert_almost_equal(fm / fm_truth, 1., decimal = 1 )

    testing.assert_array_less(radiation.force_multiplier(1e-4, 3e4), 1e-4)
