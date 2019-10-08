"""
Tests the radiation class
"""

import numpy as np
import pytest
from numpy import testing
from qwind import integration


def test_integrand_r():
    testing.assert_equal(
        integration._integrate_dblquad_kernel_r(1., 6., 10., 20.), 0.)
    testing.assert_equal(
        integration._integrate_dblquad_kernel_r(0., 10., 10., 20.), 0.)
     
