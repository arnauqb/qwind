"""
Tests the radiation class
"""

import numpy as np
import pytest
from numpy import testing
from scipy import interpolate

from qwind.compiled_functions import interpolate_point_grid_1d
from qwind import constants, wind



def test_fractions():
    """
    Tests that setting a constant uv fraction for all radii recovers the
    averaged global uv fraction.
    """
    radiation = wind.Qwind(
        M=2e8,
        mdot=0.5,
        spin=0.,
        eta=0.06,
        r_in=200,
        r_out=1600,
        r_min=6.,
        r_max=1400.,
        T=2e6,
        mu=1,
        modes=[],
        rho_shielding=2e8,
        intsteps=1,
        nr=20,
        save_dir="results",
        radiation_mode="QSOSED",
        n_cpus=1,
    ).radiation

    rad_force_fractions = radiation.force_radiation(100, 100, 1, 1)
    radiation.uv_fraction_interpolator = lambda r: radiation.uv_fraction
    testing.assert_equal(radiation.uv_fraction_interpolator(
        100), radiation.uv_fraction)
    testing.assert_equal(radiation.uv_fraction_interpolator(
        10000), radiation.uv_fraction)
    rad_force_no_fractions = radiation.force_radiation(100, 100, 1, 1)
    testing.assert_array_almost_equal(
        rad_force_fractions, radiation.uv_fraction * rad_force_no_fractions)


def test_c_interpolator():
    """
    Tests implementation of the compiled uv fraction interpolator.
    """
    radiation = wind.Qwind(
        M=2e8,
        mdot=0.5,
        spin=0.,
        eta=0.06,
        r_in=200,
        r_out=1600,
        r_min=6.,
        r_max=1400.,
        T=2e6,
        mu=1,
        modes=[],
        rho_shielding=2e8,
        intsteps=1,
        nr=20,
        save_dir="results",
        radiation_mode="QSOSED",
        n_cpus=1,
    ).radiation

    rad_force_old_interpolator = radiation.force_radiation(100, 100, 1, 1)
    def c_interpolator(r): return interpolate_point_grid_1d(
        r, np.array(radiation.uv_fraction_list), radiation.r_range_interp, 0)
    radiation.uv_fraction_interpolator = c_interpolator
    rad_force_new_interpolator = radiation.force_radiation(100, 100, 1, 1)
    testing.assert_array_almost_equal(
        rad_force_new_interpolator, rad_force_old_interpolator)
