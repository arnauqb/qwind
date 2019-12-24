"""
Tests the radiation class
"""

import numpy as np
import pytest
from numpy import testing
from qwind import constants, wind
from qwind.streamline import ida, euler

wind_instance = wind.Qwind()
# def test_initial_parameters():


def test_update_density():
    line = ida.streamline(
        wind_instance.radiation,
        wind_instance,
        r_0=500.,
        z_0=0.,
        rho_0=1e7,
        T=5e5,
        v_z_0=5e6,
        v_r_0=1.,
        dt=3.,
    )
    line.r = 4
    line.z = 3
    line.v_T = 1e6 / constants.C
    rho_expected = 5e11
    testing.assert_almost_equal(line.update_density(line.r, line.z, line.v_T), rho_expected)


def test_force_gravity():
    r = 3
    z = 4
    force_expected = - np.array([3, 4]) / 125.
    testing.assert_array_almost_equal(ida.force_gravity(r,z), force_expected)

# def test_update_positions():
#    line = streamline.streamline(
#        wind_instance.radiation,
#        wind_instance,
#        r_0 = 500.,
#        z_0 = 10.,
#        rho_0 = 1e7,
#        T = 5e5,
#        v_z_0 = 5e6,
#        v_r_0 = 1.,
#        dt = 3.,
#        )
#    fg = line.force_gravity()
#    fr = line.radiation.force_radiation(line.r, line.z, 0, 0)
#    a = fr + fg
#    a[0] += line.l**2 / line.r**3
