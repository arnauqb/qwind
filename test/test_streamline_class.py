"""
Tests the radiation class
"""

import numpy as np
import pytest
from numpy import testing
from qwind import constants, streamline, wind

wind_instance = wind.Qwind()
# def test_initial_parameters():


def test_update_density():
    line = streamline.streamline(
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
    line.d = 1000.
    line.v_T = 1e6 / constants.C
    rho_expected = 0.25 * 5e7
    testing.assert_almost_equal(line.update_density(), rho_expected)


def test_force_gravity():
    line = streamline.streamline(
        wind_instance.radiation,
        wind_instance,
        r_0=500.,
        z_0=10.,
        rho_0=1e7,
        T=5e5,
        v_z_0=5e6,
        v_r_0=1.,
        dt=3.,
    )
    line.r = 3
    line.z = 4
    line.d = 5
    force_expected = - np.array([3, 0, 4]) / 125.
    testing.assert_array_almost_equal(line.force_gravity(), force_expected)

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
