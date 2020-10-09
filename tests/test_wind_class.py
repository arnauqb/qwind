"""
Tests the wind class
"""

import numpy as np
import pytest
from numpy import testing
from qwind import wind, constants

wind = wind.Qwind(
    M=2e8,
    mdot=0.5,
    spin=0.,
    eta=0.06,
    lines_r_min=200,
    lines_r_max=1600,
    disk_r_min=6.,
    disk_r_max=1400.,
    T=2e6,
    mu=1,
    modes=[],
    rho_shielding=2e8,
    nr=20,
    save_dir=None,
)


def test_initial_parameters():
    SCHW_RADIUS_2E8_CM =  29532500761002.496
    testing.assert_almost_equal(wind.R_g, SCHW_RADIUS_2E8_CM)

    EDD_LUMIN = 2.5141303596935816e+46
    testing.assert_almost_equal(wind.eddington_luminosity, EDD_LUMIN)

    INNER_RADIUS = 200  # 2 * 8.901985418630156
    OUTER_RADIUS = 1600  # 1354.4151286369886
    testing.assert_almost_equal(INNER_RADIUS, wind.lines_r_min)
    testing.assert_almost_equal(OUTER_RADIUS, wind.lines_r_max)

    TAU_DR_0 = 2e8 * constants.SIGMA_T * wind.R_g
    testing.assert_almost_equal(TAU_DR_0, wind.tau_dr_0)

    NUM_OF_STREAMLINES = 20
    dr = (OUTER_RADIUS - INNER_RADIUS) / (NUM_OF_STREAMLINES)
    assert wind.lines_r_range[0] == (INNER_RADIUS + 0.5 * dr)
    assert wind.lines_r_range[-1] == (INNER_RADIUS + (21 - 0.5) * dr)


def test_v_kepler():
    r = 0.25
    testing.assert_equal(2, wind.v_kepler(r))


def test_v_esc():
    d = 0.5
    testing.assert_equal(2, wind.v_esc(d))


def test_thermal_velocity():
    T = 2e6
    V_AT_2E6 = 0.0004285850044051556  # in c units  
    testing.assert_almost_equal(wind.thermal_velocity(2e6), V_AT_2E6)


def test_tau_dr():
    den1 = 2e9
    den2 = 1e7
    tau1 = wind.tau_dr(den1)
    tau2 = wind.tau_dr(den2)
    ratio = tau1 / tau2
    testing.assert_almost_equal(ratio, 2e2)


