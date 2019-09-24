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
    intsteps=1,
    nr=20,
    save_dir=None,
    radiation_mode="SimpleSED",
    n_cpus=1,
)


def test_initial_parameters():
    SCHW_RADIUS_2E8_CM = 29532500761002.49
    testing.assert_almost_equal(SCHW_RADIUS_2E8_CM, wind.RG)

    EDD_LUMIN = 2.51413032723893e+46
    testing.assert_almost_equal(EDD_LUMIN, wind.eddington_luminosity)

    BOL_LUMINOSITY = 0.5 * wind.eddington_luminosity
    testing.assert_equal(BOL_LUMINOSITY, wind.bol_luminosity)

    INNER_RADIUS = 200  # 2 * 8.901985418630156
    OUTER_RADIUS = 1600  # 1354.4151286369886
    testing.assert_almost_equal(INNER_RADIUS, wind.lines_r_min)
    testing.assert_almost_equal(OUTER_RADIUS, wind.lines_r_max)

    TAU_DR_0 = 2e8 * constants.SIGMA_T * wind.RG
    testing.assert_almost_equal(TAU_DR_0, wind.tau_dr_0)

    NUM_OF_STREAMLINES = 20
    dr = (OUTER_RADIUS - INNER_RADIUS) / (NUM_OF_STREAMLINES - 1)
    assert wind.lines_r_range[0] == (INNER_RADIUS + 0.5 * dr)
    assert wind.lines_r_range[-1] == (INNER_RADIUS + (20 - 0.5) * dr)


def test_norm2d():
    vector = np.array([1, 2, 3])
    testing.assert_equal(np.sqrt(10), wind.norm2d(vector))
    vector = np.array([-1, 0, 3])
    testing.assert_equal(np.sqrt(10), wind.norm2d(vector))
    vector = np.array([1, 2, -3])
    testing.assert_equal(np.sqrt(10), wind.norm2d(vector))
    vector = np.array([1, 3])
    testing.assert_equal(np.sqrt(10), wind.norm2d(vector))


def test_dist2d():
    vector1 = np.array([1, 2, 3])
    vector2 = np.array([-2, 0, -1])
    testing.assert_equal(5, wind.dist2d(vector1, vector2))

    vector1 = np.array([1, 2, 3])
    vector2 = np.array([-2, 0, -1])
    testing.assert_equal(5, wind.dist2d(vector2, vector1))

    vector1 = np.array([-1, 0, -3])
    vector2 = np.array([2, 5, 1])
    testing.assert_equal(5, wind.dist2d(vector1, vector2))


def test_v_kepler():
    r = 0.25
    testing.assert_equal(2, wind.v_kepler(r))


def test_v_esc():
    d = 0.5
    testing.assert_equal(2, wind.v_esc(d))


def test_thermal_velocity():
    T = 2e6
    V_AT_2E6 = 0.0004285850044051556  # in c units
    testing.assert_equal(V_AT_2E6, wind.thermal_velocity(2e6))


def test_tau_dr():
    den1 = 2e9
    den2 = 1e7
    tau1 = wind.tau_dr(den1)
    tau2 = wind.tau_dr(den2)
    ratio = tau1 / tau2
    testing.assert_almost_equal(ratio, 2e2)


def test_line():
    line = wind.line(
        r_0=420,
        z_0=10,
        rho_0=2e11,
        T=3e8,
        v_r_0=0.5,
        v_z_0=4e7,
        dt=2,
    )
    assert line.r_0 == 420
    assert line.z_0 == 10
    assert line.rho_0 == 2e11
    assert line.T == 3e8
    assert line.v_r == 0.5 / constants.C
    assert line.v_z_0 == 4e7 / constants.C
    assert line.dt == 2


def test_start_lines():
    v_z_0 = 1e5
    niter = 50
    lines = wind.start_lines(v_z_0, niter)
    assert len(lines) == 20
    assert lines[0].r_hist[0] == wind.lines_r_range[0]
    assert lines[-1].r_hist[0] == wind.lines_r_range[-1]
    V_Z_0_C = 1e5 / constants.C
    for line in lines:
        assert line.v_z_hist[0] == V_Z_0_C


def test_compute_wind_mass_loss():
    lines = wind.start_lines(rho=2e8, v_z_0=1e7, niter=0)
    line = wind.line(r_0=wind.lines_r_range[1], rho_0=2e8, v_z_0=1e7)
    line.iterate(niter=500000)
    assert line.escaped is True
    wind.lines[1] = line
    mdot_w = wind.compute_wind_mass_loss()
    r = wind.lines_r_range[1]
    dr = wind.lines_r_range[1] - wind.lines_r_range[0]
    mdot_exp = 2 * np.pi * ((r + dr/2.)**2 - (r - dr/2.)**2) * wind.RG**2
    mdot_exp = mdot_exp * lines[1].rho_0 * \
        constants.M_P * lines[1].v_T_0 * constants.C
    testing.assert_almost_equal(mdot_exp / mdot_w, 1.)
