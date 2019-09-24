"""
Tests the radiation class
"""

import numpy as np
import pytest
from numpy import testing
from qwind import wind, constants

radiation = wind.Qwind(
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
).radiation


def test_initial_parameters():
    testing.assert_equal(radiation.wind.lines_r_min, 200)
    testing.assert_equal(radiation.wind.lines_r_max, 1600)
    testing.assert_approx_equal(radiation.r_x, 269.6186880256537)
    testing.assert_equal(radiation.uv_fraction, 0.7104040416295189)
    testing.assert_equal(radiation.xray_fraction, 0.13260939889639622)
    testing.assert_approx_equal(
        radiation.FORCE_RADIATION_CONSTANT, 0.7066519676112408, significant=6)
    testing.assert_approx_equal(
        radiation.wind.eddington_luminosity, 2.51413032723893e46, significant=6)
    testing.assert_approx_equal(radiation.xray_luminosity,
                                0.5 * radiation.xray_fraction *
                                radiation.wind.eddington_luminosity,
                                significant=6)


def test_optical_depth_uv():
    """
    test uv optical depth.
    """
    tau_dr = 2e-3
    tau_dr_0 = 5e-3
    r_init = radiation.r_init
    tau_uv_check = tau_dr * (400 - 250) + tau_dr_0 * (250 - r_init)
    testing.assert_approx_equal(radiation.optical_depth_uv(
        400, 0, 250, tau_dr, tau_dr_0), tau_uv_check)
    testing.assert_equal(radiation.optical_depth_uv(400, 1, 250, 0, 0), 0.)


def test_ionization_parameter():

    r = 100
    z = 100
    tau_x = 0
    xi1 = radiation.xray_luminosity / 2e8 / \
        (r**2 + z**2) / radiation.wind.RG**2.
    testing.assert_approx_equal(xi1, radiation.ionization_parameter(
        r, z, tau_x, radiation.wind.rho_shielding))


def test_critical_ionization_parameter():
    """
    Tests if we got the right ionization radius r_x.
    """
    testing.assert_equal(1e5, constants.IONIZATION_PARAMETER_CRITICAL)
    testing.assert_array_less(
        abs(radiation.ionization_radius_kernel(radiation.r_x)), 1e-8)


def test_opacity_x_r():
    testing.assert_equal(radiation.opacity_x_r(radiation.r_x + 10), 100)
    testing.assert_equal(radiation.opacity_x_r(radiation.r_x - 10), 1)


def test_optical_depth_x():
    tau_dr = 50e-4
    tau_dr_0 = 100e-4
    tau_x_1 = radiation.optical_depth_x(
        radiation.r_x, 0, radiation.r_x, tau_dr, tau_dr_0, 2e8)
    tau_truth = tau_dr_0 * (radiation.r_x - radiation.wind.r_init)
    testing.assert_approx_equal(tau_truth, tau_x_1)
    tau_x_2 = radiation.optical_depth_x(200, 10, 100, 0, 0, 2e8)
    testing.assert_approx_equal(tau_x_2, 0)


def test_sobolev_optical_depth():
    tau_dr = 20
    v_th = radiation.wind.v_thermal
    testing.assert_equal(radiation.sobolev_optical_depth(1, 1), v_th)
    testing.assert_equal(radiation.sobolev_optical_depth(0, 100), 0)
    testing.assert_equal(radiation.sobolev_optical_depth(tau_dr, 20), v_th)


def test_force_radiation():
    r_values = np.linspace(200, 1000)
    z_values = np.linspace(200, 1000)
    for r in r_values:
        for z in z_values:
            force = radiation.force_radiation(r, z, 1, 1, 1)
            assert force[0] >= 0, "Negative radial force!"
            assert force[-1] >= 0, "Negative z force!"

# TODO
## fm tests #
# def test_force_multiplier_k():
#    """>
#    Tests k interpolation
#    """
#    xi_values = 10**np.array([ -4, -3, -2.26, -2.00, -1.50, -1.00,
#              -0.42, 0.00, 0.22, 0.50, 1.0,
#              1.5, 1.8, 2.0, 2.18, 2.39,
#              2.76, 3.0, 3.29, 3.51, 3.68, 4.0 ])
#    k_values = [ 0.411, 0.411, 0.400, 0.395, 0.363, 0.300,
#               0.200, 0.132, 0.100, 0.068, 0.042,
#               0.034, 0.033, 0.021, 0.013, 0.048,
#               0.046, 0.042, 0.044, 0.045, 0.032,
#               0.013]
#
#    for xi, k_truth in zip(xi_values, k_values):
#        k = radiation.force_multiplier_k(xi)
#        testing.assert_approx_equal(k, k_truth)
#
# def test_force_multiplier_etamax():>
#    """
#    Tests eta interpolation
#    """
#    xi_values = 10**np.array([ -3, -2.5, -2.00, -1.50, -1.00,
#             -0.5, -0.23, 0.0, 0.32, 0.50,
#              1.0, 1.18,1.50, 1.68, 2.0,
#              2.02, 2.16, 2.25, 2.39, 2.79,
#              3.0, 3.32, 3.50, 3.75, 4.00 ])
#
#    etamax_values = [ 6.95, 6.95, 6.98, 7.05, 7.26,
#               7.56, 7.84, 8.00, 8.55, 8.95,
#               8.47, 8.00, 6.84, 6.00, 4.32,
#               4.00, 3.05, 2.74, 3.00, 3.10,
#               2.73, 2.00, 1.58, 1.20, 0.78 ]
#
#    for xi, etamax_truth in zip(xi_values, etamax_values):
#        etamax = radiation.force_multiplier_eta_max(xi)
#        testing.assert_approx_equal(np.log10(etamax), etamax_truth)

# def test_fm():
#    """
#    Tests force multiplier
#    """
#    xi_values = np.array([1e-4, 1e-4, 1e-4, 5e2])
#    t_values = [1e-6, 1e-4, 1,  1e-4]
#    fm_values = [1025, 1e2, 0.4,  1.2]
#
#    for i in range(0,len(fm_values)):
#        t = t_values[i]
#        xi = xi_values[i]
#        fm_truth = fm_values[i]
#        fm = radiation.force_multiplier(t, xi)
#        testing.assert_approx_equal(fm / fm_truth, 1., significant = 1 )
#
#    testing.assert_array_less(radiation.force_multiplier(1e-4, 3e4), 1e-4)
