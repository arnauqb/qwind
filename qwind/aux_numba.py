"""
Auxiliary file for Numba functions.
"""

import inspect

import numpy as np
import scipy
from numba import cfunc, float32, int32, jit, jitclass
from numba.types import CPointer, float64, intc
from scipy import LowLevelCallable
from scipy.integrate import quad

from qwind import compiled_functions
from qwind import constants as const

# aux variables for integration  ##
phids = np.linspace(0, np.pi, 100 + 1)
deltaphids = np.asarray([phids[i + 1] - phids[i]
                         for i in range(0, len(phids) - 1)])
rds = np.geomspace(6, 1400, 250 + 1)
deltards = np.asarray([rds[i + 1] - rds[i] for i in range(0, len(rds) - 1)])


def jit_integrand(integrand_function):

    jitted_function = jit(integrand_function, nopython=True, cache=True)
    no_args = len(inspect.getfullargspec(integrand_function).args)

    wrapped = None

    if no_args == 4:
        # noinspection PyUnusedLocal
        def wrapped(n, xx):
            return jitted_function(xx[0], xx[1], xx[2], xx[3])
    elif no_args == 5:
        # noinspection PyUnusedLocal
        def wrapped(n, xx):
            return jitted_function(xx[0], xx[1], xx[2], xx[3], xx[4])
    elif no_args == 6:
        # noinspection PyUnusedLocal
        def wrapped(n, xx):
            return jitted_function(xx[0], xx[1], xx[2], xx[3], xx[4], xx[5])
    elif no_args == 7:
        # noinspection PyUnusedLocal
        def wrapped(n, xx):
            return jitted_function(xx[0], xx[1], xx[2], xx[3], xx[4], xx[5], xx[6])
    elif no_args == 8:
        # noinspection PyUnusedLocal
        def wrapped(n, xx):
            return jitted_function(xx[0], xx[1], xx[2], xx[3], xx[4], xx[5], xx[6], xx[7])
    elif no_args == 9:
        # noinspection PyUnusedLocal
        def wrapped(n, xx):
            return jitted_function(xx[0], xx[1], xx[2], xx[3], xx[4], xx[5], xx[6], xx[7], xx[8])
    elif no_args == 10:
        # noinspection PyUnusedLocal
        def wrapped(n, xx):
            return jitted_function(xx[0], xx[1], xx[2], xx[3], xx[4], xx[5], xx[6], xx[7], xx[8], xx[9])
    elif no_args == 11:
        # noinspection PyUnusedLocal
        def wrapped(n, xx):
            return jitted_function(xx[0], xx[1], xx[2], xx[3], xx[4], xx[5], xx[6], xx[7], xx[8], xx[9], xx[10])

    cf = cfunc(float64(intc, CPointer(float64)))

    return LowLevelCallable(cf(wrapped).ctypes)

r_range_interp = 0
fraction_uv_list = 0
#@jit()
def uv_fraction_lookup(r):
    idx = (np.abs(r_range_interp - r)).argmin()
    return fraction_uv_list[idx]


@jit(nopython=True)
def _qwind_integral_kernel(r_d, phi_d, r, z):
    """
    Kernel of the radiation force integral.

    Args:
        r_d: disc element radial coordinate.
        phi_d: disc element angular coordinate.
        r: gas blob r coordinate.
        z: gas blob z coordinate.

    Returns:
        array[0]: kernel for the radial integration.
        array[1]: kernel for the z integration.
    """
    delta = r ** 2. + r_d ** 2. + z ** 2. - 2. * r * r_d * np.cos(phi_d)
    cos_gamma = (r - r_d * np.cos(phi_d))
    ff = 1. / delta ** 2.
    return [ff * cos_gamma, ff]


@jit(nopython=True)
def qwind_integration(r, z):
    """
    RE2010 integration method. Faster, but poor convergence near the disc, since it is a non adaptive method.

    Args:
        r: gas blob radial coordinate
        z: gas blob height coordinate

    Returns:
        array[0]: Result of the radial integration.
        array[1]: Result of the z integration.
    """
    integral = [0., 0.]
    for i in range(0, len(deltards)):
        int_step = [0., 0., ]
        deltar = deltards[i]
        r_d = rds[i]
        ff0 = (1. - np.sqrt(6. / r_d)) / r_d ** 2.
        for j in range(0, len(deltaphids)):
            phi_d = phids[j]
            deltaphi = deltaphids[j]
            aux = _qwind_integral_kernel(r_d, phi_d,  r, z)
            int_step[0] += aux[0] * deltaphi
            int_step[1] += aux[1] * deltaphi
        integral[0] += int_step[0] * deltar * ff0
        integral[1] += int_step[1] * deltar * ff0
    integral[0] = 2. * z * integral[0]
    integral[1] = 2. * z**2. * integral[1]
    return integral

@jit_integrand
def integration_quad_r_phid(phi_d, r_d, r, z):
    aux1 = (r - r_d * np.cos(phi_d))
    delta = r ** 2. + r_d ** 2. + z ** 2. - 2. * r * r_d * np.cos(phi_d)
    result = aux1 / delta**2.
    return result


def integration_quad_r_phid_test(phi_d, r_d, r, z):
    aux1 = (r - r_d * np.cos(phi_d))
    delta = r ** 2. + r_d ** 2. + z ** 2. - 2. * r * r_d * np.cos(phi_d)
    result = aux1 / delta**2.
    return result


@jit_integrand
def integration_quad_z_phid(phi_d, r_d, r, z):
    delta = r ** 2. + r_d ** 2. + z ** 2. - 2. * r * r_d * np.cos(phi_d)
    result = 1. / delta**2.
    return result


def integration_quad_z_phid_test(phi_d, r_d, r, z):
    delta = r ** 2. + r_d ** 2. + z ** 2. - 2. * r * r_d * np.cos(phi_d)
    result = 1. / delta**2.
    return result


def integration_quad_r_rd(r_d, r, z, uv_fraction_interpolator):
    phi_int = quad(integration_quad_r_phid, 0., np.pi, args=(r_d, r, z))[0]
    uv_fraction = uv_fraction_lookup(r)#uv_fraction_interpolator(r_d)
    ff0 = (1. - np.sqrt(6./r_d)) / r_d**2.
    result = ff0 * phi_int * uv_fraction
    return result


def integration_quad_z_rd(r_d, r, z, uv_fraction_interpolator):
    phi_int = quad(integration_quad_z_phid, 0., np.pi, args=(r_d, r, z))[0]
    uv_fraction = uv_fraction_lookup(r)#uv_fraction_interpolator(r_d)
    ff0 = (1. - np.sqrt(6./r_d)) / r_d**2.
    result = ff0 * phi_int * uv_fraction
    return result


def integration_quad(r, z, r_min, r_max, uv_fraction_interpolator):
    r_part = quad(integration_quad_r_rd, r_min, r_max, args=(
        r, z, uv_fraction_interpolator), points=[r])[0]
    z_part = quad(integration_quad_z_rd, r_min, r_max, args=(
        r, z, uv_fraction_interpolator), points=[r])[0]
    r_part = 2. * z * r_part
    z_part = 2. * z**2 * z_part
    return [r_part, z_part]


def integration_quad_r_rd_nointerp(r_d, r, z):
    phi_int = quad(integration_quad_r_phid, 0., np.pi, args=(r_d, r, z))[0]
    ff0 = (1. - np.sqrt(6./r_d)) / r_d**2.
    result = ff0 * phi_int
    return result


def integration_quad_z_rd_nointerp(r_d, r, z):
    phi_int = quad(integration_quad_z_phid, 0., np.pi, args=(r_d, r, z))[0]
    ff0 = (1. - np.sqrt(6./r_d)) / r_d**2.
    result = ff0 * phi_int
    return result


def integration_quad_nointerp(r, z, r_min, r_max):
    r_part = quad(integration_quad_r_rd_nointerp, r_min,
                  r_max, args=(r, z), points=[r])[0]
    z_part = quad(integration_quad_z_rd_nointerp, r_min,
                  r_max, args=(r, z), points=[r])[0]
    r_part = 2. * z * r_part
    z_part = 2. * z**2 * z_part
    return [r_part, z_part]


####

@jit_integrand
def _integrate_dblquad_kernel_r(r_d, phi_d, r, z):
    ff0 = (1. - np.sqrt(6. / r_d)) / r_d ** 2.
    delta = r ** 2. + r_d ** 2. + z ** 2. - 2. * r * r_d * np.cos(phi_d)
    cos_gamma = (r - r_d * np.cos(phi_d)) / delta**2.
    ff = ff0 * cos_gamma
    return ff


@jit_integrand
def _integrate_dblquad_kernel_z(r_d, phi_d, r, z):
    ff0 = (1. - np.sqrt(6. / r_d)) / r_d ** 2.
    delta = r ** 2. + r_d ** 2. + z ** 2. - 2. * r * r_d * np.cos(phi_d)
    ff = ff0 * 1. / delta**2.
    return ff


@jit()
def qwind_integration_dblquad(r, z, Rmin, Rmax):
    r_int, r_error = scipy.integrate.dblquad(
        _integrate_dblquad_kernel_r, 0 + 0.001, np.pi - 0.001, Rmin, Rmax, args=(r, z))
    z_int, z_error = scipy.integrate.dblquad(
        _integrate_dblquad_kernel_z, 0 + 0.001, np.pi - 0.001, Rmin, Rmax, args=(r, z))
    r_int = 2. * z * r_int
    z_int = 2. * z**2 * z_int
    return [r_int, z_int, r_error, z_error]
