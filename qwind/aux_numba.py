"""
Auxiliary file for Numba functions.
"""
import numpy as np
import scipy
import constants as const
import inspect
from numba import jitclass, jit, int32, float32, cfunc
from numba.types import intc, CPointer, float64
from scipy import LowLevelCallable
from scipy.integrate import quad

## aux variables for integration  ##
phids = np.linspace(0, np.pi, 100 + 1)
deltaphids = np.asarray([phids[i + 1] - phids[i] for i in range(0, len(phids) - 1)])
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

@jit(nopython=True)
def _Distance_gas_disc(r_d, phi_d, r, z):
    return np.sqrt(
        r ** 2. +
        r_d ** 2. +
        z ** 2. -
        2. *
        r *
        r_d *
        np.cos(phi_d))

@jit(nopython=True)
def _qwind_integral_kernel(r_d, phi_d, deltaphi, deltar, r, z, abs_uv):
    ff0 = (1. - np.sqrt(6. / r_d)) / r_d ** 3.
    delta = _Distance_gas_disc(r_d, phi_d, r, z)
    cos_gamma = (r - r_d * np.cos(phi_d)) / delta
    sin_gamma = z / delta
    darea = r_d * deltar * deltaphi
    ff = 2. * ff0 * darea * sin_gamma / delta ** 2. * abs_uv
    return [ff * cos_gamma, ff * sin_gamma]


@jit(nopython=True)
def qwind_integration(r, z, tau_uv):
    """
    Old integration method. Much faster, but poor convergence near the disc.
    """
    abs_uv = np.exp(-tau_uv)
    integral = [0., 0.]
    for i in range(0, len(deltards)):
        for j in range(0, len(deltaphids)):
            aux = _qwind_integral_kernel(
                rds[i], phids[j], deltaphids[j], deltards[i], r, z, abs_uv)
            integral[0] += aux[0]
            integral[1] += aux[1]
    return integral


@jit(nopython=True)
def _integrate_dblquad_kernel_r(r_d, phi_d, r, z, abs_uv):
    ff0 = (1. - np.sqrt(6. / r_d)) / r_d ** 3.
    delta = _Distance_gas_disc(r_d, phi_d, r, z)
    cos_gamma = (r - r_d * np.cos(phi_d)) / delta
    sin_gamma = z / delta
    darea = r_d 
    ff = 2. * ff0 * darea * sin_gamma / delta ** 2. * abs_uv
    return ff * cos_gamma 


@jit(nopython=True)
def _integrate_dblquad_kernel_z(r_d, phi_d, r, z, abs_uv):
    ff0 = (1. - np.sqrt(6. / r_d)) / r_d ** 3.
    delta = _Distance_gas_disc(r_d, phi_d, r, z)
    sin_gamma = z / delta
    darea = r_d 
    ff = 2. * ff0 * darea * sin_gamma / delta ** 2. * abs_uv
    return ff * sin_gamma


def qwind_integration_dblquad(r, z, tau_uv, Rmin, Rmax):
    abs_uv = np.exp(-tau_uv)
    r_int = scipy.integrate.dblquad(_integrate_dblquad_kernel_r, 0, np.pi, Rmin, Rmax, args=(r,z,abs_uv))[0]
    z_int = scipy.integrate.dblquad(_integrate_dblquad_kernel_z, 0, np.pi, Rmin, Rmax, args=(r,z,abs_uv))[0]
    return [r_int, z_int]

