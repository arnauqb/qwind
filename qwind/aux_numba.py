"""
Auxiliary file for Numba functions.
"""

import numpy as np
import scipy
from qwind import constants as const
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
    """
    Distance between a disc element at (r_d, phi_d), and at a gas blob at (r,0,z).
    Args:
        r_d: disc element radial coordinate.
        phi_d: disc element angular coordinate.
        r: gas blob radial coordinate.
        z: gas blob height coordinate.
    Returns:
        Distance between disc element and gas blob.
    """
    return np.sqrt(
        r ** 2. +
        r_d ** 2. +
        z ** 2. -
        2. *
        r *
        r_d *
        np.cos(phi_d))

@jit(nopython=True)
def _qwind_integral_kernel(r_d, phi_d, r, z):
    """
    Kernel of the radiation force integral.

    Args:
        r_d: disc element radial coordinate.
        phi_d: disc element angular coordinate.
        deltaphi: spacing of the interval of the angular integration.
        deltar: spacing of the interval of the radial integration.
        r: gas blob r coordinate.
        z: gas blob z coordinate.

    Returns:
        array[0]: kernel for the radial integration.
        array[1]: kernel for the z integration.
    """
    delta = r ** 2. + r_d ** 2. + z ** 2. - 2. * r * r_d * np.cos(phi_d)
    cos_gamma = (r - r_d * np.cos(phi_d)) 
    ff = 1. / delta ** 2.
    return [ff * cos_gamma, ff ]


@jit(nopython=True)
def qwind_integration(r, z):
    """
    RE2010 integration method. Much faster, but poor convergence near the disc, since it is a non adaptive method.

    Args:
        r: gas blob radial coordinate
        z: gas blob height coordinate
    """
    integral = [0., 0.]
    for i in range(0, len(deltards)):
        int_step = [0., 0.,]
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
def _integrate_dblquad_kernel_r(r_d, phi_d, r, z):
    ff0 = (1. - np.sqrt(6. / r_d)) / r_d ** 3.
    #delta = _Distance_gas_disc(r_d, phi_d, r, z)
    delta = np.sqrt(r ** 2. + r_d ** 2. + z ** 2. - 2. * r * r_d * np.cos(phi_d))
    cos_gamma = (r - r_d * np.cos(phi_d)) / delta
    sin_gamma = z / delta
    darea = r_d 
    ff = 2. * ff0 * darea * sin_gamma / delta ** 2. 
    return ff * cos_gamma 


@jit_integrand
def _integrate_dblquad_kernel_z(r_d, phi_d, r, z):
    ff0 = (1. - np.sqrt(6. / r_d)) / r_d ** 3.
    delta = np.sqrt(r ** 2. + r_d ** 2. + z ** 2. - 2. * r * r_d * np.cos(phi_d))
    sin_gamma = z / delta
    darea = r_d 
    ff = 2. * ff0 * darea * sin_gamma / delta ** 2. 
    return ff * sin_gamma


def old_qwind_integration_dblquad(r, z, Rmin, Rmax):
    r_int = abs_uv * scipy.integrate.dblquad(_integrate_dblquad_kernel_r, 0, np.pi, Rmin, Rmax, args=(r,z))[0]
    z_int = abs_uv * scipy.integrate.dblquad(_integrate_dblquad_kernel_z, 0, np.pi, Rmin, Rmax, args=(r,z))[0]
    return [r_int, z_int]


def qwind_integration_dblquad(r, z, Rmin, Rmax):

    integral_r = 2 * z**2. * scipy.integrate.quad(integration_r_kernel_rd, Rmin, Rmax, args = (r,z))[0]
    integral_z = 2 * z * scipy.integrate.quad(integration_z_kernel_rd, Rmin, Rmax, args = (r,z))[0]
    return [integral_r, integral_z]

@jit_integrand
def integration_r_kernel_phid(phi_d, r_d, r, z):
    aux = (r - r_d * np.cos(phi_d))
    delta = np.sqrt(r ** 2. + r_d ** 2. + z ** 2. - 2. * r * r_d * np.cos(phi_d))
    return aux / delta**4.

@jit()
def integration_r_kernel_rd(r_d, r, z):

    aux = (1. - np.sqrt(6 / r_d)) / r_d**2.
    int_phi = scipy.integrate.quad(integration_r_kernel_phid, 0, np.pi, args = (r_d, r, z))[0]
    return aux * int_phi

@jit_integrand
def integration_z_kernel_phid(phi_d, r_d, r, z):
    delta = np.sqrt(r ** 2. + r_d ** 2. + z ** 2. - 2. * r * r_d * np.cos(phi_d))
    return 1. / delta**4.

@jit()
def integration_z_kernel_rd(r_d, r, z):
    aux = (1. - np.sqrt(6 / r_d)) / r_d**2.
    int_phi = scipy.integrate.quad(integration_z_kernel_phid, 0, np.pi, args = (r_d, r, z))[0]
    return aux * int_phi

