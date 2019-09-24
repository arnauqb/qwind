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

from qwind import constants as const


def jit_integrand(integrand_function):
    """
    Turns a function into a LowLevelCallable function.
    """

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


def _integrate_dblquad_kernel_r(phi_d, r_d, r, z):
    """
    Radial part the radiation force integral.

    Args:
        phi_d: disc angle integration variable in radians.
        r_d: disc radius integration variable in Rg.
        r: disc radius position in Rg
        z: disc height position in Rg

    Returns:
        Radial integral kernel.

    """
    ff0 = (1. - np.sqrt(6. / r_d)) / r_d ** 2.
    delta = r**2. + r_d**2. + z**2. - 2.*r*r_d * np.cos(phi_d)
    cos_gamma = (r - r_d*np.cos(phi_d)) / delta**2.
    ff = ff0 * cos_gamma
    return ff

@jit_integrand
def _integrate_dblquad_kernel_r_jitted(phi_d, r_d, r, z):
    """
    Radial part the radiation force integral.

    Args:
        phi_d: disc angle integration variable in radians.
        r_d: disc radius integration variable in Rg.
        r: disc radius position in Rg
        z: disc height position in Rg

    Returns:
        Radial integral kernel.

    """
    ff0 = (1. - np.sqrt(6. / r_d)) / r_d ** 2.
    delta = r**2. + r_d**2. + z**2. - 2.*r*r_d * np.cos(phi_d)
    cos_gamma = (r - r_d*np.cos(phi_d)) / delta**2.
    ff = ff0 * cos_gamma
    return ff


def _integrate_dblquad_kernel_z(phi_d, r_d, r, z):
    """
    Z part the radiation force integral.

    Args:
        phi_d: disc angle integration variable in radians.
        r_d: disc radius integration variable in Rg.
        r: disc radius position in Rg
        z: disc height position in Rg

    Returns:
        Z integral kernel.

    """
    ff0 = (1. - np.sqrt(6. / r_d)) / r_d ** 2.
    delta = r ** 2. + r_d ** 2. + z ** 2. - 2. * r * r_d * np.cos(phi_d)
    ff = ff0 * 1. / delta**2.
    return ff

@jit_integrand
def _integrate_dblquad_kernel_z_jitted(phi_d, r_d, r, z):
    """
    Z part the radiation force integral.

    Args:
        phi_d: disc angle integration variable in radians.
        r_d: disc radius integration variable in Rg.
        r: disc radius position in Rg
        z: disc height position in Rg

    Returns:
        Z integral kernel.

    """
    ff0 = (1. - np.sqrt(6. / r_d)) / r_d ** 2.
    delta = r ** 2. + r_d ** 2. + z ** 2. - 2. * r * r_d * np.cos(phi_d)
    ff = ff0 * 1. / delta**2.
    return ff




def qwind_integration_dblquad(r, z, disk_r_min, disk_r_max):
    """
    Double quad integration of the radiation force integral, using the Nquad
    algorithm. 

    Args:
        r: position r coordinate in Rg.
        z: height z coordinate in Rg.
        r_min: Inner disc integration boundary (usually ISCO).
        r_max: Outer disc integration boundary (defaults to 1600).

    Returns:
        4 element tuple
            0: result of the radial integral.
            1: result of the z integral.
            2: error of the radial integral.
            3: error of the z integral
    """
    r_int, r_error = scipy.integrate.nquad(
        _integrate_dblquad_kernel_r_jitted, ((0, np.pi), (disk_r_min, disk_r_max)),
        args=(r, z),
        opts=[{'points': [0]}, {'points': [r]}])
    z_int, z_error = scipy.integrate.nquad(
        _integrate_dblquad_kernel_z_jitted, ((0, np.pi), (disk_r_min, disk_r_max)),
        args=(r, z),
        opts=[{'points': [0]}, {'points': [r]}])
    r_int = 2. * z * r_int
    z_int = 2. * z**2 * z_int
    return (r_int, z_int, r_error, z_error)
