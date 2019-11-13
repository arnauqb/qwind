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

@jit(nopython=True)
def nt_rel_factors(r, astar=0, isco=6):
    """
    Relatistic A,B,C factors of the Novikov-Thorne model.
    
    Parameters
    ¦   Black Hole Mass in solar Masses
    -----------
    r : float
    ¦   disk radial distance.
    """
    yms = np.sqrt(isco)
    y1 = 2 * np.cos((np.arccos(astar) - np.pi) / 3)
    y2 = 2 * np.cos((np.arccos(astar) + np.pi) / 3)
    y3 = -2 * np.cos(np.arccos(astar) / 3)
    y = np.sqrt(r)
    C = 1 - 3 / r + 2 * astar / r**(1.5)
    B = 3 * (y1 - astar)**2 * np.log(
       (y - y1) / (yms - y1)) / (y * y1 * (y1 - y2) * (y1 - y3))
    B += 3 * (y2 - astar)**2 * np.log(
       (y - y2) / (yms - y2)) / (y * y2 * (y2 - y1) * (y2 - y3))
    B += 3 * (y3 - astar)**2 * np.log(
       (y - y3) / (yms - y3)) / (y * y3 * (y3 - y1) * (y3 - y2))
    A = 1 - yms / y - 3 * astar * np.log(y / yms) / (2 * y)
    factor = (A-B)/C
    return factor

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
    ff = ff0 * cos_gamma# * abs_uv
    return ff

@jit_integrand
def _integrate_dblquad_kernel_r_jitted_rel(phi_d, r_d, r, z):
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
    ff0 = nt_rel_factors(r_d) / r_d**2.
    delta = r**2. + r_d**2. + z**2. - 2.*r*r_d * np.cos(phi_d)
    cos_gamma = (r - r_d*np.cos(phi_d)) / delta**2.
    ff = ff0 * cos_gamma# * abs_uv
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
    ff0 = (1. - np.sqrt(6. / r_d)) / r_d**2.
    delta = r ** 2. + r_d ** 2. + z ** 2. - 2. * r * r_d * np.cos(phi_d)
    ff = ff0 * 1. / delta**2.
    return ff

@jit_integrand
def _integrate_dblquad_kernel_z_jitted_rel(phi_d, r_d, r, z):
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
    ff0 = nt_rel_factors(r_d) / r_d**2.
    delta = r ** 2. + r_d ** 2. + z ** 2. - 2. * r * r_d * np.cos(phi_d)
    ff = ff0 * 1. / delta**2.# * abs_uv
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
        _integrate_dblquad_kernel_r_jitted, ((
            0, np.pi), (disk_r_min, disk_r_max)),
        args=(r, z),
        opts=[{'points': [0]}, {'points': [r]}])
    z_int, z_error = scipy.integrate.nquad(
        _integrate_dblquad_kernel_z_jitted, ((
            0, np.pi), (disk_r_min, disk_r_max)),
        args=(r, z),
        opts=[{'points': [0]}, {'points': [r]}])
    r_int = 2. * z * r_int
    z_int = 2. * z**2 * z_int
    return (r_int, z_int, r_error, z_error)

def qwind_integration_rel(r, z, disk_r_min, disk_r_max):
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
        _integrate_dblquad_kernel_r_jitted_rel, ((
            0, np.pi), (disk_r_min, disk_r_max)),
        args=(r, z),
        opts=[{'points': [0]}, {'points': [r]}])
    z_int, z_error = scipy.integrate.nquad(
        _integrate_dblquad_kernel_z_jitted_rel, ((
            0, np.pi), (disk_r_min, disk_r_max)),
        args=(r, z),
        opts=[{'points': [0]}, {'points': [r]}])
    r_int = 2. * z * r_int
    z_int = 2. * z**2 * z_int
    return (r_int, z_int, r_error, z_error)


## old integral ##


phids = np.linspace(0, np.pi, 100 + 1)
deltaphids = np.asarray([phids[i + 1] - phids[i]
                         for i in range(0, len(phids) - 1)])
rds = np.geomspace(6, 1400, 250 + 1)
deltards = np.asarray([rds[i + 1] - rds[i] for i in range(0, len(rds) - 1)])


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
def qwind_old_integration(r, z):
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


