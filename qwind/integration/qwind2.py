"""
Auxiliary file for Numba functions.
"""

import inspect

import numpy as np
import scipy
from numba import cfunc, float32, int32, jit, jitclass, njit
from numba.types import CPointer, float64, intc
from scipy import LowLevelCallable
from scipy.integrate import quad
from qwind import grid
from qwind import constants as const
from qwind.c_functions import wrapper
RG = 0. 
#UV_FRACTION_GRID_RANGE = np.ones(500)
GRID_1D_RANGE = grid.GRID_1D_RANGE 
MDOT_GRID = grid.MDOT_GRID
UV_FRACTION_GRID = grid.UV_FRACTION_GRID 
#MDOT_GRID_RANGE = np.linspace(6, 1600, 500)
DENSITY_GRID = grid.DENSITY_GRID 
GRID_R_RANGE = grid.GRID_R_RANGE
GRID_Z_RANGE = grid.GRID_Z_RANGE
N_R = grid.N_R
N_Z = grid.N_Z
#mdot_grid = MDOT_GRID

def jit_integrand(integrand_function):
    """
    Turns a function into a LowLevelCallable function.
    """

    jitted_function = jit(integrand_function, nopython=True)
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

@njit
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

def get_density_value(r,z):
    return_values = np.zeros_like(r)
    mask = np.where((r >= grid_r_range[-1]) + (r <= grid_r_range[0]) + (z >= grid_z_range[-1]) + (z <= grid_z_range[0]))[0]
    if mask.size > 0:
        return_values[mask] = 1e2 #self.density_floor
        r_arg = np.searchsorted(grid_r_range, r[~mask], side="left")
        z_arg = np.searchsorted(grid_z_range, z[~mask], side="left")
        return_values[~mask] = grid[r_arg, z_arg]
    else:
        r_arg = np.searchsorted(grid_r_range, r, side="left")
        z_arg = np.searchsorted(grid_z_range, z, side="left")
        return_values = grid[r_arg, z_arg]
    return return_values

@njit
def get_mdot_value(r_d):
    rd_arg = min(np.searchsorted(GRID_1D_RANGE, r_d), len(GRID_1D_RANGE)-1)
    mdot = MDOT_GRID[rd_arg]
    return mdot 

@njit
def get_uv_fraction_value(r_d):
    rd_arg = min(np.searchsorted(GRID_1D_RANGE, r_d), len(GRID_1D_RANGE)-1)
    uv_fraction = UV_FRACTION_GRID[rd_arg]
    return uv_fraction 

#@njit
def optical_depth_uv(r_d, phi_d, r, z):
    """
    UV optical depth.
    
    Args:
        r: radius in Rg units.
        z: height in Rg units.
    
    Returns:
        UV optical depth at point (r,z) 
    """
    #line_element = np.sqrt(r**2 + r_d**2 + z**2 - 2 * r * r_d * np.cos(phi_d))
    #t_range = np.linspace(0,1)
    #int_values = optical_depth_uv_integrand(t_range, r_d, phi_d, r, z)
    #tau_uv_int = np.trapz(x=t_range, y=int_values)
    #tau_uv = tau_uv_int * line_element * RG
    tau_uv = wrapper.tau_uv_disk_blob(r_d, phi_d, r, z, DENSITY_GRID, GRID_R_RANGE, GRID_Z_RANGE) * const.SIGMA_T * RG
    return tau_uv


def _integrate_r_kernel(phi_d, r_d, r, z):
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
    #tau_uv = wrapper.use_tau_uv_disk_blob(r_d, phi_d, r, z, DENSITY_GRID.ravel(), GRID_R_RANGE, GRID_Z_RANGE, N_R, N_Z)   #optical_depth_uv(r_d, phi_d, r, z)
    mdot = get_mdot_value(r_d)
    uv_fraction = get_uv_fraction_value(r_d)
    ff0 = nt_rel_factors(r_d) / r_d**2.
    delta = r**2. + r_d**2. + z**2. - 2.*r*r_d * np.cos(phi_d)
    cos_gamma = (r - r_d*np.cos(phi_d)) / delta**2.
    ff = ff0 * cos_gamma * mdot * uv_fraction #* np.exp(-tau_uv)
    return ff

def _integrate_z_kernel(phi_d, r_d, r, z):
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
    #tau_uv = optical_depth_uv(r_d, phi_d, r, z)
    mdot = get_mdot_value(r_d)
    uv_fraction = get_uv_fraction_value(r_d)
    ff0 = nt_rel_factors(r_d) / r_d**2.
    #ff0 = ( 1 - np.sqrt(6/r_d)) / r_d**2
    delta = r ** 2. + r_d ** 2. + z ** 2. - 2. * r * r_d * np.cos(phi_d)
    ff = ff0 * 1. / delta**2. * mdot * uv_fraction #* np.exp(-tau_uv)
    return ff

def _integrate_r_kernel_notau(phi_d, r_d, r, z):
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
    mdot = 1#get_mdot_value(r_d)
    uv_fraction = 1#get_uv_fraction_value(r_d)
    #ff0 = nt_rel_factors(r_d) / r_d**2.
    ff0 = ( 1 - np.sqrt(6/r_d)) / r_d**2
    delta = r**2. + r_d**2. + z**2. - 2.*r*r_d * np.cos(phi_d)
    cos_gamma = (r - r_d*np.cos(phi_d)) / delta**2.
    ff = ff0 * cos_gamma * mdot * uv_fraction
    return ff

def _integrate_z_kernel_notau(phi_d, r_d, r, z):
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
    mdot = 1#get_mdot_value(r_d)
    uv_fraction = 1#get_uv_fraction_value(r_d)
    #ff0 = nt_rel_factors(r_d) / r_d**2.
    ff0 = ( 1 - np.sqrt(6/r_d)) / r_d**2
    delta = r ** 2. + r_d ** 2. + z ** 2. - 2. * r * r_d * np.cos(phi_d)
    ff = ff0 * 1. / delta**2. * mdot * uv_fraction
    return ff


class Integrator:
    def __init__(self, radiation, tau_uv_integrand=False):

        self.radiation = radiation
        global RG
        RG = self.radiation.wind.RG
        global DENSITY_GRID
        DENSITY_GRID = self.radiation.density_grid.grid
        global UV_FRACTION_GRID
        UV_FRACTION_GRID = self.radiation.uv_fraction_grid.grid
        global MDOT_GRID
        MDOT_GRID = self.radiation.mdot_grid.grid
        get_mdot_value.recompile()
        get_uv_fraction_value.recompile()
        if not tau_uv_integrand:
            self._integrate_z = jit_integrand(_integrate_z_kernel_notau)
            self._integrate_r = jit_integrand(_integrate_r_kernel_notau)
            #self._integrate_z = _integrate_z_kernel_notau
            #self._integrate_r = _integrate_r_kernel_notau
        else:
            #optical_depth_uv.recompile()
            self._integrate_z = jit_integrand(_integrate_z_kernel)
            self._integrate_r = jit_integrand(_integrate_r_kernel)
            #self._integrate_z = _integrate_z_kernel
            #self._integrate_r = _integrate_r_kernel

    def integrate(self, r, z, disk_r_min=6, disk_r_max=1600, epsabs=0, epsrel=1e-4):
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
            self._integrate_r, ((
                0, np.pi), (disk_r_min, disk_r_max)),
            args=(r, z),
            opts=[{'points': [], 'epsabs' : epsabs, 'epsrel': epsrel},
                {'points': [], 'epsabs' : epsabs, 'epsrel' : epsrel}])
        z_int, z_error = scipy.integrate.nquad(
            self._integrate_z, ((
                0, np.pi), (disk_r_min, disk_r_max)),
            args=(r, z),
            opts=[{'points': [], 'epsabs' : epsabs, 'epsrel': epsrel},
                {'points': [], 'epsabs' : epsabs, 'epsrel' : epsrel}])
        r_int = 2. * z * r_int
        z_int = 2. * z**2 * z_int
        return (r_int, z_int, r_error, z_error)

