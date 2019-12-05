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
RG = 0. 
DENSITY_FLOOR = 2e8
grid = DENSITY_FLOOR * np.ones((500,500)) 
grid_r_range = np.zeros(500) 
grid_z_range = np.zeros(500) 

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

@jit(nopython=True)
def optical_depth_uv_integrand(t_range, r_d, phi_d, r, z):
    x = r_d * np.cos(phi_d) + t_range * (r - r_d * np.cos(phi_d))
    y = r_d * np.sin(phi_d) + t_range * (- r_d * np.sin(phi_d))
    z = t_range * z
    r = np.sqrt(x**2 + y**2)
    r_arg = np.searchsorted(grid_r_range, r, side="left")
    z_arg = np.searchsorted(grid_z_range, z, side="left")
    density_values = []
    for i in range(0,len(r_arg)):
        dvalue = grid[r_arg[i], z_arg[i]]
        density_values.append(dvalue)
    dtau = const.SIGMA_T * np.array(density_values) #np.array(density_values) 
    return dtau


@jit(nopython=True)
def optical_depth_uv(r_d, phi_d, r, z):
    """
    UV optical depth.
    
    Args:
        r: radius in Rg units.
        z: height in Rg units.
    
    Returns:
        UV optical depth at point (r,z) 
    """
    line_element = np.sqrt(r**2 + r_d**2 + z**2 - 2 * r * r_d * np.cos(phi_d))
    t_range = np.linspace(0,1)
    int_values = optical_depth_uv_integrand(t_range, r_d, phi_d, r, z)
    tau_uv_int = np.trapz(x=t_range, y=int_values)
    tau_uv = tau_uv_int * line_element * RG
    return tau_uv


@jit_integrand
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
    tau_uv = optical_depth_uv(r_d, phi_d, r, z)
    abs_uv = np.exp(-tau_uv)
    ff0 = nt_rel_factors(r_d) / r_d**2.
    delta = r**2. + r_d**2. + z**2. - 2.*r*r_d * np.cos(phi_d)
    cos_gamma = (r - r_d*np.cos(phi_d)) / delta**2.
    ff = ff0 * cos_gamma * abs_uv
    return ff

@jit_integrand
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
    tau_uv = optical_depth_uv(r_d, phi_d, r, z)
    abs_uv = np.exp(-tau_uv)
    ff0 = nt_rel_factors(r_d) / r_d**2.
    delta = r ** 2. + r_d ** 2. + z ** 2. - 2. * r * r_d * np.cos(phi_d)
    ff = ff0 * 1. / delta**2. * abs_uv
    return ff

class Integrator:
    def __init__(self, radiation):

        self.radiation = radiation
        global RG
        RG = self.radiation.wind.RG
        global grid
        grid = self.radiation.wind.density_grid.grid
        global grid_r_range
        grid_r_range = self.radiation.wind.density_grid.grid_r_range
        global grid_z_range
        grid_z_range = self.radiation.wind.density_grid.grid_z_range
    
        self.Rg = self.radiation.wind.RG
        
    
    def integrate(self, r, z, disk_r_min, disk_r_max, epsabs=0, epsrel=1e-11):
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
            _integrate_r_kernel, ((
                0, np.pi), (disk_r_min, disk_r_max)),
            args=(r, z),
            opts=[{'points': [], 'epsabs' : epsabs, 'epsrel': epsrel},
                {'points': [], 'epsabs' : epsabs, 'epsrel' : epsrel}])
        z_int, z_error = scipy.integrate.nquad(
            _integrate_z_kernel, ((
                0, np.pi), (disk_r_min, disk_r_max)),
            args=(r, z),
            opts=[{'points': [], 'epsabs' : epsabs, 'epsrel': epsrel},
                {'points': [], 'epsabs' : epsabs, 'epsrel' : epsrel}])
        r_int = 2. * z * r_int
        z_int = 2. * z**2 * z_int
        return (r_int, z_int, r_error, z_error)
