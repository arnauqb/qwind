"""
Auxiliary file for Numba functions.
"""

import inspect

import numpy as np
import scipy
import numba as nb
from numba import cfunc, float32, int32, jit, jitclass, carray
from numba.types import CPointer, float64, intc, intp
from numba import types
from scipy import LowLevelCallable
from scipy.integrate import quad, nquad
import ctypes

from qwind import constants as const
RG = 14766250380501.244 
#DENSITY_FLOOR = 2e8
#grid = DENSITY_FLOOR * np.ones((500,500)) 
#grid_r_range = np.linspace(0,2000,500)
#grid_z_range = np.linspace(0,2000,501)

def create_jit_integrand_function(integrand_function,args,args_dtype):
    jitted_function = nb.jit(integrand_function, nopython=True)
	
    @nb.cfunc(types.float64(int32,CPointer(float64),types.CPointer(args_dtype)))
    def wrapped(phi_d, r_d, user_data_p):
        #Array of structs
        user_data = nb.carray(user_data_p, 1)
        
        #Extract the data
        r = user_data[0].r
        z = user_data[0].z
        grid = user_data[0].grid
        grid_r_range = user_data[0].grid_r_range
        grid_z_range = user_data[0].grid_z_range
        return jitted_function(phi_d, r_d, r, z, grid, grid_r_range, grid_z_range)
    return wrapped

def integrand_z_jit_dummy(args,args_dtype):
    func=create_jit_integrand_function(_integrate_z_kernel,args,args_dtype)
    return func

def integrand_r_jit_dummy(args,args_dtype):
    func=create_jit_integrand_function(_integrate_r_kernel,args,args_dtype)
    return func


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
def optical_depth_uv_integrand(t_range, r_d, phi_d, r, z, grid, grid_r_range, grid_z_range):
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
def optical_depth_uv(r_d, phi_d, r, z, grid, grid_r_range, grid_z_range):
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
    int_values = optical_depth_uv_integrand(t_range, r_d, phi_d, r, z, grid, grid_r_range, grid_z_range)
    tau_uv_int = np.trapz(x=t_range, y=int_values)
    tau_uv = tau_uv_int * line_element * RG
    return tau_uv

#@jit(nopython=True)
#@jit_integrand
def _integrate_r_kernel(n_arg, x, r, z, grid, grid_r_range, grid_z_range):
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
    phi_d = x[0]
    r_d = x[1]
    tau_uv = optical_depth_uv(r_d, phi_d, r, z, grid, grid_r_range, grid_z_range)
    abs_uv = np.exp(-tau_uv)
    ff0 = nt_rel_factors(r_d) / r_d**2.
    delta = r**2. + r_d**2. + z**2. - 2.*r*r_d * np.cos(phi_d)
    cos_gamma = (r - r_d*np.cos(phi_d)) / delta**2.
    ff = ff0 * cos_gamma * abs_uv
    return ff

#@jit(nopython=True)
#@cfunc(float64(float64, float64, float64, float64, CPointer(float64), CPointer(float64), intp, CPointer(float64), intp))
#@jit_integrand
def _integrate_z_kernel(n_arg, x, r, z, grid, grid_r_range, grid_z_range):
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
    phi_d = x[0]
    r_d = x[1]
    tau_uv = optical_depth_uv(r_d, phi_d, r, z, grid, grid_r_range, grid_z_range)
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
           
        
    
    def integrate(self,
                  r,
                  z,
                  grid,
                  grid_r_range,
                  grid_z_range,
                  disk_r_min,
                  disk_r_max,
                  epsabs=0, 
                  epsrel=1e-3):
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
        #global grid
        #grid = self.radiation.wind.density_grid.grid
        #print(np.min(grid))
        #print(np.mean(grid))
        #global grid_r_range
        #grid_r_range = self.radiation.wind.density_grid.grid_r_range
        #global grid_z_range
        #grid_z_range = self.radiation.wind.density_grid.grid_z_range
        #nr = len(grid_r_range)
        #nz = len(grid_z_range)
        args_dtype = types.Record.make_c_struct([('r', types.float64),
                                                ('z', types.float64),
                                                ('grid', types.NestedArray(dtype=types.float64, shape=grid.shape)),
                                                ('grid_r_range', types.NestedArray(dtype=types.float64, shape=grid_r_range.shape)),
                                                ('grid_z_range', types.NestedArray(dtype=types.float64, shape=grid_z_range.shape)),])
                                                
        args=np.array((r,z,grid,grid_r_range, grid_z_range),dtype=args_dtype)
        func_r = integrand_r_jit_dummy(args,args_dtype)
        func_z = integrand_z_jit_dummy(args,args_dtype)
        integrand_func_r = LowLevelCallable(func_r.ctypes,user_data=args.ctypes.data_as(ctypes.c_void_p))
        integrand_func_z = LowLevelCallable(func_z.ctypes,user_data=args.ctypes.data_as(ctypes.c_void_p))
        r_int, r_error = nquad(integrand_func_r, [(0, np.pi), (disk_r_min, disk_r_max)], opts=[{'epsabs' : epsabs, 'epsrel': epsrel}, {'epsabs': epsabs, 'epsrel': epsrel}])
        #r_int, r_error = nquad(integrand_func_r, [(0, np.pi), (disk_r_min, disk_r_max)]) #, opts=[{'epsabs' : epsabs, 'epsrel': epsrel}, {'epsabs': epsabs, 'epsrel': epsrel}])
        z_int, z_error = nquad(integrand_func_z, [(0, np.pi), (disk_r_min, disk_r_max)], opts=[{'epsabs' : epsabs, 'epsrel': epsrel}, {'epsabs': epsabs, 'epsrel': epsrel}])
        r_int = 2. * z * r_int
        z_int = 2. * z**2 * z_int
        return (r_int, z_int, r_error, z_error)

        #r_int, r_error = scipy.integrate.nquad(
        #    _integrate_r_kernel, ((
        #        0, np.pi), (disk_r_min, disk_r_max)),
        #    args=(r, z, grid, grid_r_range, nr, grid_z_range, nz),
        #    opts=[{'points': [], 'epsabs' : epsabs, 'epsrel': epsrel},
        #        {'points': [], 'epsabs' : epsabs, 'epsrel' : epsrel}])
        #z_int, z_error = scipy.integrate.nquad(
        #    _integrate_z_kernel, ((
        #        0, np.pi), (disk_r_min, disk_r_max)),
        #    args=(r, z, grid, grid_r_range, nr, grid_z_range, nz),
        #    opts=[{'points': [], 'epsabs' : epsabs, 'epsrel': epsrel},
        #        {'points': [], 'epsabs' : epsabs, 'epsrel' : epsrel}])
