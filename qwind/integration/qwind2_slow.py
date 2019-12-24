"""
Auxiliary file for Numba functions.
"""

import inspect
from qwind import grid

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

class Integrator:
    def __init__(self, radiation):

        self.radiation = radiation
        global RG
        RG = self.radiation.wind.RG
        self.R_RANGE = grid.GRID_R_RANGE
        self.Z_RANGE = grid.GRID_Z_RANGE
        args_dtype = types.Record.make_c_struct([('r', types.float64),
                                                ('z', types.float64),
                                                ('r_range', types.NestedArray(dtype=types.float64, shape=grid.GRID_R_RANGE.shape)),
                                                ('z_range', types.NestedArray(dtype=types.float64, shape=grid.GRID_Z_RANGE.shape)),
                                                ('mdot_grid', types.NestedArray(dtype=types.float64, shape=grid.MDOT_GRID.shape)),])
                                                
        args=np.array((r,z, self.radiation.mdot_grid, self.R_RANGE, self.Z_RANGE ),dtype=args_dtype)
        func_r = integrand_r_jit_dummy(args,args_dtype)
        func_z = integrand_z_jit_dummy(args,args_dtype)
        self.integrand_func_r = LowLevelCallable(func_r.ctypes,user_data=args.ctypes.data_as(ctypes.c_void_p))
        self.integrand_func_z = LowLevelCallable(func_z.ctypes,user_data=args.ctypes.data_as(ctypes.c_void_p))

           
    def integrate(self,
                  r,
                  z,
                  mdot_grid,
                  disk_r_min=6,
                  disk_r_max=1600,
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
        r_int, r_error = nquad(self.integrand_func_r, [(0, np.pi), (disk_r_min, disk_r_max)], opts=[{'epsabs' : epsabs, 'epsrel': epsrel}, {'epsabs': epsabs, 'epsrel': epsrel}])
        z_int, z_error = nquad(self.integrand_func_z, [(0, np.pi), (disk_r_min, disk_r_max)], opts=[{'epsabs' : epsabs, 'epsrel': epsrel}, {'epsabs': epsabs, 'epsrel': epsrel}])
        r_int = 2. * z * r_int
        z_int = 2. * z**2 * z_int
        return (r_int, z_int, r_error, z_error)

    @staticmethod
    def create_jit_integrand_function(integrand_function,args,args_dtype):
        jitted_function = nb.jit(integrand_function, nopython=True, cache=True)
    	
        #@nb.cfunc(types.float64(int32,CPointer(float64),types.CPointer(args_dtype)))
        #def wrapped(phi_d, r_d, user_data_p):
        #    #Array of structs
        #    user_data = nb.carray(user_data_p, 1)
        #    
        #    #Extract the data
        #    r = user_data[0].r
        #    z = user_data[0].z
        #    grid = user_data[0].grid
        #    grid_r_range = user_data[0].grid_r_range
        #    grid_z_range = user_data[0].grid_z_range
        #    return jitted_function(phi_d, r_d, r, z, grid, grid_r_range, grid_z_range)
        #return wrapped
        @cfunc(float64(intc, CPointer(float64)))
        def wrapped(n, xx):
            return jitted_function(xx[0], xx[1], xx[2], xx[3])
        return LowLevelCallable(wrapped.ctypes)

    
    @staticmethod
    def integrand_z_jit_dummy(args,args_dtype):
        func=create_jit_integrand_function(_integrate_z_kernel,args,args_dtype)
        return func
    
    @staticmethod
    def integrand_r_jit_dummy(args,args_dtype):
        func=create_jit_integrand_function(_integrate_r_kernel,args,args_dtype)
        return func
    
    
    @staticmethod
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
      
    #@jit(nopython=True)
    #@jit_integrand
    @staticmethod
    #def _integrate_r_kernel(n_arg, x, r, z, r_range, z_range, mdot_grid):
    def _integrate_z_kernel(x, *args):
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
        r = args[0]
        z = args[1]
        r_range = args[2]
        z_range = args[3]
        mdot_grid = args[4]
        mdot_arg = np.argmin(np.abs(r_range) - r_d)
        mdot = mdot_grid[mdot_arg]
        ff0 = nt_rel_factors(r_d) / r_d**2.
        delta = r**2. + r_d**2. + z**2. - 2.*r*r_d * np.cos(phi_d)
        cos_gamma = (r - r_d*np.cos(phi_d)) / delta**2.
        ff = ff0 * cos_gamma * mdot 
        return ff
    
    #@jit(nopython=True)
    #@cfunc(float64(float64, float64, float64, float64, CPointer(float64), CPointer(float64), intp, CPointer(float64), intp))
    #@jit_integrand
    @staticmethod
    #def _integrate_z_kernel(n_arg, x, r, z, grid, grid_r_range, grid_z_range):
    def _integrate_z_kernel(x, *args):
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
        r = args[0]
        z = args[1]
        r_range = args[2]
        z_range = args[3]
        mdot_grid = args[4]
        mdot_arg = np.argmin(np.abs(r_range) - r_d)
        mdot = mdot_grid[mdot_arg]
        ff0 = nt_rel_factors(r_d) / r_d**2.
        delta = r ** 2. + r_d ** 2. + z ** 2. - 2. * r * r_d * np.cos(phi_d)
        ff = ff0 * 1. / delta**2. * mdot
        return ff
    
