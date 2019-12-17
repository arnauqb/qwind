import ctypes
from ctypes import CFUNCTYPE
import numpy as np
import os
from numpy.ctypeslib import ndpointer
from numba import njit, jit
import numba as nb
from ctypes import *

"""
Python Wrapper for C functions.
"""


# Load C shared library
libdir = os.path.dirname(__file__)
libname = "grid_utils.so"
funclib = ctypes.CDLL(os.path.join(libdir, libname))

# update tau_x_grid
funclib.update_tau_x_grid.argtypes = [
        ndpointer(dtype=ctypes.c_double, flags="C_CONTIGUOUS"),
        ndpointer(dtype=ctypes.c_double, flags="C_CONTIGUOUS"),
        ndpointer(dtype=ctypes.c_double, flags="C_CONTIGUOUS"),
        ndpointer(dtype=ctypes.c_double, flags="C_CONTIGUOUS"),
        ndpointer(dtype=ctypes.c_double, flags="C_CONTIGUOUS"),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ]
funclib.update_tau_x_grid.restype = ctypes.c_void_p 

def update_tau_x_grid(density_grid, ionization_grid, grid_r_range, grid_z_range):
    n_r = len(grid_r_range)
    n_z = len(grid_z_range)
    density_grid = density_grid.ravel()
    ionization_grid = ionization_grid.ravel()
    tau_x_grid = np.empty((n_r, n_z)).ravel()
    funclib.update_tau_x_grid(density_grid, ionization_grid, tau_x_grid, grid_z_range, grid_r_range, n_r, n_z)
    tau_x_grid = tau_x_grid.reshape(n_r, n_z)
    return tau_x_grid

# draw line of sight
funclib.drawline.argtypes = [
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ndpointer(dtype=ctypes.c_int, flags="C_CONTIGUOUS"),
        ctypes.c_size_t,
        ]
funclib.drawline.restype = ctypes.c_void_p

def line_coordinates(x1, y1, x2, y2):
    dx = abs(x2-x1)
    dy = abs(y2-y1)
    length = max(dx,dy) + 1
    results = np.empty(2 * length, dtype=ctypes.c_int)
    funclib.drawline(x1, y1, x2, y2, results, length)
    new_results = np.empty_like(results).reshape(length,2)
    new_results[:,0] = results[:length]
    new_results[:,1] = results[length:]
    return new_results

# optical depth uv
funclib.tau_uv.argtypes = [
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int32,
        ctypes.c_int32,
        ndpointer(dtype=ctypes.c_double, flags="C_CONTIGUOUS"),
        ctypes.c_size_t,
        ]
funclib.tau_uv.restype = ctypes.c_double

def tau_uv(r, z, density_grid):
    n_z = density_grid.grid.shape[1]
    r_arg, z_arg = density_grid.get_arg(r,z)
    tau_uv = funclib.tau_uv(r, z, r_arg, z_arg, density_grid.grid.ravel(), n_z)
    return tau_uv

# optical depth uv disk blob
tau_uv_disk_blob = funclib.tau_uv_disk_blob
tau_uv_disk_blob.argtypes = [
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        POINTER(c_double),
        POINTER(c_double),
        POINTER(c_double),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ]
tau_uv_disk_blob.restype = ctypes.c_double
#tau_uv_disk_blob = funclib.tau_uv_disk_blob
#tau_uv_disk_blob.argtypes = [
#        ctypes.c_double,
#        ctypes.c_double,
#        ctypes.c_double,
#        ctypes.c_double,
#        POINTER(c_double),
#        POINTER(c_double),
#        POINTER(c_double),
#        ctypes.c_size_t,
#        ctypes.c_size_t,
#        ]
#tau_uv_disk_blob.restype = ctypes.c_double


@njit
def use_tau_uv_disk_blob(r_d, phi_d, r, z, density_grid, grid_r_range, grid_z_range, n_r, n_z):
    tau_uv = tau_uv_disk_blob(r_d,
            phi_d,
            r,
            z,
            density_grid, 
            grid_r_range,
            grid_z_range,
            n_r,
            n_z)
    return tau_uv

ctype_wrapping = CFUNCTYPE(c_double, c_double, c_double, c_double, c_double, POINTER(c_double), POINTER(c_double), POINTER(c_double), c_size_t, c_size_t)(tau_uv_disk_blob)

from numba import types, cfunc
c_sig = types.double(types.double,
        types.double,
        types.double,
        types.double,
        types.CPointer(types.double),
        types.CPointer(types.double),
        types.CPointer(types.double),
        types.intc,
        types.intc)

@cfunc(c_sig)
def use_ctype_wrapping(r_d, phi_d, r, z, density_grid, grid_r_range, grid_z_range, n_r, n_z):
    #return ctype_wrapping(r_d, phi_d, r, z, density_grid, grid_r_range, grid_z_range, n_r, n_z)
    tau_uv =tau_uv_disk_blob(r_d,
            phi_d,
            r,
            z,
            density_grid, 
            grid_r_range,
            grid_z_range,
            n_r,
            n_z)
    return tau_uv










