import ctypes
import numpy as np
import os
"""
Python Wrapper for C functions.
"""
# Load C Shared Library
libdir = os.path.dirname(__file__)
libname = "/compiled_functions_library.so"
funclib = ctypes.CDLL(libdir + libname)

# interpolate point in grid
funclib.interpolate_point_grid_1d.argtypes = [ctypes.c_double, np.ctypeslib.ndpointer(
    dtype=ctypes.c_double, ndim=1), ctypes.c_int, np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=1), ctypes.c_double]
funclib.interpolate_point_grid_1d.restype = ctypes.c_double


def interpolate_point_grid_1d(x, grid_values, x_range, fill_value):
    x_size = len(x_range)
    value = funclib.interpolate_point_grid_1d(x, grid_values, x_size, x_range, fill_value)
    return value
