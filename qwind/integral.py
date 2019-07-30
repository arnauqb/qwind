import ctypes
import numpy as np
import os

"""
Python wrapper for C-compiled functions.
"""

# Load C Shared Library

libdir = os.path.dirname(__file__)
libname = "integral_c.so"
funclib = ctypes.CDLL(os.path.join(libdir, libname))

# delta gas disk
funclib.delta_gas_disk.argtypes = [
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double
]
funclib.delta_gas_disk.restype = ctypes.c_double

# non adaptive r
funclib.non_adaptive_integral_r.argtypes = [
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_int,
    ctypes.c_int
]
funclib.non_adaptive_integral_r.restype = ctypes.c_double

# non adaptive z
funclib.non_adaptive_integral_z.argtypes = [
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_int,
    ctypes.c_int
]
funclib.non_adaptive_integral_z.restype = ctypes.c_double


def non_adaptive_integral_r(r, z, r_min = 6., r_max = 1400., n_phi = 100, n_r = 250):
    result = funclib.non_adaptive_integral_r(r,z, r_min, r_max, n_phi, n_r)
    return result

def non_adaptive_integral_z(r, z, r_min = 6., r_max = 1400., n_phi = 100, n_r = 250):
    result = funclib.non_adaptive_integral_z(r,z, r_min, r_max, n_phi, n_r)
    return result

def non_adaptive_integral(r, z, r_min = 6., r_max = 1400., n_phi = 100, n_r = 250):
    return [
            non_adaptive_integral_r(r,z,r_min,r_max, n_phi, n_r),
            non_adaptive_integral_z(r,z,r_min,r_max, n_phi, n_r)
    ] 

