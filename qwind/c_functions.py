import ctypes
import numpy as np

# load C shared library
libdir = "./"
libname = "c_functions.so"

funclib = ctypes.CDLL(libdir + libname)

# integration 
funclib.integrate.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, np.ctypeslib.ndpointer(dtype = ctypes.c_double, ndim=1)]
funclib.integrate.restype = ctypes.c_void_p


def integrate(r,z,tau_uv):
    results = np.zeros(2, dtype = ctypes.c_double)
    funclib.integrate(r,z,tau_uv,results)
    return results
