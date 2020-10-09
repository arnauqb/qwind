import os
import numpy as np
from qwind import constants
from ctypes import *
from scipy import LowLevelCallable
from scipy.integrate import nquad
from numpy.ctypeslib import ndpointer

c_double_p = POINTER(c_double)

class Parameters(Structure):
    _fields_ = [("r", c_double),
            ("z", c_double),
            ("r_d", c_double),
            ("R_g", c_double),
            ("astar", c_double),
            ("isco", c_double),
            ("r_min", c_double),
            ("r_max", c_double),
            ("epsabs", c_double),
            ("epsrel", c_double),
            ]
library_path = os.path.dirname(__file__)
lib = CDLL(os.path.join(library_path, "qwind_library.so"))

nt_rel_factors = lib.nt_rel_factors
nt_rel_factors.restype = c_double
nt_rel_factors.argtypes = (c_double, c_double, c_double)

workspace_initializer = lib.initialize_integrators

integrate_simplesed_r = lib.integrate_simplesed_r
integrate_simplesed_r.restype = c_double
integrate_simplesed_r.argtypes = [POINTER(Parameters)] 

integrate_simplesed_z = lib.integrate_simplesed_z
integrate_simplesed_z.restype = c_double
integrate_simplesed_z.argtypes = [POINTER(Parameters)]


class IntegratorSimplesed:

    def __init__(self,
            Rg,
            r_min = 6.,
            r_max = 1600.,
            epsabs=0,
            epsrel=1e-4,
            astar =0.,
            isco = 6.):

        self.params = Parameters(
                r = 0.,
                z = 0.,
                r_d = 0.,
                R_g = Rg,
                astar = astar,
                isco = isco,
                r_min = r_min,
                r_max = r_max,
                epsabs = epsabs,
                epsrel = epsrel,
                )
        workspace_initializer()
   
    def integrate(self, r, z):
        self.params.r = r
        self.params.z = z
        r_int = integrate_simplesed_r(byref(self.params))
        z_int = integrate_simplesed_z(byref(self.params))
        return [r_int, z_int]
