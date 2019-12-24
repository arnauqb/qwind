import os
import numpy as np
from qwind import constants
from ctypes import *
from scipy import LowLevelCallable
from scipy.integrate import nquad
from numpy.ctypeslib import ndpointer

#c_array = ndpointer(c_double, flags="C_CONTIGUOUS")
c_double_p = POINTER(c_double)

class Parameters(Structure):
    _fields_ = [("r", c_double),
            ("z", c_double),
            ("r_d", c_double),
            ("grid_r_range", c_double_p),
            ("grid_z_range", c_double_p),
            ("n_r", c_size_t),
            ("n_z", c_size_t),
            ("density_grid", c_double_p),
            ("uv_fraction_grid", c_double_p),
            ("mdot_grid", c_double_p),
            ("grid_disk_range", c_double_p),
            ("n_disk", c_size_t),
            ("R_g", c_double),
            ("astar", c_double),
            ("isco", c_double),
            ("r_min", c_double),
            ("r_max", c_double),
            ("epsabs", c_double),
            ("epsrel", c_double),
            ]
try:
    libdir = os.path.dirname(__file__)
    lib = CDLL(os.path.join(libdir, "qwind_library.so"))
except:
    lib = CDLL(os.path.abspath("qwind_library.so"))

test_c = lib.test
test_c.argtypes = [POINTER(Parameters)]
def test(parameters):
    test_c(byref(parameters))
    
tau_uv_disk_blob_c = lib.tau_uv_disk_blob
tau_uv_disk_blob_c.restype = c_double
tau_uv_disk_blob_c.argtypes = (c_double, c_double, c_double, c_double)

nt_rel_factors = lib.nt_rel_factors
nt_rel_factors.restype = c_double
nt_rel_factors.argtypes = (c_double, c_double, c_double)

workspace_initializer = lib.initialize_integrators

integrate_r = lib.integrate_r
integrate_r.restype = c_double
integrate_r.argtypes = [POINTER(Parameters)] 

integrate_z = lib.integrate_z
integrate_z.restype = c_double
integrate_z.argtypes = [POINTER(Parameters)]

integrate_notau_r = lib.integrate_notau_r
integrate_notau_r.restype = c_double
integrate_notau_r.argtypes = [POINTER(Parameters)]

integrate_notau_z = lib.integrate_notau_z
integrate_notau_z.restype = c_double
integrate_notau_z.argtypes = [POINTER(Parameters)]

integrate_simplesed_r = lib.integrate_simplesed_r
integrate_simplesed_r.restype = c_double
integrate_simplesed_r.argtypes = [POINTER(Parameters)] 

integrate_simplesed_z = lib.integrate_simplesed_z
integrate_simplesed_z.restype = c_double
integrate_simplesed_z.argtypes = [POINTER(Parameters)]


get_arg_c = lib.get_arg
get_arg_c.restype = c_int
get_arg_c.argtypes = (c_double, c_double_p, c_size_t)


def get_arg(value, array):
    n = len(array)
    array_p = array.ctypes.data_as(array)
    return get_arg_c(value, array_p, n)

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


class Integrator:

    def __init__(self,
            Rg,
            density_grid,
            grid_r_range,
            grid_z_range,
            mdot_grid,
            uv_fraction_grid,
            grid_disk_range,
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
                grid_r_range = grid_r_range.ctypes.data_as(c_double_p),
                grid_z_range = grid_z_range.ctypes.data_as(c_double_p),
                n_r = len(grid_r_range),
                n_z = len(grid_z_range),
                density_grid = density_grid.ctypes.data_as(c_double_p), 
                uv_fraction_grid = uv_fraction_grid.ctypes.data_as(c_double_p),
                mdot_grid = mdot_grid.ctypes.data_as(c_double_p),
                grid_disk_range = grid_disk_range.ctypes.data_as(c_double_p),
                n_disk = len(grid_disk_range),
                R_g = Rg,
                astar = astar,
                isco = isco,
                r_min = r_min,
                r_max = r_max,
                epsabs = epsabs,
                epsrel = epsrel,
                )
        workspace_initializer()
   
    def update(self, grid):
        self.params.density_grid = grid.density_grid.values.ctypes.data_as(c_double_p)
        self.params.mdot_grid = grid.mdot_grid.ctypes.data_as(c_double_p)

    def integrate(self, r, z):
        self.params.r = r
        self.params.z = z
        r_int = integrate_r(byref(self.params))
        z_int = integrate_z(byref(self.params))
        return [r_int, z_int]

    def integrate_notau(self, r, z):
        self.params.r = r
        self.params.z = z
        r_int = integrate_notau_r(byref(self.params))
        z_int = integrate_notau_z(byref(self.params))
        return [r_int, z_int]
    
    def tau_uv_disk_blob(self, r_d, phi_d, r, z):
        tau_uv = tau_uv_disk_blob_c(r_d, phi_d, r, z)
        return tau_uv * constants.SIGMA_T * self.params.R_g

if __name__ == "__main__":

    r = 100
    z = 50
    n_r = 10
    n_z = 10
    Rg = 14766250380501.244 
    density_grid = 2e8 * np.ones((1000,1001))
    grid_r_range = np.linspace(6,1000,1000)
    grid_z_range = np.linspace(6,1000,1001)
    integ = Integrator(Rg, density_grid, grid_r_range, grid_z_range)
    print(integ.integrate(r,z))
    
