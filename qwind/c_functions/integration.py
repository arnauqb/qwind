import os
import numpy as np
from ctypes import *
from scipy import LowLevelCallable
from scipy.integrate import nquad

c_double_p = POINTER(c_double)

try:
    libdir = os.path.dirname(__file__)
    lib = CDLL(os.path.join(libdir, "integrand.so"))
except:
    lib = CDLL(os.path.abspath("integrand.so"))
    
integrand_r = lib.integrand_r
integrand_r.restype = c_double
integrand_r.argtypes = (c_int, POINTER(c_double), c_void_p)

integrand_z = lib.integrand_z
integrand_z.restype = c_double
integrand_z.argtypes = (c_int, POINTER(c_double), c_void_p)

array_initializer = lib.initialize_arrays
array_initializer.restype = c_void_p
array_initializer.argtypes = (POINTER(c_double),
        POINTER(c_double),
        POINTER(c_double),
        POINTER(c_double),
        POINTER(c_double),
        POINTER(c_double),
        c_size_t,
        c_size_t,
        c_size_t)


class Parameters(Structure):
    _fields_ = [("r", c_double), ("z", c_double), ("R_g", c_double)]

class Integrator:

    def __init__(self,
            Rg,
            density_grid,
            grid_r_range,
            grid_z_range,
            mdot_grid,
            uv_fraction_grid,
            grid_disk_range,
            epsabs=0,
            epsrel=1e-4):

        self.Rg = Rg
        self.grid_r_range = grid_r_range
        self.grid_z_range = grid_z_range
        try:
            assert len(self.grid_r_range) == density_grid.shape[0]
            assert len(self.grid_z_range) == density_grid.shape[1]
        except:
            print(len(self.grid_r_range), len(self.grid_z_range), density_grid.shape)
            raise TypeError
        self.density_grid = density_grid.ravel()
        self.mdot_grid = mdot_grid
        self.uv_fraction_grid = uv_fraction_grid
        self.grid_disk_range = grid_disk_range
        self.epsabs = epsabs
        self.epsrel = epsrel
        self.initialize()

    def initialize(self):
        density_grid_p = self.density_grid.ctypes.data_as(c_double_p)
        grid_r_range_p = self.grid_r_range.ctypes.data_as(c_double_p)
        grid_z_range_p = self.grid_z_range.ctypes.data_as(c_double_p)
        mdot_grid_p = self.mdot_grid.ctypes.data_as(c_double_p)
        uv_fraction_grid_p = self.uv_fraction_grid.ctypes.data_as(c_double_p)
        grid_disk_range_p = self.grid_disk_range.ctypes.data_as(c_double_p)
        array_initializer(grid_r_range_p,
                grid_r_range_p,
                density_grid_p,
                mdot_grid_p,
                uv_fraction_grid_p,
                grid_disk_range_p,
                len(self.grid_r_range),
                len(self.grid_z_range),
                len(self.grid_disk_range),
                )
        self.params = Parameters(10., 10., self.Rg)
        user_data = cast(pointer(self.params), c_void_p)
        self.integrand_r_llc = LowLevelCallable(integrand_r, user_data)
        self.integrand_z_llc = LowLevelCallable(integrand_z, user_data)

    def integrate(self, r, z):
        self.params.r = r
        self.params.z = z
        r_int, r_error = nquad(self.integrand_r_llc, ((6., 1600.), (0., np.pi)),
                opts=[{'points': [], 'epsabs' : self.epsabs, 'epsrel': self.epsrel},
                    {'points': [], 'epsabs' : self.epsabs, 'epsrel' : self.epsrel}])
        z_int, z_error = nquad(self.integrand_z_llc, ((6., 1600.), (0., np.pi)),
                opts=[{'points': [], 'epsabs' : self.epsabs, 'epsrel': self.epsrel},
                    {'points': [], 'epsabs' : self.epsabs, 'epsrel' : self.epsrel}])
        r_int = 2 * z * r_int 
        z_int = 2 * z**2 * z_int
        return [r_int, z_int]

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
    
