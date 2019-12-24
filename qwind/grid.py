import numpy as np
from scipy import integrate
from numba import jit, njit
from qwind import constants as const
import pyquad
import sys, inspect
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from skimage.draw import line_aa as compute_line_coordinates
import cmocean.cm as colormaps
from qwind.c_functions import wrapper

#N_R = 1000
#N_Z = 1001
#N_DISK = 100
#R_MAX_DEFAULT = 3000
#Z_MAX_DEFAULT = 3000 
#GRID_R_RANGE = np.linspace(0.01, R_MAX_DEFAULT, N_R)
#GRID_Z_RANGE = np.linspace(0.01, Z_MAX_DEFAULT, N_Z) 
#DENSITY_GRID = 2e8 * np.ones((N_R, N_Z))
#IONIZATION_GRID = 1e3 * np.ones((N_R, N_Z))
#GRID_DISK_RANGE = np.linspace(6, 1600, N_DISK)
#UV_FRACTION_GRID = np.ones(N_DISK)
#MDOT_GRID = np.ones(N_DISK)

@njit
def _opacity_xray(xi):
    if xi < 1e5:
        return 100 * const.SIGMA_T
    else:
        return const.SIGMA_T

def _opacity_xray_array(xi):
    xi = np.array(xi)
    return_values = const.SIGMA_T * np.ones_like(xi)
    mask = xi < 1e5
    return_values[mask] *= 100
    return return_values
    #if xi < 1e5:
    #    return 100 * const.SIGMA_T
    #else:
    #    return const.SIGMA_T
class GridTemplate:
    def __init__(self):
        pass

    def get_value(self, r, z):
        args = self.get_arg(r, z)
        return_values = self.values[args[0], args[1]]
        return return_values

    def get_arg(self, r, z):
        r_arg = np.minimum(np.searchsorted(self.grid.grid_r_range, r, side="right"), len(self.grid.grid_r_range)-1)
        z_arg = np.minimum(np.searchsorted(self.grid.grid_z_range, z, side="right"), len(self.grid.grid_z_range)-1)
        return [r_arg, z_arg]
    
    def plot(self, cmap="thermal", vmin=None, vmax=None):
        cmap = getattr(colormaps, cmap)
        if vmin is None or vmax is None:
            plt.pcolormesh(self.grid.grid_r_range, self.grid.grid_z_range, self.values.T, cmap = cmap)
        else:
            plt.pcolormesh(self.grid.grid_r_range, self.grid.grid_z_range, self.values.T, cmap = cmap, norm=LogNorm(vmin=vmin, vmax=vmax))
        plt.colorbar()
        plt.show()


class Grid:
    """
    General grid class
    """
    def __init__(self, wind, grid_r_min=0.01, grid_z_min=0.01, n_r=1000, n_z=1000, n_disk=100):

        self.wind = wind
        self.grid_r_range = np.linspace(grid_r_min, self.wind.d_max, n_r)
        self.grid_z_range = np.linspace(grid_z_min, self.wind.d_max, n_z)
        self.n_r = n_r
        self.n_z = n_z
        self.n_disk = n_disk
        self.grid_disk_range = np.linspace(wind.disk_r_min, wind.disk_r_max, n_disk)
        #self.density_grid = DensityGrid(self)
        #self.ionization_grid = IonizationParameterGrid(self)
        #self.tau_x_grid = OpticalDepthXrayGrid(self)

    def update_all(self, init=False):
        if not init:
            self.density_grid.update()
        self.ionization_grid.update()
        self.tau_x_grid.update()
        self.ionization_grid.update()

    def initialize_all(self, first_iter=True, init=False):
        self.density_grid = DensityGrid(self)
        self.ionization_grid = IonizationParameterGrid(self)
        self.tau_x_grid = OpticalDepthXrayGrid(self)
        self.update_all(init=True)
        if init:
            self.mdot_grid = self.wind.radiation.mdot_0 * np.ones(self.n_disk) 
            self.uv_fraction_grid = self.wind.radiation.uv_radial_flux_fraction

    
class DensityGrid(GridTemplate):
    
    def __init__(self, grid):
        self.values = grid.wind.rho_shielding * np.ones((grid.n_r, grid.n_z))
        self.grid = grid
        #super().__init__(rho_0)
            
    def get_line_boundaries(self, line, dr):
        """
        Given line computes the width of the line for all its range.
            r2
          /    \
        r1      r3
          \    /
            r4
        """
        r_hist = line.r_hist 
        z_hist = line.z_hist 
        if True:#np.max(line.z_hist) > 1:#line.escaped == True:
            line_width = np.array(r_hist) / line.r_0 * dr / 2
            rectangles = []
            for i in range(0,len(r_hist)-1):
                p1 = np.array([r_hist[i], z_hist[i]])
                p2 = np.array([r_hist[i+1], z_hist[i+1]])
                v = p2 - p1
                v_norm =np.linalg.norm(v)
                if v_norm != 0:
                    v_unity = v / v_norm
                else:
                    v_unity = v
                vperp = np.cross(v_unity , [0,0,1])
                r4 = p1 + line_width[i] * vperp[[0,1]]
                r1 = p1 - line_width[i] * vperp[[0,1]]
                r2 = r1 + v
                r3 = r4 + v
                rectangle = [r1,r2,r3,r4]
                rectangles.append(rectangle)
        else:
            r1 = [line.r_0 - dr/2, 0]#line.z_0]
            r2 = [line.r_0 - dr/2, np.max(line.z_hist)]
            r3 = [line.r_0 + dr/2, np.max(line.z_hist)]
            r4 = [line.r_0 + dr/2, 0]#line.z_0]
            rectangles = [r1,r2,r3,r4]

        return rectangles 

    def fill_rho_values(self, line):
        dr = line.line_width
        if True: #np.max(line.z_hist) > 1:
            rectangles = self.get_line_boundaries(line,dr)
            for i,rectangle in enumerate(rectangles):
                rectangle_idx = []
                for vertex in rectangle:
                    r_arg, z_arg = self.get_arg(vertex[0], vertex[1])
                    rectangle_idx.append([r_arg, z_arg])
                rectangle_idx = np.array(rectangle_idx)
                #rec_un, counts = rec_unique = np.unique(rectangle_idx, return_counts=True, axis = 0)
                #if counts.max() > 1:
                #    for vert in rec_un:
                #        self.grid[[vert[1], vert[0]]] = line.rho_hist[i]
                r1_idx = np.argmin(rectangle_idx[:,0])
                r2_idx = np.argmax(rectangle_idx[:,1])
                r3_idx = np.argmax(rectangle_idx[:,0])
                r4_idx = np.argmin(rectangle_idx[:,1])
                r1 = rectangle_idx[r1_idx]
                r2 = rectangle_idx[r2_idx]
                r3 = rectangle_idx[r3_idx]
                r4 = rectangle_idx[r4_idx]
                assert (r2[1] >= r4[1])
                assert (r1[0] <= r3[0])
                # r1, r2, r3, r4 = rectangle_idx
                self.values[r1[0]:r3[0], r4[1]:r2[1]] = line.rho_hist[i]
        else:
            rectangle = self.get_line_boundaries(line, dr)
            rectangle_idx = []
            for vertex in rectangle:
                r_arg, z_arg = self.get_arg(vertex[0], vertex[1])
                rectangle_idx.append([r_arg, z_arg])
            r1, r2, r3, r4 = rectangle_idx
            #rec_un, counts = rec_unique = np.unique(rectangle_idx, return_counts=True, axis=-1)
            assert (r2[1] >= r4[1])
            assert (r1[0] <= r3[0])
            self.values[r1[0]:r3[0]+1, r4[1]:r2[1]+1] = line.rho_0

    def update(self):
        for line in self.grid.wind.lines:
            self.fill_rho_values(line)

class OpticalDepthXrayGrid(GridTemplate):
    def __init__(self, grid):
        self.grid = grid
        self.R_g = grid.wind.R_g
        rr, zz = np.meshgrid(grid.grid_r_range, grid.grid_z_range, indexing="ij")
        self.rz_grid = np.array([rr.flatten(), zz.flatten()]).T
        self.values = np.zeros((grid.n_r, grid.n_z))

    def update(self):
        self.values = wrapper.update_tau_x_grid(self.grid.density_grid.values, self.grid.ionization_grid.values, self.grid.grid_r_range, self.grid.grid_z_range) * self.R_g * const.SIGMA_T
        
class IonizationParameterGrid(GridTemplate):
    def __init__(self, grid):
        self.grid = grid
        rr, zz = np.meshgrid(grid.grid_r_range, grid.grid_z_range, indexing="ij")
        rz_grid = np.array([rr.flatten(), zz.flatten()]).T
        self.R_g = grid.wind.R_g
        self.d2_grid = (rz_grid[:,0]**2 + rz_grid[:,1]**2).reshape(grid.n_r, grid.n_z) * self.R_g**2
    
    def update(self):
        xi = self.grid.wind.radiation.xray_luminosity * np.exp(-self.grid.tau_x_grid.values) / (self.grid.density_grid.values * self.d2_grid) 
        self.values = xi + 1e-11

