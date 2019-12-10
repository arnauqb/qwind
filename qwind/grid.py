import numpy as np
from scipy import integrate
from numba import jit, njit
from qwind import constants as const
import pyquad
import sys, inspect
import matplotlib.pyplot as plt

N_R_DEFAULT = 1000
N_Z_DEFAULT = 1001
R_MAX_DEFAULT = 2000
Z_MAX_DEFAULT = 2000 
GRID_R_RANGE = np.geomspace(1, R_MAX_DEFAULT, N_R_DEFAULT)
GRID_Z_RANGE = np.geomspace(1, Z_MAX_DEFAULT, N_Z_DEFAULT) 
DENSITY_GRID = 2e8 * np.ones((N_R_DEFAULT,N_Z_DEFAULT))
IONIZATION_GRID = 1e3 * np.ones((N_R_DEFAULT, N_Z_DEFAULT))
#taux_grid_r_range = np.linspace(0, R_MAX_DEFAULT, N_R_DEFAULT)
#taux_grid_z_range= np.linspace(0, Z_MAX_DEFAULT, N_Z_DEFAULT)
#taux_density_grid = 2e8 * np.ones((N_R_DEFAULT,N_Z_DEFAULT))
#taux_ionization_grid = 1e3 * np.ones((N_R_DEFAULT, N_Z_DEFAULT))

@jit(nopython=True)
def find_index(r,z, grid_r_range, grid_z_range):
    r_idx = np.argmin(np.abs(grid_r_range - r))
    z_idx = np.argmin(np.abs(grid_z_range - z))
    return [r_idx, z_idx]

class Grid:
    """
    General grid class
    """
    def __init__(self, initial_value):
        self.grid = initial_value * np.ones((N_R_DEFAULT, N_Z_DEFAULT))

    def get_value(self, r,z):
        r = np.array(r, ndmin=1)
        z = np.array(z, ndmin=1)
        return_values = np.zeros_like(r)
        mask1 = (r >= GRID_R_RANGE[-1]) # + (z >= GRID_Z_RANGE[-1])
        mask2 = (z >= GRID_Z_RANGE[-1])
        mask3 = mask1 * mask2
        mask_total = mask1 + mask2
        r_arg = np.searchsorted(GRID_R_RANGE, r, side="right")
        z_arg = np.searchsorted(GRID_Z_RANGE, z, side="right")
        if mask1.size > 0:
            return_values[mask1] = self.grid[-1 * np.ones_like(mask1), z_arg[mask1]] 
        if mask2.size > 0:
            return_values[mask2] = self.grid[r_arg[mask2], -1*np.ones_like(mask2)] 
        if mask3.size > 0:
            return_values[mask3] = self.grid[-1,-1]
        return_values[~mask_total] = self.grid[r_arg[~mask_total], z_arg[~mask_total]]
        return return_values


class DensityGrid(Grid):
    
    def __init__(self, rho_0):
        super().__init__(rho_0)
            
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
        if np.max(line.z_hist) > 50:#line.escaped == True:
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
        if np.max(line.z_hist) > 50:
            rectangles = self.get_line_boundaries(line,dr)
            for i,rectangle in enumerate(rectangles):
                rectangle_idx = []
                for vertex in rectangle:
                    r_arg, z_arg = find_index(vertex[0], vertex[1], GRID_R_RANGE, GRID_Z_RANGE)
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
                self.grid[r1[0]:r3[0], r4[1]:r2[1]] = line.rho_hist[i]
        else:
            rectangle = self.get_line_boundaries(line, dr)
            rectangle_idx = []
            for vertex in rectangle:
                r_arg, z_arg = find_index(vertex[0], vertex[1], GRID_R_RANGE, GRID_Z_RANGE)
                rectangle_idx.append([r_arg, z_arg])
            r1, r2, r3, r4 = rectangle_idx
            #rec_un, counts = rec_unique = np.unique(rectangle_idx, return_counts=True, axis=-1)
            assert (r2[1] >= r4[1])
            assert (r1[0] <= r3[0])
            self.grid[r1[0]:r3[0]+1, r4[1]:r2[1]+1] = line.rho_0

    def update_grid(self, wind):
        for line in wind.lines:
            self.fill_rho_values(line)

@njit
def _opacity_xray(xi):
    #xi = np.array(xi)
    #return_values = const.SIGMA_T * np.ones_like(xi)
    #mask = xi < 1e5
    #return_values[mask] *= 100
    #return return_values
    if xi < 1e5:
        return 100 * const.SIGMA_T
    else:
        return const.SIGMA_T


def optical_depth_x_integrand(t, r, z):#, density_grid, ionization_grid, grid_r_range, grid_z_range):
    r = t * r
    z = t * z
    r_arg = np.searchsorted(GRID_R_RANGE, r, side="left")
    z_arg = np.searchsorted(GRID_Z_RANGE, z, side="left")
    r_arg = min(r_arg, GRID_R_RANGE.shape[0] - 1)
    z_arg = min(z_arg, GRID_Z_RANGE.shape[0] - 1)
    line_element = np.sqrt(r**2 + z**2)
    density = DENSITY_GRID[r_arg, z_arg]
    xi = IONIZATION_GRID[r_arg, z_arg]
    dtau = _opacity_xray(xi) * density * line_element
    #dtau = density * line_element 
    return dtau


class OpticalDepthXrayGrid(Grid):
    def __init__(self, Rg, initial_value = 0):
        super().__init__(initial_value)
        rr, zz = np.meshgrid(GRID_R_RANGE, GRID_Z_RANGE)
        self.rz_grid = np.array([rr.flatten(), zz.flatten()]).T
        self.Rg = Rg

    def update_grid(self, density_grid, ionization_grid):
        global DENSITY_GRID 
        DENSITY_GRID = density_grid.grid.copy()
        global IONIZATION_GRID 
        IONIZATION_GRID = ionization_grid.grid.copy()
        res, error = pyquad.quad_grid(optical_depth_x_integrand, 0, 1, self.rz_grid, parallel=True, cache=False)
        res= res.reshape(len(GRID_Z_RANGE), len(GRID_R_RANGE)).T * self.Rg
        self.grid = res
        
class IonizationParameterGrid(Grid):
    
    def __init__(self, xray_luminosity, Rg, initial_value = 1):
        super().__init__(initial_value)
        self.xray_luminosity = xray_luminosity
        rr, zz = np.meshgrid(GRID_R_RANGE, GRID_Z_RANGE)
        rz_grid = np.array([rr.flatten(), zz.flatten()]).T
        self.Rg = Rg
        self.d2_grid = (rz_grid[:,0]**2 + rz_grid[:,1]**2).reshape(N_Z_DEFAULT, N_R_DEFAULT).T * self.Rg**2
    
    def update_grid(self, density_grid, tau_x_grid):
        xi = self.xray_luminosity * np.exp(-tau_x_grid.grid) / (density_grid.grid * self.d2_grid) 
        self.grid = xi + 1e-11

