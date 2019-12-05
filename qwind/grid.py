import numpy as np
from scipy import integrate
from numba import jit


@jit(nopython=True)
def find_index(r,z, grid_r_range, grid_z_range):
    r_idx = np.argmin(np.abs(grid_r_range - r))
    z_idx = np.argmin(np.abs(grid_z_range - z))
    return [r_idx, z_idx]


class Grid:
    
    def __init__(self, r_i=0, r_f=2000, z_i=1, z_f=1000, n_r=500, n_z=500):
        self.grid = np.zeros((n_r, n_z))
        self.grid_r_range = np.linspace(r_i, r_f, n_r)
        self.grid_z_range = np.linspace(z_i, z_f, n_z)


         
    def get_line_boundaries(self, line, dr):
        """
        Given line computes the width of the line for all its range.
        """
        r_hist = line.r_hist 
        z_hist = line.z_hist 
        if line.escaped == True:
            line_width = np.array(r_hist) / line.r_0 * dr / 2
            lower = []
            upper = []
            center = []
            for i in range(0,len(r_hist)-1):
                p1 = np.array([r_hist[i], z_hist[i]])
                p2 = np.array([r_hist[i+1], z_hist[i+1]])
                v = p2 - p1
                v_norm =np.linalg.norm(v)
                if v_norm != 0:
                    v = v / v_norm
                vperp = np.cross(v , [0,0,1])
                p_upper = p1 + line_width[i] * vperp[[0,1]]
                p_lower = p1 - line_width[i] * vperp[[0,1]]
                center.append(p1)
                lower.append(p_lower)
                upper.append(p_upper)

            lower = np.array(lower)
            upper = np.array(upper)
            center = np.array(center)

        else:
            x_range = np.linspace(line.r_0 - dr/2, line.r_0+dr/2, len(r_hist))
            center = np.array([r_hist, z_hist]).T
            lower = np.array([x_range, line.z_0 * np.ones(len(z_hist))]).T
            upper = np.array([x_range, np.max(z_hist) * np.ones(len(z_hist))]).T
        return [lower, center, upper]

    def line_width_parametrisation(self, x_low, y_low, x_up, y_up):
        t = 0
        while t<=1:
            x_new = x_low + t * (x_up - x_low)
            y_new = y_low + t * (y_up - y_low)
            yield [x_new, y_new]
            t += 0.01


    def fill_rho_values(self, line):
        b_low , b_center, b_up = get_line_boundaries(line, dr)
        rho_hist = line.rho_hist# rho_interp(t_range)
        for i in range(0,len(rho_hist) -1):
            if line.escaped == True:
                rho = rho_hist[i]
            else:
                rho = 2e8 #np.mean(rho_hist)
            x_low, y_low = b_low[i]
            x_up, y_up = b_up[i]
            line_param = line_width_parametrisation(x_low, y_low, x_up, y_up)
            while True:
                try:
                    x_new, y_new = next(line_param)
                    x_new_arg, y_new_arg = find_index(x_new, y_new, self.grid_r_range, self.grid_z_range)
                    if self.grid[x_new_arg, y_new_arg] != 0:
                        self.grid[x_new_arg, y_new_arg] = np.max((rho, self.grid[x_new_arg, y_new_arg]))
                    else:
                        self.grid[x_new_arg, y_new_arg] = rho
                except StopIteration:
                    break
    
    def update_grid(self, wind):
        for line in wind.lines:
            self.fill_rho_values(line)


    
    
    
