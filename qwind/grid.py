import numpy as np
from scipy import integrate
from numba import jit


@jit(nopython=True)
def find_index(r,z, grid_r_range, grid_z_range):
    r_idx = np.argmin(np.abs(grid_r_range - r))
    z_idx = np.argmin(np.abs(grid_z_range - z))
    return [r_idx, z_idx]


class Grid:
    
    def __init__(self, r_i=0, r_f=2000, z_i=1, z_f=1000, n_r=500, n_z=501):
        self.grid = np.zeros((n_z, n_r)).T
        self.grid_r_range = np.linspace(r_i, r_f, n_r)
        self.grid_z_range = np.linspace(z_i, z_f, n_z)

         
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
                    r_arg, z_arg = find_index(vertex[0], vertex[1], self.grid_r_range, self.grid_z_range)
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
                r_arg, z_arg = find_index(vertex[0], vertex[1], self.grid_r_range, self.grid_z_range)
                rectangle_idx.append([r_arg, z_arg])
            r1, r2, r3, r4 = rectangle_idx
            rec_un, counts = rec_unique = np.unique(rectangle_idx, return_counts=True, axis=-1)
            assert (r2[1] >= r4[1])
            assert (r1[0] <= r3[0])
            self.grid[r1[0]:r3[0], r4[1]:r2[1]] = line.rho_0

    def update_grid(self, wind):
        for line in wind.lines:
            self.fill_rho_values(line)


    
    
    
