import numpy as np
import qwind.constants as const
from qwind import radiation
from scipy import interpolate
from multiprocessing import Pool
from qwind import utils
import os
import shutil
import pandas as pd
from numba import jitclass, jit
from qwind import aux_numba


# check backend to import appropiate progress bar #
def tqdm_dump(array):
    return array
backend = utils.type_of_script()
if(backend == 'jupyter'):
    from tqdm import tqdm_notebook as tqdm
else:
    tqdm = tqdm_dump

def evolve(line, niter):
    line.iterate(niter=niter)
    return line

class Qwind:
    """
    A class used to represent the global properties of the wind, i.e, the accretion disc and black hole properties as well as attributes shared among streamlines.
    """
    def __init__(self, M = 2e8, mdot = 0.5, spin=0.,eta=0.0313, fx = 0.15, r_in = 200, r_out = 1600, r_min = 6., r_max=1400, T=2e6, mu = 1, modes =[], rho_shielding = 2e8, intsteps=1, nr=20, save_dir="Results", radiation_mode = "Qwind", n_cpus = 1):
        """
        Parameters
        ----------
        r_init : float
            Radius of the first streamline to launch, in Rg units.
        M : float
            Black Hole Mass in solar mass units.
        mdot : float
            Accretion rate (mdot = L / Ledd)
        spin : float
            Spin black hole parameter between [0,1]
        eta : float
            Accretion efficiency (default is for scalar black hole).
        fx : float
            Ratio of luminosity in X-Rays, fx = Lx / Lbolumetric
        Rmin : float
            Minimum radius of acc. disc, default is ISCO for scalar black hole.
        Rmax : float
            Maximum radius of acc. disc.
        T : float
            Temperature of the disc atmosphere. Wind is assumed to be isothermal.
        mu : float
            Mean molecular weight ( 1 = pure hydrogen)
        modes : list 
            List of modes for debugging purposes. Available modes are:
                - 'old_integral': Non adaptive disc integration (much faster but convergence is unreliable.)
                - 'altopts': Alternative opacities (experimental)
                - 'gravityonly': Disable radiation force, very useful for debugging.
        rho_shielding : float
            Initial density of the shielding material.
        intsteps : int
            If old_integral mode enabled, this refined the integration grid.
        save_dir : str
            Directory to save results.
        """

        self.n_cpus = n_cpus
        
        # array containing different modes for debugging #
        self.modes = modes
        # black hole and disc variables #
        self.M = M * const.Ms
        self.mdot = mdot
        self.spin = spin
        self.mu = mu
        self.fx = fx
        self.r_min = r_min 
        self.r_max = r_max 
        self.r_in = r_in
        self.r_out = r_out
        self.eta = eta

        
        self.Rg = const.G * self.M / (const.c ** 2) # gravitational radius
        self.rho_shielding = rho_shielding
        self.bol_luminosity = self.mdot * self.eddington_luminosity
        self.xray_luminosity = self.mdot * self.eddington_luminosity * self.fx
        
        self.tau_dr_0 = self.tau_dr(rho_shielding)
        self.v_thermal = self.thermal_velocity(T)
       
        # create directory if it doesnt exist. Warning, this overwrites previous outputs.
        self.save_dir = save_dir
        try:
            os.mkdir(save_dir)
        except BaseException:
            pass

        #try:
        self.radiation = radiation.Radiation(self)
        #except:
        #    print("Radiation mode not found")
        #    return None
        
        self.reff_hist = [0] # for debugging
        dr = (r_out - r_in) / (nr -1)
        self.lines_r_range = [r_in + (i-0.5) * dr for i in range(1,nr+1)]
        self.r_init = self.lines_r_range[0]

        self.nr = nr
        self.lines = [] # list of streamline objects
        self.lines_hist = [] # save all iterations info

    def norm2d(self, vector):
        return np.sqrt(vector[0] ** 2 + vector[-1] ** 2)
    
    def dist2d(self, x, y):
        # 2d distance in cyl coordinates #
        dr = y[0] - x[0]
        dz = y[2] - x[2]
        return np.sqrt(dr**2 + dz**2)   
    
    def v_kepler(self, r ):
        """
        Keplerian tangential velocity in units of c.
        """
        
        return np.sqrt(1. / r)

    def v_esc(self,d):
        """
        Escape velocity in units of c.
        
        Parameters
        -----------
        d : float
            spherical radial distance.
        """
        
        return np.sqrt(2. / d)

    @property
    def eddington_luminosity(self):
        """ 
        Returns the Eddington Luminosity. 
        """
        return const.emissivity_constant * self.Rg

    def thermal_velocity(self, T):
        """
        Thermal velocity for gas with molecular weight mu and temperature T
        """
        
        return np.sqrt(const.k_B * T / (self.mu * const.m_p)) / const.c

    def tau_dr(self, density):
        """ 
        Differential optical depth.
        
        Parameters
        -----------
        opacity : float
            opacity of the material.
        density : float
            shielding density.
        """
        tau_dr = const.sigma_t * self.mu * density * self.Rg
        return tau_dr
    
    def line(self,
            r_0=375.,
            z_0=1., 
            rho_0=2e8,
            T=2e6,
            v_r_0=0.,
            v_z_0=5e7,
            dt=4.096 / 10.
            ):
        """
        Initialises a streamline object.
        
        Parameters
        -----------
        r_0 : float
            Initial radius in Rg units.
        z_0: float
            Initial height in Rg units.
        rho_0 : float
            Initial number density. Units of 1/cm^3.
        T : float
            Initial stramline temperature.
        v_r_0 : float
            Initial radial velocity in units of cm/s.
        v_z_0 : float
            Initial vertical velocity in units of cm/s.
        dt : float
            Timestep in units of Rg/c.
        """
        from qwind.streamline import streamline
        return streamline(
            self.radiation,
            parent = self,
            r_0 = r_0,
            z_0 = z_0,
            rho_0 = rho_0,
            T = T,
            v_r_0 = v_r_0,
            v_z_0 = v_z_0,
            dt = dt
            )

    
    def start_lines(self, v_z_0 = 5e7, niter=5000):        
        """
        Starts and evolves a set of equally spaced streamlines.
        
        Parameters
        -----------
        nr : int 
            Number of streamlines.
        v_z_0 : float
            Initial vertical velocity.
        niter : int 
            Number of timesteps.
        """
        print("Starting line iteration")

        self.lines = []

        for i, r in enumerate(self.lines_r_range):
            self.lines.append(self.line(r_0=r,v_z_0=v_z_0))
        i = 0
        if(self.n_cpus==1):
            for line in self.lines:
               i += 1
               print("Line %d of %d"%(i, len(self.lines)))
               line.iterate(niter=niter)
            return self.lines
        print("multiple cpus")
        niter_array = niter * np.ones(len(self.lines))
        niter_array = niter_array.astype('int')

        with Pool(self.n_cpus) as multiprocessing_pool:
            self.lines = multiprocessing_pool.starmap(evolve, zip(self.lines, niter_array))
        return self.lines

    def save_results(self, folder_name = "Results"):
        """
        Saves results to filename.
        """
        try:
            os.mkdir(folder_name)
        except:
            answer = input("warning, folder exists, delete? (y/N)")
            if (answer == 'y'):
                shutil.rmtree(folder_name)
                os.mkdir(folder_name)
            else:
                return 0

        metadata_file = os.path.join(folder_name, "metadata.txt") 
        with open(metadata_file, "w") as f:
            f.write("M: \t %.2e\n"%self.M)
            f.write("Mdot: \t %.2e\n"%self.mdot)
            f.write("a: \t %.2e\n"%self.spin)

        for i, line in enumerate(self.lines):
            line_name = "line_%02d"%i
            position_file = os.path.join(folder_name, line_name + "_positions.csv")
            radiation_file = os.path.join(folder_name, line_name + "_radiation.csv")
            position_data = {
               'R' : line.r_hist,
               'P' : line.phi_hist,
               'Z' : line.z_hist,
               'X' : line.x_hist,
               'V_R' : line.v_r_hist,
               'V_PHI' : line.v_phi_hist,
               'V_Z' : line.v_z_hist,
               'V_T' : line.v_T_hist,
               'a' : line.a_hist,
            }

            df_pos = pd.DataFrame.from_dict(position_data)
            df_pos.to_csv(position_file, index = False)

            radiation_data = {
                'rho' : line.rho_hist,
                'xi' : line.xi_hist,
                'fm' : line.fm_hist,
                'tau_dr' : line.tau_dr_hist,
                'tau_uv' : line.tau_uv_hist,
                'tau_x' : line.tau_x_hist,
                'dv_dr' : line.dv_dr_hist,
                'dr_e' : line.dr_e_hist,
            }
            df_rad = pd.DataFrame.from_dict(radiation_data)
            df_rad.to_csv(radiation_file, index = False)

        return 1



if __name__ == '__main__':
    qwind = Qwind(modes = ['old_integral'], n_cpus = 3)
    qwind.start_lines(niter=10000)
    qwind.save_results("Results")
