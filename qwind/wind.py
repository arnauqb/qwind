import shutil
import sys, os
import importlib
import matplotlib.pyplot as plt
from qwind.plot import Plotter

import numpy as np
import pandas as pd
from numba import jit, jitclass
from scipy import interpolate

import qwind.constants as const
from qwind import utils, grid

from assimulo.solvers.sundials import IDAError
from qwind.c_functions import integration

backend = utils.type_of_script()
if(backend == 'jupyter'):
    from tqdm import tqdm_notebook as tqdm
else:
    #tqdm = tqdm_dump
    from tqdm import tqdm as tqdm


def evolve(line, niter):
    line.iterate(niter=niter)
    return line


class Qwind:
    """
    A class used to represent the global properties of the wind, i.e, the accretion disc and black hole properties as well as attributes shared among streamlines.
    """

    def __init__(self,
                 M=1e8,
                 mdot=0.5,
                 spin=0.,
                 eta=0.057,
                 lines_r_min=200,
                 lines_r_max=1600,
                 disk_r_min=6.,
                 disk_r_max=1600,
                 f_x=0.15,
                 f_uv=None,
                 T=2e6,
                 mu=1,
                 modes=[],
                 rho_shielding=2e8,
                 intsteps=1,
                 nr=20,
                 d_max=1e3,
                 save_dir=None,
                 radiation_class="simple_sed",
                 solver="ida",
                 iterations = 1,
                 refresh_grids = True,
                 log_spaced = False,
                 n_cpus=1):
        """
        Parameters
        ----------
        M : float
            Black Hole Mass in solar mass units.
        mdot : float
            Accretion rate (mdot = L / Ledd)
        spin : float
            Spin black hole parameter between [0,1]
        eta : float
            Accretion efficiency (default is for scalar black hole).
        line_r_min : float
            Radius of the first streamline to launch, in R_g units.
        line_r_max : float
            Radius of the last streamline to launch, in R_g units.
        disc_r_min: float
            Minimum radius of acc. disc, default is ISCO for scalar black hole.
        disc_r_max: float
            Maximum radius of acc. disc.
        T : float
            Temperature of the disc atmosphere. Wind is assumed to be isothermal.
        mu : float
            Mean molecular weight ( 1 = pure atomic hydrogen)
        modes : list 
            List of modes for debugging purposes. Available modes are:
                - "gravity_only": Disable radiation force, very useful for debugging.
                - "analytic_fm" : Use analytic approximation of the force multiplier.
        rho_shielding : float
            Initial density of the shielding material.
        intsteps : int
            If old_integral mode enabled, this refined the integration grid.
        save_dir : str
            Directory to save results.
        n_cpus: int
            Number of cpus to use.
        """
        # array containing different modes for debugging #
        self.modes = modes
        self.iterations = iterations
        # black hole and disc variables #
        self.M = M * const.M_SUN
        self.R_g = const.G * self.M / (const.C ** 2)  # gravitational radius
        self.mdot = mdot
        self.spin = spin
        self.mu = mu
        self.disk_r_min = disk_r_min
        self.disk_r_max = disk_r_max
        self.eta = eta
        self.nr = nr + 1 # nr denotes the borders between lines, so...
        self.d_max = d_max
        self.rho_shielding = rho_shielding
        if solver == "euler":
            from qwind.streamline.euler import streamline as streamline_solver
        elif solver == "rk4":
            from qwind.streamline.rk4 import streamline as streamline_solver 
        elif solver == "ida":
            from qwind.streamline.ida import streamline as streamline_solver 
        elif solver == "ida_interp":
            from qwind.streamline.ida_interp import streamline as streamline_solver 
        else:
            print("solver not found")
            raise Exception
        self.streamline_solver = streamline_solver
        self.T = T
        self.v_th = self.thermal_velocity(T)
        self.lines_r_min = lines_r_min
        self.lines_r_max = lines_r_max
        self.f_x = f_x
        self.f_uv = f_uv

        # initialize radiation class
        self.radiation_class = radiation_class
        radiation_module_name = "qwind.radiation." + radiation_class
        radiation_module = importlib.import_module(radiation_module_name)
        self.radiation = radiation_module.Radiation(self)
        self.tau_dr_shielding = self.tau_dr(self.rho_shielding)
        # initialize grid and integrator # 
        if self.radiation_class != "simple_sed":
            self.radiation.grid.initialize_all(init=True)
            self.radiation.grid.update_all(init=True)
            self.radiation.integrator = integration.Integrator(self.R_g,
                    self.radiation.grid.density_grid.values,
                    self.radiation.grid.grid_r_range,
                    self.radiation.grid.grid_z_range,
                    self.radiation.grid.mdot_grid,
                    self.radiation.grid.uv_fraction_grid,
                    self.radiation.grid.grid_disk_range,
                    epsrel=1e-3,
                    epsabs=0)

        # compute initial radii of streamlines
        if log_spaced == True:
            dr_log = (np.log10(self.lines_r_max) - np.log10(self.lines_r_min)) / (self.nr - 1)
            lines_r_range_log = np.array([np.log10(self.lines_r_min) + (i-0.5) * dr_log for i in range(1, self.nr+1)])
            self.lines_r_range = 10**lines_r_range_log
        else:
            dr = (self.lines_r_max - self.lines_r_min) / (self.nr - 1)
            self.lines_r_range = np.array([self.lines_r_min + (i-0.5) * dr for i in range(1, self.nr + 1)])
        self.lines_widths = np.diff(self.lines_r_range)
        self.r_init = self.lines_r_range[0]

                
        # create directory if it doesnt exist. Warning, this overwrites previous outputs.
        if save_dir is not None:
            self.save_dir = save_dir
            try:
                os.mkdir(save_dir)
            except BaseException:
                pass

        
        self.lines = []  # list of streamline objects
        self.lines_hist = []  # save all iterations info

        self.plotter = Plotter(self.radiation.grid)
        self.first_iter = True
        self.iteration_info = []
        #if radiation_class == "qsosed":
        #    self.plotter.plot_all_grids()

    def v_kepler(self, r):
        """
        Keplerian tangential velocity in units of c.

        Args:
            r : r coordinate in R_g.
        Returns:
            v_phi: tangential velocity in units of c.
        """
        v_phi = np.sqrt(1./r)
        return v_phi

    def v_esc(self, d):
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
        return const.EMISSIVITY_CONSTANT * self.R_g

    def thermal_velocity(self, T):
        """
        Thermal velocity for gas with molecular weight mu and temperature T
        """

        return np.sqrt(const.K_B * T / (self.mu * const.M_P)) / const.C

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
        tau_dr = const.SIGMA_T * self.mu * density * self.R_g
        return tau_dr

    def line(self,
             r_0,
             derive_from_ss = False,
             z_0=1.,
             rho_0=2e8,
             T=2e6,
             v_r_0=0.,
             v_z_0=1e7,
             dt=4.096 / 10.,
             **kwargs,
             ):
        """
        Initialises a streamline object.

        Parameters
        -----------
        r_0 : float
            Initial radius in R_g units.
        z_0: float
            Initial height in R_g units.
        rho_0 : float
            Initial number density. Units of 1/cm^3.
        T : float
            Initial stramline temperature.
        v_r_0 : float
            Initial radial velocity in units of cm/s.
        v_z_0 : float
            Initial vertical velocity in units of cm/s.
        dt : float
            Timestep in units of R_g/c.
        """
        if derive_from_ss:
            #z_0 = self.radiation.sed_class.disk_scale_height(r_0)
            temperature = self.radiation.qsosed.disk_nt_temperature4(r_0)**(1./4.)
            v_z_0 = self.thermal_velocity(temperature) * const.C
            rho_0 = self.radiation.qsosed.disk_number_density(r_0)
        if v_z_0 == "thermal":
            temperature = self.radiation.qsosed.disk_nt_temperature4(r_0)**(1./4.)
            v_z_0 = self.thermal_velocity(temperature) * const.C
            #T = temperature
        return self.streamline_solver(
            self.radiation,
            wind=self,
            r_0=r_0,
            z_0=z_0,
            rho_0=rho_0,
            T=T,
            v_r_0=v_r_0,
            v_z_0=v_z_0,
            dt=dt,
            **kwargs,
        )

    def initialize_lines(self,
                         derive_from_ss=False,
                         v_z_0=1e7,
                         niter=5000,
                         rho_0=2e8,
                         z_0=1,
                         dt=4.096/10,
                         show_plots=True,
                         **kwargs):

        self.lines = []
        for i, r in enumerate(self.lines_r_range[:-1]):
            self.lines.append(self.line(r_0=r,
                              line_width = self.lines_widths[i],
                              derive_from_ss=derive_from_ss,
                              v_z_0=v_z_0,
                              rho_0=rho_0,
                              z_0=z_0,
                              dt = dt,
                              **kwargs,
                              ))
    def start_lines(self,
                    derive_from_ss=False,
                    v_z_0=1e7,
                    niter=5000,
                    rho_0=2e8,
                    z_0=1,
                    dt=4.096/10,
                    show_plots=True,
                    **kwargs):
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
        self.progress_bar = tqdm(total=10000)
        if show_plots and self.radiation_class != "simple_sed":
            self.plotter.plot_all_grids()
            plt.show()
        for i in range(0, self.iterations):
            print(f"Iteration {i+1} of {self.iterations}")
            if (self.radiation_class == "qsosed") and ("old_taus" not in self.modes) and (not self.first_iter):
                self.radiation.grid.initialize_all()
                self.radiation.integrator.update(self.radiation.grid)

            self.lines = []
            for i, r in enumerate(self.lines_r_range[:-1]):
                self.lines.append(self.line(r_0=r,
                                            line_width = self.lines_widths[i],
                                            derive_from_ss=derive_from_ss,
                                            v_z_0=v_z_0,
                                            rho_0=rho_0,
                                            z_0=z_0,
                                            dt = dt,
                                            **kwargs,
                                            ))
            for i, line in enumerate(self.lines):
                #print(f"Line {i+1} of {len(self.lines)}")
                line.iterate(niter=niter)
                #except IDAError:
                #    print("Terminating gracefully...")
                #    pass
                if self.radiation_class == "qsosed" and "uv_interp" not in self.modes and "dont_update_grids" not in self.modes: 
                    self.radiation.grid.update_all(init=False)
                    if show_plots:
                        self.plotter.plot_all_grids()
                        plt.show()
            self.mdot_w, self.kinetic_luminosity, self.angle, self.v_terminal = self.compute_wind_properties()
            if self.radiation_class != "simple_sed": 
                if "uv_interp" not in self.modes:
                    self.radiation.initialize_all_grids()
                self.radiation.grid.update_all()
                self.radiation.compute_mass_accretion_rate_grid(self.lines)
                if show_plots:
                    self.plotter.plot_all_grids()
                    plt.show()
                    plt.plot(self.radiation.grid.grid_disk_range, self.radiation.grid.mdot_grid)
                    plt.show()
            self.iteration_info.append([self.mdot_w, self.kinetic_luminosity, self.angle, self.v_terminal])
            if "uv_interp" in self.modes:
                self.first_iter = False #Atention change!
        #return self.lines

    def compute_line_mass_loss(self, line):
        """
        Computes wind mass loss rate after evolving the streamlines.
        """
        mdot_w_total = 0
        width = line.line_width
        area = 2 * np.pi * ((line.r_0 + width/2.)**2. -
                            (line.r_0 - width/2.)**2) * self.R_g**2.
        mdot_w = line.rho_0 * const.M_P * line.v_T_0 * const.C * area
        return mdot_w
    
    def compute_line_kinetic_luminosity(self, line):
        """
        Computes wind kinetic luminosity
        """
        dR = self.lines_r_range[1] - self.lines_r_range[0]
        area = 2 * np.pi * ((line.r_0 + dR/2.)**2. -
                            (line.r_0 - dR/2.)**2) * self.R_g**2.
        mdot_w = line.rho_0 * const.M_P * line.v_T_0 * const.C * area
        kl = 0.5 * mdot_w * (const.C * line.v_T_hist[-1])**2
        return kl 


    def compute_wind_properties(self):
        """
        Computes wind mass loss rate, kinetic luminosity, and the terminal velocity and angle of the fastest streamline.
        """
        escaped_mask = []
        for line in self.lines:
            escaped_mask.append(line.escaped)
        escaped_mask = np.array(escaped_mask, dtype=int)
        wind_exists = False
        lines_escaped = np.array(self.lines)[escaped_mask == True]

        if(len(lines_escaped) == 0):
            print("No wind escapes")
            return [0,0,0,0]

        mdot_w_total = 0
        kinetic_energy_total = 0
        angles = []
        terminal_vs = []
        for line in lines_escaped:
            mdot_w_total += self.compute_line_mass_loss(line)
            kinetic_energy_total += self.compute_line_kinetic_luminosity(line)
            angles.append(line.escaping_angle)
            terminal_vs.append(line.terminal_velocity)

        fastest_line_idx = np.argmax(terminal_vs)
        v_fastest = terminal_vs[fastest_line_idx]
        angle_fastest = angles[fastest_line_idx] * 180 / np.pi 

        return [mdot_w_total, kinetic_energy_total, angle_fastest, v_fastest]
     
    

if __name__ == '__main__':
    qwind = Qwind(M=1e8, mdot=0.1, rho_shielding=2e8,  n_cpus=4, nr=4)
    qwind.start_lines(niter=50000)
    utils.save_results(qwind, "Results")
