import os
import shutil
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from numba import jit, jitclass
from scipy import interpolate

import qwind.constants as const
from qwind import utils


def evolve(line, niter):
    line.iterate(niter=niter)
    return line


class Qwind:
    """
    A class used to represent the global properties of the wind, i.e, the accretion disc and black hole properties as well as attributes shared among streamlines.
    """

    def __init__(self,
                 M=2e8,
                 mdot=0.5,
                 spin=0.,
                 eta=0.06,
                 lines_r_min=200,
                 lines_r_max=1600,
                 disk_r_min=6.,
                 disk_r_max=1400,
                 T=2e6,
                 mu=1,
                 modes=[],
                 rho_shielding=2e8,
                 intsteps=1,
                 nr=20,
                 save_dir=None,
                 radiation_mode="SimpleSED",
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
            Radius of the first streamline to launch, in Rg units.
        line_r_max : float
            Radius of the last streamline to launch, in Rg units.
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

        self.n_cpus = n_cpus

        # array containing different modes for debugging #
        self.modes = modes
        # black hole and disc variables #
        self.M = M * const.M_SUN
        self.mdot = mdot
        self.spin = spin
        self.mu = mu
        self.disk_r_min = disk_r_min
        self.disk_r_max = disk_r_max
        self.eta = eta
        self.nr = nr
        self.rho_shielding = rho_shielding

        self.RG = const.G * self.M / (const.C ** 2)  # gravitational radius

        self.bol_luminosity = self.mdot * self.eddington_luminosity
        #self.tau_dr_0 = self.tau_dr(rho_shielding)
        self.tau_dr_shielding = self.tau_dr(rho_shielding)
        self.v_thermal = self.thermal_velocity(T)
        self.lines_r_min = lines_r_min
        self.lines_r_max = lines_r_max

        # load radiation module
        if (radiation_mode == "QSOSED"):
            from qwind.radiation import qsosed
            self.radiation = qsosed.QSOSED(self)
        elif (radiation_mode == "SimpleSED"):
            from qwind.radiation import simple_sed
            self.radiation = simple_sed.SimpleSED(self)
        else:
            print("Unknown radiation module.")
            sys.exit()

        print("disk_r_min: %f \n disk_r_max: %f" %
              (self.disk_r_min, self.disk_r_max))

        # create directory if it doesnt exist. Warning, this overwrites previous outputs.
        if save_dir is not None:
            self.save_dir = save_dir
            try:
                os.mkdir(save_dir)
            except BaseException:
                pass

        # compute initial radii of streamlines
        dr = (self.lines_r_max - self.lines_r_min) / (nr - 1)
        self.lines_r_range = [self.lines_r_min +
                              (i-0.5) * dr for i in range(1, nr+1)]
        self.r_init = self.lines_r_range[0]

        self.lines = []  # list of streamline objects
        self.lines_hist = []  # save all iterations info

    def norm2d(self, vector):
        """
        Norm of a 2D vector.
        """
        return np.sqrt(vector[0] ** 2 + vector[-1] ** 2)

    def dist2d(self, x, y):
        """
        Distance between vectors x and y in 2D.
        """
        dr = y[0] - x[0]
        dz = y[2] - x[2]
        return np.sqrt(dr**2 + dz**2)

    def v_kepler(self, r):
        """
        Keplerian tangential velocity in units of c.

        Args:
            r : r coordinate in Rg.
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
        return const.EMISSIVITY_CONSTANT * self.RG

    def thermal_velocity(self, T):
        """
        Thermal velocity for gas with molecular weight mu and temperature T
        """

        return np.sqrt(const.K_B * T / (self.mu * const.M_P)) / const.C

    def density_ss(self, r):
        R = r * 2 * self.RG
        cut = 18 * (self.M / const.M_SUN)**(2./21.) * \
            (self.mdot / self.eta)**(16./21.) * 2 * self.RG
        if (R <= cut):
            rho = 5.24e-4 * (self.M/const.M_SUN)**(-1.) * \
                (self.mdot/self.eta)**(-2.) * (R/(2*self.RG))**(3./2.)
        else:
            rho = 4.66 * (self.M/const.M_SUN)**(-7./10.) * (self.mdot /
                                                            self.eta)**(2./5.) * (R/(2*self.RG))**(-33./20.)
        return rho / const.M_P

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
        tau_dr = const.SIGMA_T * self.mu * density * self.RG
        return tau_dr

    def line(self,
             r_0="ss",
             z_0=10.,
             rho_0=2e8,
             T=2e6,
             v_r_0=0.,
             v_z_0=1e7,
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
            wind=self,
            r_0=r_0,
            z_0=z_0,
            rho_0=rho_0,
            T=T,
            v_r_0=v_r_0,
            v_z_0=v_z_0,
            dt=dt
        )

    def start_lines(self, v_z_0=1e7, niter=5000, rho=None, z_0=10):
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
            if (v_z_0 == "auto"):
                if (r > self.radiation.sed_class.corona_radius):
                    if (r < 2 * self.radiation.sed_class.corona_radius):
                        v_z_0 = self.thermal_velocity(2.3e6) * const.C
                    else:
                        v_z_0 = self.thermal_velocity(
                            self.radiation.sed_class.disk_temperature4(r)**(1./4.)) * const.C
                else:
                    print("streamline would be inside corona radius, ignoring.")
                    continue
            else:
                v_z_0 = v_z_0
            if(rho is None):
                rho = self.density_ss(r)
            self.lines.append(
                self.line(r_0=r, v_z_0=v_z_0, rho_0=rho, z_0=z_0))
        i = 0
        if(self.n_cpus == 1):
            for line in self.lines:
                i += 1
                print("Line %d of %d" % (i, len(self.lines)))
                line.iterate(niter=niter)
            self.mdot_w = self.compute_wind_mass_loss()
            return self.lines
        print("multiple cpus")
        niter_array = niter * np.ones(len(self.lines))
        niter_array = niter_array.astype('int')

        with Pool(self.n_cpus) as multiprocessing_pool:
            self.lines = multiprocessing_pool.starmap(
                evolve, zip(self.lines, niter_array))
        self.mdot_w = self.compute_wind_mass_loss()
        return self.lines

    def compute_wind_mass_loss(self):
        """
        Computes wind mass loss rate after evolving the streamlines.
        """
        escaped_mask = []
        for line in self.lines:
            escaped_mask.append(line.escaped)
        escaped_mask = np.array(escaped_mask, dtype=int)
        wind_exists = False
        lines_escaped = np.array(self.lines)[escaped_mask == True]

        if(len(lines_escaped) == 0):
            print("No wind escapes")
            return 0

        dR = self.lines_r_range[1] - self.lines_r_range[0]
        mdot_w_total = 0

        for line in lines_escaped:
            area = 2 * np.pi * ((line.r_0 + dR/2.)**2. -
                                (line.r_0 - dR/2.)**2) * self.RG**2.
            mdot_w = line.rho_0 * const.M_P * line.v_T_0 * const.C * area
            mdot_w_total += mdot_w

        return mdot_w_total


if __name__ == '__main__':
    qwind = Qwind(M=1e8, mdot=0.1, rho_shielding=2e8,  n_cpus=4, nr=4)
    qwind.start_lines(niter=50000)
    utils.save_results(qwind, "Results")
