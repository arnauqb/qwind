"""
This module implements the streamline class, that initializes and evolves a streamline from the disc.
"""

import numpy as np
import scipy.integrate
from qwind import utils
from decimal import Decimal
from qwind import constants as const
import pickle

# check backend to import appropiate progress bar #


def tqdm_dump(array):
    return array


backend = utils.type_of_script()
if(backend == 'jupyter'):
    from tqdm import tqdm_notebook as tqdm
else:
    tqdm = tqdm_dump


class streamline():
    """
    This class represents a streamline. It inherits from the wind class all the global properties of the accretion disc, black hole and atmosphere.
    """

    def __init__(
            self,
            radiation_class,
            wind,
            r_0=375.,
            z_0=10.,
            rho_0=2e8,
            T=2e6,
            v_z_0=1e7,
            v_r_0=0.,
            dt=4.096 / 10.,
            integral_epsabs=0,
            integral_epsrel=1e-4,
            terminate_stalling=True,
    ):
        """
        Args:
            radiation_class: radiation class used.
            parent : Parents class (wind object), to inherit global properties.
            r_0 : Initial radius in Rg units.
            z_0: Initial height in Rg units.
            rho_0 : Initial number density. Units of 1/cm^3.
            T : Initial streamline temperature in K.
            v_z_0 : Initial vertical velocity in units of cm/s.
            v_r_0 : Initial radial velocity in units of cm/s.
            dt : Timestep in units of Rg/c.
        """
        self.wind = wind
        self.radiation = radiation_class

        self.terminate_stalling = terminate_stalling
        # black hole and disc variables #
        self.a = np.array([0, 0, 0])  # / u.s**2
        self.T = T  # * u.K
        self.v_th = self.wind.thermal_velocity(self.T)
        self.rho_0 = rho_0
        self.rho = self.rho_0

        ## position variables ##
        ## coordinates of particle are (R, phi, z) ##
        ## all positions are in units of Rg, all velocities in units of c. ##
        self.dt = dt  # units of  Rg / c
        self.r = r_0
        self.phi = 0
        self.z = z_0
        self.x = [self.r, self.phi, self.z]
        self.d = np.sqrt(self.r**2 + self.z**2)
        self.t = 0  # in seconds
        self.r_0 = r_0
        self.z_0 = z_0
        self.v_r = v_r_0 / const.C
        self.v_r_hist = [self.v_r]
        self.v_phi = self.wind.v_kepler(r_0)
        self.l = self.v_phi * self.r  # initial angular momentum
        self.v_phi_0 = self.v_phi
        self.v_z_0 = v_z_0 / const.C
        self.v_z = self.v_z_0
        self.v = [self.v_r, self.v_phi, self.v_z]
        self.v_T_0 = np.sqrt(self.v_z ** 2 + self.v_r ** 2)
        self.v_T = self.v_T_0
        self.v_esc = self.wind.v_esc(self.d)
        self.v_esc_hist = [self.v_esc]
        self.dv_dr = 0
        self.dr_e = 0
        self.integral_epsrel = integral_epsrel
        self.integral_epsabs = integral_epsabs
        # this variable tracks whether the wind has reached the escape velocity
        self.escaped = False

        ## optical depths ##
        self.tau_dr = self.wind.tau_dr(self.rho)
        self.tau_dr_hist = [self.tau_dr]
        self.tau_dr_0 = self.tau_dr
        self.tau_dr_shielding = self.wind.tau_dr(self.wind.rho_shielding)

        self.tau_uv = self.radiation.optical_depth_uv(
            self.r, self.z, self.r_0, self.tau_dr, self.tau_dr_shielding)
        self.tau_x = self.radiation.optical_depth_x(
            self.r, self.z, self.r_0, self.tau_dr, self.tau_dr_shielding, self.wind.rho_shielding)

        self.tau_eff = 0
        self.fm = 0
        self.xi = self.radiation.ionization_parameter(
            self.r, self.z, self.tau_x, self.wind.rho_shielding)  # self.wind.Xi(self.d, self.z / self.r)

        # force related variables #
        self.Fgrav = []
        self.Frad = []
        self.iter = []

        #### history variables ####

        # position and velocities histories #
        self.x_hist = [self.x]
        self.d_hist = [self.d]
        self.t_hist = [0]
        self.r_hist = [r_0]
        self.phi_hist = [0]
        self.z_hist = [z_0]
        self.v_phi_hist = [self.v_phi]
        self.v_z_hist = [self.v_z]
        self.v_hist = [self.v]
        self.v_T_hist = [self.v_T_0]

        # radiation related histories #
        self.rho_hist = [self.rho]
        self.tau_dr_hist = [self.tau_dr]
        self.dvt_hist = [0]
        self.v2_hist = [0]
        self.dv_dr_hist = [0]
        self.dr_e_hist = [self.dr_e]
        self.tau_uv_hist = [self.tau_uv]
        self.tau_x_hist = [self.tau_x]
        self.tau_eff_hist = [0]
        self.taumax_hist = []
        self.fm_hist = [1]
        self.xi_hist = [self.xi]

        #force histories #
        self.a_hist = [self.a]

    ###############
    ## streaming ##
    ###############

    def update_density(self):
        """
        Updates the density of the streamline at the current position.
        If the density is below a critical value ( 10 cm/s ), then the density is left unchanged.

        Returns:
            rho: updated density at the current point.
        """
        #V_Z_CRIT = 0
        # if(self.v_z < V_Z_CRIT):
        #    self.rho_hist.append(self.rho)
        #    return self.rho

        radial = (self.d / self.r_0) ** (-2.)
        v_ratio = self.v_z_0 / self.v_T
        self.rho = self.rho_0 * radial * v_ratio
        # save to grid #
        self.rho_hist.append(self.rho)
        return self.rho

    def force_gravity(self):
        """
        Computes gravitational force at the current position. 

        Returns:
            grav: graviational force per unit mass in units of c^2 / Rg.
        """

        array = np.asarray([self.r / self.d, 0., self.z / self.d])
        grav = - 1. / (self.d**2) * array
        return grav

   ## kinematics ##

    def update_positions(self):
        """
        Updates position of streamline, by solving the equation of motion using a simple Euler integration.
        """
        # compute acceleration vector #
        fg = self.force_gravity()
        self.Fgrav.append(fg)
        fr = self.radiation.force_radiation(self.r,
                                            self.z,
                                            self.fm,
                                            self.tau_uv,
                                            epsrel = self.integral_epsrel,
                                            epsabs = self.integral_epsabs,
                                            return_error=False)
        self.a = fg
        if('gravityonly' in self.wind.modes):  # useful for debugging
            self.a += 0.
        else:
            self.a += fr

        self.a[0] += self.l**2 / self.r**3  # centrifugal term
        if "debug_mode" in self.wind.modes:
            self.a[0] = 1e-8 + 1e-5 * \
                (self.t/25000) + (self.t**2 / np.sqrt(25000))
            self.a[-1] = 1e-8 + 1e-5 * \
                (self.t/25000) + (self.t**2 / np.sqrt(25000))
        # update r #
        rnew = self.r + self.v_r * self.dt + 0.5 * self.a[0] * self.dt**2
        vrnew = self.v_r + self.a[0] * self.dt

        # update z #
        znew = self.z + self.v_z * self.dt + 0.5 * self.a[2] * self.dt**2
        vznew = self.v_z + self.a[2] * self.dt

        # update phi #
        phinew = self.phi + self.l / self.r**2 * self.dt
        vphinew = self.l / self.r

        # store new positions
        self.r = rnew
        self.v_r = vrnew
        self.z = znew
        self.v_z = vznew
        self.phi = phinew
        self.v_phi = vphinew
        self.x = [self.r, self.phi, self.z]
        self.v = [self.v_r, self.v_phi, self.v_z]

        # total velocity #
        self.v_T = np.sqrt(self.v_r ** 2 + self.v_z**2)
        if "dv_dr" in self.wind.modes:
        # compute dv_dr #
            v_T_2 = self.v_T_hist[-1]
            self.sobolev_delta_r = Decimal(np.linalg.norm(np.asarray(
                self.x)[[0, 2]] - np.asarray(self.x_hist[-1])[[0, 2]]))
            # use decimal to prevent round off error
            dv = min((Decimal(self.v_T) - Decimal(v_T_2)), self.wind.v_thermal)
            self.dv_dr = abs(float(dv) / float(self.sobolev_delta_r))
        else:
            self.dv_dr = np.sqrt(self.a[0]**2 + self.a[-1]**2) / self.v_T

        # finally update time #
        self.t = self.t + self.dt

        # append to history all the data for plotting and debugging#
        self.d = np.sqrt(self.r**2 + self.z**2)
        self.d_hist.append(self.d)
        self.r_hist.append(self.r)
        self.phi_hist.append(self.phi)
        self.z_hist.append(self.z)
        self.x_hist.append(self.x)
        self.v_r_hist.append(self.v_r)
        self.v_phi_hist.append(self.v_phi)
        self.v_z_hist.append(self.v_z)
        self.v_hist.append(self.v)
        self.Frad.append(fr)
        self.a_hist.append(self.a)
        self.dv_dr_hist.append(self.dv_dr)
        self.v_T_hist.append(self.v_T)
        self.t_hist.append(self.t)

    def update_radiation(self):
        """
        Updates all parameters related to the radiation field, given the new streamline position.
        """
        self.rho = self.update_density()
        self.tau_dr = self.wind.tau_dr(self.rho)
        self.tau_eff = self.radiation.sobolev_optical_depth(
            self.tau_dr, self.dv_dr)
        tau_eff_max = self.tau_dr * self.d  # abs(self.r - self.r_0)
        self.tau_uv = self.radiation.optical_depth_uv(
            self.r, self.z, self.r_0, self.tau_dr, self.tau_dr_shielding)
        self.tau_x = self.radiation.optical_depth_x(
            self.r, self.z, self.r_0, self.tau_dr, self.tau_dr_shielding, self.wind.rho_shielding)
        self.xi = self.radiation.ionization_parameter(
            self.r, self.z, self.tau_x, self.rho)
        self.fm = self.radiation.force_multiplier(self.tau_eff, self.xi)

        # append to history #
        self.tau_dr_hist.append(self.tau_dr)
        self.dr_e_hist.append(self.tau_eff/self.tau_dr)
        self.tau_eff_hist.append(self.tau_eff)
        self.tau_uv_hist.append(self.tau_uv)
        self.tau_x_hist.append(self.tau_x)
        self.xi_hist.append(self.xi)
        self.fm_hist.append(self.fm)

    def step(self):
        """
        Performs time step.
        """
        # update positions and velocities #
        self.update_positions()
        # update radiation field #
        self.update_radiation()

    def iterate(self, niter=5000):
        """
        Iterates the streamline

        Args:        
            niter : Number of iterations
        """
        print(' ', end='', flush=True)
        stalling_timer = 0

        for it in tqdm(range(0, niter)):
            self.step()
            # print(f"{self.t}")
            #print(f"a : {self.a} \n v_T: {self.v_T} \n dv_dr: {self.dv_dr} \n\n")
            v_esc = self.wind.v_esc(self.d)
            self.v_esc_hist.append(v_esc)
            # record number of iterations #
            self.it = it
            self.iter.append(it)

            if ((it == 99) or (it == 9999) or (it == 99999)):
                # update time step  at 100 iterations#
                self.dt = self.dt * 10.

            # termination condition for a failed wind #
            if(((self.z <= self.z_0) and (self.v_z < 0.0)) or ((self.z < np.max(self.z_hist)) and (self.v_z < 0.0) and self.terminate_stalling)):
                print("Failed wind! \n")
                break

            # record when streamline escapes #
            if((self.v_T > v_esc) and (not self.escaped)):
                self.escaped = True
                print("escape velocity reached.")
            a_t = np.sqrt(self.a[0]**2 + self.a[2]**2)

            #termination condition for an escaped wind #
            # if(self.escaped and a_t < 1e-8):
            #    print("Wind escaped")
            #    break
            if(self.d > 5000):
                print("out of grid \n")
                break

            # check line stalling
            # if self.v_z - self.v_th < 1e-5:
            #    self.r, self.phi, self.z = self.x_hist[-2]
            #    self.v_r, self.v_phi, self.v_z = self.v_hist[-2]
            #    self.x = self.x_hist[-2]
            #    self.v = self.v_hist[-2]
            #    stalling_timer += 1
            #    print("stalling...")
            #    #if stalling_timer == 1:
            #    #    self.dv_dr += 1e-5 * self.dv_dr
            #    #elif stalling_timer == 2:
            #    #    self.r += 1e-5 * self.r
            #    #elif stalling_timer == 3:
            #    #   self.v_r += 1e-5 * self.v_r
            #    #else:
            #    if stalling_timer == 4:
            #        print("Line stalled, terminating")
            #        break
            # else:
            #    stalling_timer = 0
