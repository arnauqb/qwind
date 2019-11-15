"""
This module implements the streamline class, that initializes and evolves a streamline from the disc.
"""

import numpy as np
from scipy import integrate
from qwind import utils
from decimal import Decimal, DivisionByZero
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
            dt=4.096 / 10.
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
        self.r_previous = self.r
        self.phi = 0
        self.z = z_0
        self.z_previous = self.z
        self.x = [self.r, self.phi, self.z]
        self.d = np.sqrt(self.r**2 + self.z**2)
        self.t = 0  # in seconds
        self.r_0 = r_0
        self.z_0 = z_0
        self.v_r = v_r_0 / const.C
        self.v_r_previous = self.v_r
        self.v_r_0 = v_r_0 / const.C
        self.v_r_hist = [self.v_r]
        self.v_phi = self.wind.v_kepler(r_0)
        self.l = self.v_phi * self.r  # initial angular momentum
        self.v_phi_0 = self.v_phi
        self.v_z_0 = v_z_0 / const.C
        self.v_z = self.v_z_0
        self.v_z_previous = self.v_z
        self.v = [self.v_r, self.v_phi, self.v_z]
        self.v_T_0 = np.sqrt(self.v_z ** 2 + self.v_r ** 2)
        self.v_T = self.v_T_0
        self.v_T_previous = self.v_T
        self.v_esc = self.wind.v_esc(self.d)
        self.v_esc_hist = [self.v_esc]
        self.dv_dr = 0
        self.dr_e = 0
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
        self.dv_hist=[]
        self.delta_r_sob_hist =[]

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

    def update_density(self, r, z, v_T):
        """
        Updates the density of the streamline at the current position.
        If the density is below a critical value ( 10 cm/s ), then the density is left unchanged.

        Returns:
            rho: updated density at the current point.
        """
        d = np.sqrt(r**2 + z**2)
        radial = (d / self.r_0) ** (-2.)
        v_ratio = self.v_z_0 / v_T
        rho = self.rho_0 * radial * v_ratio
        return rho

    def compute_velocity_gradient(self, x_0, x_1, v_T_0, v_T_1):
        """
        Computes dv/dl
        """
        delta_l = Decimal(np.linalg.norm(x_1 - x_0))
        dv = Decimal(v_T_1) - Decimal(v_T_0)
        try:
            dv_dr = abs(float(dv / delta_l))
        except DivisionByZero:
            return 0
        return dv_dr

    def force_gravity(self, r, z):
        """
        Computes gravitational force at the current position. 

        Returns:
            grav: graviational force per unit mass in units of c^2 / Rg.
        """
        d = np.sqrt(r**2 + z**2)
        array = np.asarray([r / d, 0., z / d])
        grav = - 1. / (d**2) * array
        return grav

   ## kinematics ##
    
    def rk4_ydot(self, t, y):#, v_T, r_previous, z_previous, v_T_previous):

        r, z, v_r, v_z = y
        v_T = np.sqrt(r**2 + z**2)
        self.update_radiation(r, z, v_T, self.r_previous, self.z_previous, self.v_T_previous)
        fg = self.force_gravity(r,z)
        fr = self.radiation.force_radiation(r, z, self.fm, self.tau_uv)
        a = fg + fr
        a[0] += self.l**2 / r**3
        return [v_r, v_z, a[0], a[-1]]


    def initialize_ode_solver(self):
        t_0 = 0
        y_0 = [self.r_0, self.z_0, self.v_r_0, self.v_z_0]
        solver = integrate.RK45(fun=self.rk4_ydot, t0=t_0, y0=y_0, t_bound=50000000*self.wind.RG/const.C, max_step = 10000 * self.wind.RG/const.C)
        return solver
        

    def update_positions(self):
        """
        Updates position of streamline, by solving the equation of motion using a simple Euler integration.
        """
        # compute acceleration vector #
        fg = self.force_gravity(self.r, self.z)
        self.Fgrav.append(fg)
        fr = self.radiation.force_radiation(self.r,
                                            self.z,
                                            self.fm,
                                            self.tau_uv,
                                            return_error=False)
        self.a = fg
        if('gravityonly' in self.wind.modes):  # useful for debugging
            self.a += 0.
        else:
            self.a += fr

        self.a[0] += self.l**2 / self.r**3  # centrifugal term

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

        # compute dv_dr #
        try:
            v_T_2 = self.v_T_hist[-10]
        except:
            v_T_2 = self.v_T_hist[-1]
        x_1 = np.array(self.x, dtype=np.dtype(Decimal))[[0,2]]
        x_2 = np.array(self.x_hist[-1], dtype=np.dtype(Decimal))[[0,2]]
        self.sobolev_delta_r = Decimal(np.linalg.norm(x_1 - x_2))
        #self.sobolev_delta_r= Decimal(np.linalg.norm(np.asarray(
        #    self.x)[[0, 2]] - np.asarray(self.x_hist[-1])[[0, 2]]))
        # use decimal to prevent round off error
        dv = abs(Decimal(self.v_T) - Decimal(v_T_2))
        self.dv_hist.append(dv)
        self.dv_dr = float(dv / self.sobolev_delta_r)

        self.delta_r_sob_hist.append(self.sobolev_delta_r)
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

    def update_radiation(self, r, z, v_T, r_previous, z_previous, v_T_previous):
        """
        Updates all parameters related to the radiation field, given the new streamline position.
        """
        x_0 = np.array([r, z])
        x_1 = np.array([r_previous, z_previous])
        self.rho = self.update_density(r, z, v_T)
        self.tau_dr = self.wind.tau_dr(self.rho)
        self.dv_dr = self.compute_velocity_gradient(x_0, x_1, v_T, v_T_previous)
        self.tau_eff = self.radiation.sobolev_optical_depth(self.tau_dr, self.dv_dr)
        if self.tau_eff == np.inf:
            self.tau_eff = 1
        self.tau_uv = self.radiation.optical_depth_uv(r, z, self.r_0, self.tau_dr, self.tau_dr_shielding)
        self.tau_x = self.radiation.optical_depth_x(r, z, self.r_0, self.tau_dr, self.tau_dr_shielding, self.wind.rho_shielding)
        self.xi = self.radiation.ionization_parameter(r, z, self.tau_x, self.rho)
        self.fm = self.radiation.force_multiplier(self.tau_eff, self.xi)

        # append to history #
        #self.tau_dr_hist.append(self.tau_dr)
        #self.dr_e_hist.append(self.tau_eff/self.tau_dr)
        #self.tau_eff_hist.append(self.tau_eff)
        #self.tau_uv_hist.append(self.tau_uv)
        #self.tau_x_hist.append(self.tau_x)
        #self.xi_hist.append(self.xi)
        #self.fm_hist.append(self.fm)
        

    def step(self,r, z):
        """
        Performs time step.
        """
        # update positions and velocities #
        self.update_positions()
        # update radiation field #
        self.update_radiation(r, z)

    def iterate(self, niter=5000):
        """
        Iterates the streamline

        Args:        
            niter : Number of iterations
        """
        print(' ', end='', flush=True)
        self.solver = self.initialize_ode_solver()
        y_0 = [self.r_0, self.z_0, self.v_r_0, self.v_z_0]
        self.y_hist = [y_0]
        for it in tqdm(range(0, niter)):
            if "rk4" in self.wind.modes:
                self.solver.step()
                y = self.solver.y
                if np.sqrt(y[0]**2 + y[1]**2) > 10000:
                    print("line escaped.")
                    break
                self.y_hist.append(y)
            else:
                self.step(self.r, self.z)

            self.r_previous = self.r
            self.z_previous = self.z
            self.v_T_previous = self.v_T
            v_esc = self.wind.v_esc(self.d)
            self.v_esc_hist.append(v_esc)
            # record number of iterations #
            self.it = it
            self.iter.append(it)

            if ((it == 99) or (it == 9999) or (it == 99999)):
                # update time step  at 100 iterations#
                self.dt = self.dt * 10.

            # termination condition for a failed wind #
            if(((self.z <= self.z_0) and (self.v_z < 0.0)) or ((self.z <  np.max(self.z_hist)) and (self.v_z < 0.0))):
                print("Failed wind! \n")
                break

            # record when streamline escapes #
            if((self.v_T > v_esc) and (not self.escaped)):
                self.escaped = True
                print("escape velocity reached.")
            a_t = np.sqrt(self.a[0]**2 + self.a[2]**2)

            #termination condition for an escaped wind #
            #if(self.escaped and a_t < 1e-8):
            #    print("Wind escaped")
            #    break
            if(self.d > 100000):
                print("out of grid \n")
                break

            #check line stalling
            #if self.v_z - self.v_th < 1e-5:
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
            #else:
            #    stalling_timer = 0
