"""
This module implements the streamline class, that initializes and evolves a streamline from the disc.
"""

import numpy as np
from scipy import integrate, interpolate
from qwind import utils
from decimal import Decimal, DivisionByZero
from qwind import constants as const
import pickle
from assimulo.problem import Implicit_Problem
from assimulo.solvers import IDA, Radau5DAE, ODASSL
from assimulo.exception import TerminateSimulation
from numba import njit


# check backend to import appropiate progress bar #


class BackToDisk(Exception):
    pass


class Escape(Exception):
    pass


class OutOfGrid(Exception):
    pass


class Stalling(Exception):
    pass


@njit
def force_gravity(r, z):
    """
    Computes gravitational force at the current position. 
    
    Returns:
        grav: graviational force per unit mass in units of c^2 / R_g.
    """
    d = np.sqrt(r ** 2 + z ** 2)
    array = np.array([r / d, z / d])
    grav = -1.0 / (d ** 2) * array
    return grav


class streamline:
    """
    This class represents a streamline. It inherits from the wind class all the global properties of the accretion disc, black hole and atmosphere.
    """

    def __init__(
        self,
        radiation_class,
        wind,
        line_width=20,
        r_0=375.0,
        z_0=10.0,
        rho_0=2e8,
        T=None,
        v_z_0=1e7,
        v_r_0=0.0,
        dt=np.inf,  # 4.096 / 10.
        solver_rtol=1e-6,
        solver_atol=1e-3,
        integral_atol=0,
        integral_rtol=1e-3,
        t_max=10000,
        d_max=3e3,
        no_tau_z=False,
        no_tau_uv=False,
        es_only=False,
        max_iter=1000,
    ):
        """
        Args:
            parent : Parents class (wind object), to inherit global properties.
            r_0 : Initial radius in R_g units.
            z_0: Initial height in R_g units.
            rho_0 : Initial number density. Units of 1/cm^3.
            T : Initial streamline temperature in K.
            v_z_0 : Initial vertical velocity in units of cm/s.
            v_r_0 : Initial radial velocity in units of cm/s.
            dt : Timestep in units of R_g/c.
        """
        self.line_width = line_width
        self.wind = wind
        self.radiation = radiation_class
        if "debug_mode" in self.wind.modes:
            self.streamline_pos = np.loadtxt("streamline.txt")

        self.solver_rtol = solver_rtol
        self.solver_atol = solver_atol
        self.integral_atol = integral_atol
        self.integral_rtol = integral_rtol
        self.iter = 0
        self.max_iter = max_iter
        self.t_max = t_max * self.wind.R_g / const.C
        self.d_max = d_max
        self.no_tau_z = no_tau_z
        self.no_tau_uv = no_tau_uv
        self.es_only = es_only
        self.self_crossing_counter = 0

        # black hole and disc variables #
        self.T = T
        if self.T is None:
            self.T = self.wind.T  # * u.K
        self.v_th = self.wind.thermal_velocity(self.T)
        self.rho_0 = rho_0
        self.rho = self.rho_0
        ## position variables ##
        ## coordinates of particle are (R, phi, z) ##
        ## all positions are in units of R_g, all velocities in units of c. ##
        self.dt = dt  # units of  R_g / c
        self.r = r_0
        self.phi = 0
        self.z = z_0
        self.x = [self.r, self.phi, self.z]
        self.d = np.sqrt(self.r ** 2 + self.z ** 2)
        self.t = 0  # in seconds
        self.r_0 = r_0
        self.z_0 = z_0
        self.v_r = v_r_0 / const.C
        self.v_r_0 = v_r_0 / const.C
        self.v_phi = self.wind.v_kepler(r_0)
        self.l = self.v_phi * self.r  # initial angular momentum
        self.v_phi_0 = self.v_phi
        self.v_z_0 = v_z_0 / const.C
        self.v_z = self.v_z_0
        self.v = [self.v_r, self.v_phi, self.v_z]
        self.v_T_0 = np.sqrt(self.v_z ** 2 + self.v_r ** 2)
        self.v_T = self.v_T_0
        self.v_esc = self.wind.v_esc(self.d)
        # this variable tracks whether the wind has reached the escape velocity
        self.escaped = False

        ## optical depths ##
        self.tau_dr = self.wind.tau_dr(self.rho)
        self.tau_dr_0 = self.tau_dr
        self.tau_dr_shielding = self.wind.tau_dr(self.wind.rho_shielding)

        self.tau_uv = self.radiation.optical_depth_uv(
            self.r, self.z, self.r_0, self.tau_dr, self.tau_dr_shielding
        )
        self.tau_x = self.radiation.optical_depth_x(
            r=self.r,
            z=self.z,
            r_0=self.r_0,
            tau_dr=self.tau_dr,
            tau_dr_0=self.tau_dr_shielding,
            es_only=self.es_only,
        )

        self.xi = self.radiation.ionization_parameter(
            self.r, self.z, self.tau_x, self.wind.rho_shielding
        )  # self.wind.Xi(self.d, self.z / self.r)

        fgrav = force_gravity(self.r_0, self.z_0)
        frad = self.radiation.force_radiation(
            r=self.r_0,
            z=self.z_0,
            fm=0,
            tau_uv=self.tau_uv,
        )[[0, -1]]
        centrifugal_term = self.l ** 2 / self.r_0 ** 3
        a_r = fgrav[0] + frad[0] + centrifugal_term
        a_z = fgrav[-1] + frad[-1]
        self.a_0 = np.array([a_r, a_z])  # / u.s**2
        self.a = self.a_0
        self.a_T = np.sqrt(self.a[0] ** 2 + self.a[-1] ** 2)
        self.dv_dr_0 = self.a_T / self.v_T
        self.dv_dr = self.dv_dr_0
        self.tau_eff = self.radiation.sobolev_optical_depth(
            self.tau_dr, self.dv_dr, self.v_th
        )
        self.fm = self.radiation.force_multiplier(self.tau_eff, self.xi)
        frad = self.radiation.force_radiation(
            self.r_0,
            self.z_0,
            0,
            self.tau_uv,
        )[[0, -1]]
        # hists #
        # force related variables #
        self.fg_hist = [fgrav]
        self.fr_hist = [frad]

        #### history variables ####
        # position and velocities histories #
        self.d_hist = [self.d]
        self.t_hist = [0]
        self.r_hist = [self.r_0]
        self.phi_hist = [self.phi]
        self.z_hist = [self.z_0]
        self.v_r_hist = [0]
        self.v_phi_hist = [self.v_phi_0]
        self.v_z_hist = [self.v_z_0]
        self.v_T_hist = [self.v_T_0]
        self.v_th_hist = [self.v_th]

        # radiation related histories #
        self.rho_hist = [self.rho_0]
        self.tau_dr_hist = [self.tau_dr_0]
        self.dv_dr_hist = [self.dv_dr]
        self.tau_uv_hist = [self.tau_uv]
        self.tau_x_hist = [self.tau_x]
        self.tau_eff_hist = [self.tau_eff]
        self.fm_hist = [self.fm]
        self.xi_hist = [self.xi]
        self.T_hist = [self.T]

        # force histories #
        self.a_hist = [self.a]
        self.a_T_hist = [self.a_T]

        y_0 = [self.r_0, self.z_0, self.v_r_0, self.v_z_0]
        yd_0 = [self.v_r_0, self.v_z_0, self.a_0[0], self.a_0[1]]
        self.solver = self.initialize_ode_solver(y_0, yd_0, 0)
        self.stalling_exception_counter = 0

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
        d = np.sqrt(r ** 2 + z ** 2)
        radial = (d / self.r_0) ** (-2.0)
        v_ratio = self.v_z_0 / v_T
        rho = self.rho_0 * radial * v_ratio
        return rho

    ## kinematics ##

    def handle_result(self, solver, t, y, yd):
        """
        Event handling. This functions is called when Assimulo finds an event as
        specified by the event functions.
        """
        r, z, v_r, v_z = y
        solver.t_sol.extend([t])
        solver.y_sol.extend([y])
        solver.yd_sol.extend([yd])
        d = np.sqrt(r ** 2 + z ** 2)
        v_T = np.sqrt(v_r ** 2 + v_z ** 2)
        v_esc = self.wind.v_esc(d)
        if v_T > v_esc:
            self.escaped = True
        a_T = np.sqrt(solver.yd[2] ** 2 + solver.yd[3] ** 2)
        self.update_radiation(r, z, v_T, a_T, save_hist=True)
        if d > self.d_max:
            if self.escaped:
                raise Escape
            else:
                raise OutOfGrid
        if z < (self.z_0 - 0.01):
            raise BackToDisk

        if r < (self.r_0 - 0.01):
            self.self_crossing_counter += 1
            if self.self_crossing_counter == 3:
                raise BackToDisk

        if self.iter > self.max_iter:
            raise Stalling

    def residual(self, t, y, ydot):

        r, z, v_r, v_z = y
        r_dot, z_dot, v_r_dot, v_z_dot = ydot
        a_T = np.sqrt(v_r_dot ** 2 + v_z_dot ** 2)
        v_T = np.sqrt(r_dot ** 2 + z_dot ** 2)
        fg = force_gravity(r, z)
        self.update_radiation(r, z, v_T, a_T, save_hist=False)
        fr = self.radiation.force_radiation(
            r,
            z,
            self.fm,
            self.tau_uv,
            no_tau_z=self.no_tau_z,
            no_tau_uv=self.no_tau_uv,
        )[[0, 2]]
        centrifugal_term = self.l ** 2 / r ** 3
        a_r = fg[0] + centrifugal_term  # + fr[0]
        a_z = fg[-1]  # + fr[-1]
        if "gravityonly" not in self.wind.modes:
            a_r += fr[0]
            a_z += fr[-1]
        residue = np.zeros(4)
        residue[0] = r_dot - v_r
        residue[1] = z_dot - v_z
        residue[2] = v_r_dot - a_r
        residue[3] = v_z_dot - a_z
        return residue

    def initialize_ode_solver(self, y_0, yd_0, t_0):
        model = Implicit_Problem(self.residual, y_0, yd_0, t_0)
        model.handle_result = self.handle_result
        solver = IDA(model)
        solver.rtol = self.solver_rtol
        solver.atol = self.solver_atol  # * np.array([100, 10, 1e-4, 1e-4])
        solver.inith = 0.1  # self.wind.R_g / const.C
        solver.maxh = self.dt * self.wind.R_g / const.C
        solver.report_continuously = True
        solver.display_progress = False
        solver.verbosity = 50  # 50 = quiet
        solver.num_threads = 3

        # solver.display_progress = True
        return solver

    def save_hist(self):
        r, z, v_r, v_z = self.solver.y
        a_T = np.sqrt(self.solver.yd[2] ** 2 + self.solver.yd[3] ** 2)
        self.iter += 1
        self.a_T_hist.append(a_T)
        self.t_hist.append(self.solver.t)
        self.r_hist.append(r)
        self.z_hist.append(z)
        self.v_r_hist.append(v_r)
        self.v_z_hist.append(v_z)
        v_T = np.sqrt(v_r ** 2 + v_z ** 2)
        self.v_T_hist.append(v_T)
        frad = self.radiation.force_radiation(
            r,
            z,
            self.fm,
            self.tau_uv,
        )[[0, -1]]
        self.fr_hist.append(frad)
        fgrav = force_gravity(r, z)
        self.fg_hist.append(fgrav)
        self.rho_hist.append(self.rho)
        self.tau_dr_hist.append(self.tau_dr)
        self.tau_eff_hist.append(self.tau_eff)
        self.tau_x_hist.append(self.tau_x)
        self.tau_uv_hist.append(self.tau_uv)
        self.T_hist.append(self.T)
        self.xi_hist.append(self.xi)
        self.fm_hist.append(self.fm)
        self.dv_dr_hist.append(self.dv_dr)

    def update_radiation(self, r, z, v_T, a_T, save_hist=False):
        """
        Updates all parameters related to the radiation field, given the new streamline position.
        """
        self.rho = self.update_density(r, z, v_T)
        self.tau_dr = self.wind.tau_dr(self.rho)
        self.dv_dr = a_T / v_T
        self.tau_uv = self.radiation.optical_depth_uv(
            r=r, z=z, r_0=self.r_0, tau_dr=self.tau_dr, tau_dr_0=self.tau_dr_shielding
        )
        self.tau_x = self.radiation.optical_depth_x(
            r=r,
            z=z,
            r_0=self.r_0,
            tau_dr=self.tau_dr,
            tau_dr_0=self.tau_dr_shielding,
            es_only=self.es_only,
        )
        self.xi = self.radiation.ionization_parameter(
            r=r, z=z, tau_x=self.tau_x, rho=self.rho
        )
        self.v_th = self.wind.thermal_velocity(self.T)
        self.tau_eff = self.radiation.sobolev_optical_depth(
            self.tau_dr, self.dv_dr, self.v_th
        )
        self.fm = self.radiation.force_multiplier(self.tau_eff, self.xi)
        if save_hist:
            self.save_hist()

    def iterate(self, niter=5000):
        """
        Iterates the streamline

        Args:        
            niter : Number of iterations
        """
        print(" ", end="", flush=True)
        self.first_iter = False
        self.stalling_timer = 0
        self.stalling = False
        self.solver.initialize()
        try:
            self.solver.simulate(self.t_max)
        except Escape:
            self.solver.finalize()
            self.escaping_angle = np.arctan(self.z_hist[-1] / self.r_hist[-1])
            self.terminal_velocity = np.sqrt(
                self.v_r_hist[-1] ** 2 + self.v_z_hist[-1] ** 2
            )
            print("\U0001F4A8", end=" ")
            pass
        except OutOfGrid:
            self.solver.finalize()
            print("\U00002753", end=" ")
        except BackToDisk:
            self.escaped = False
            self.solver.finalize()
            print("\U0001F4A5", end=" ")
            pass
        except Stalling:
            print("\U00002753", end=" ")
            pass
