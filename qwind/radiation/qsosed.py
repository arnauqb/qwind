"""
This module handles the radiation transfer aspects of Qwind.
"""
import numpy as np
from numba import njit
from scipy import interpolate, optimize
from qwind.c_functions import integration
import qwind.constants as const
from qsosed import sed
from qwind import grid
from qwind.c_functions.wrapper import tau_uv as tau_uv_c


N_R = 1000
N_Z = 1001
N_DISK = 100

@njit
def cooling_g_compton(T, xi, Tx=1e8):
    g_compton = 8.9e-36 * xi * (Tx - 4 * T)
    return g_compton

@njit
def cooling_g_photoi(T, xi, Tx=1e8):
    g_photo = 1.5e-21 * xi**(1./4.) * T**(0.5) * ( 1. - T/Tx)
    return g_photo

@njit
def cooling_brem(T, xi):
    lbl = 3.3e-27 * T**(0.5) + 1.7e-18 / (xi * T**(0.5)) * np.exp(-1.3e5 / T) + 1e-24
    return lbl

@njit
def cooling_total(T, n, xi, Tx=1e8):
    total = n**2 * (cooling_g_compton(T, xi, Tx) + cooling_g_photoi(T, xi, Tx) - cooling_brem(T, xi))
    return total

class Radiation:
    """
    This class handles all the calculations involving the radiation field, i.e., radiative opacities, optical depths, radiation force, etc.
    Original implementation of RE2010.
    """

    def __init__(self, wind):
        self.wind = wind
        self.qsosed = sed.SED(M=self.wind.M / const.M_SUN,
                mdot=self.wind.mdot,
                number_bins_fractions=N_DISK)
        if "qsosed_geometry" in self.wind.modes:
            self.uv_radial_flux_fraction = self.qsosed.compute_uv_fractions(return_all = False)
            self.xray_fraction = self.qsosed.xray_fraction
            self.uv_fraction = self.qsosed.uv_fraction
            r_range_aux = np.linspace(self.qsosed.warm_radius, self.qsosed.gravity_radius, N_DISK)
            self.FORCE_RADIATION_CONSTANT = 3. / (8. * np.pi * self.wind.eta)
            self.wind.disk_r_min = self.qsosed.warm_radius
            self.wind.disk_r_max = r_range_aux[np.argwhere(self.uv_radial_flux_fraction < 0.05)[0][0]]
            self.wind.lines_r_min = self.wind.disk_r_min
            self.wind.lines_r_max = self.wind.disk_r_max
        else:
            self.xray_fraction = self.wind.f_x 
            if self.wind.f_uv is None:
                self.uv_fraction = 1 - self.wind.f_x
            else:
                self.uv_fraction = self.wind.f_uv
            self.uv_radial_flux_fraction = np.ones(N_DISK)
            self.FORCE_RADIATION_CONSTANT = 3. / (8. * np.pi * self.wind.eta) * self.uv_fraction
        self.mdot_0 = self.wind.mdot
        self.dr = (self.wind.lines_r_max - self.wind.lines_r_min) / (self.wind.nr - 1)
        self.wind.tau_dr_0 = self.wind.tau_dr(self.wind.rho_shielding)
        self.xray_luminosity = self.wind.mdot * self.wind.eddington_luminosity * self.xray_fraction
        if "old_taus" in self.wind.modes:
            self.r_x = self.ionization_radius()
            (8. * np.pi * self.wind.eta) * self.uv_fraction
        self.int_hist = []
        self.int_error_hist = []

        # interpolation values for force multiplier #
        K_INTERP_XI_VALUES = [-4, -3, -2.26, -2.00, -1.50, -1.00,
                              -0.42, 0.00, 0.22, 0.50, 1.0,
                              1.5, 1.8, 2.0, 2.18, 2.39,
                              2.76, 3.0, 3.29, 3.51, 3.68, 4.0]
        K_INTERP_K_VALUES = [0.411, 0.411, 0.400, 0.395, 0.363, 0.300,
                             0.200, 0.132, 0.100, 0.068, 0.042,
                             0.034, 0.033, 0.021, 0.013, 0.048,
                             0.046, 0.042, 0.044, 0.045, 0.032,
                             0.013]
        ETAMAX_INTERP_XI_VALUES = [-3, -2.5, -2.00, -1.50, -1.00,
                                   -0.5, -0.23, 0.0, 0.32, 0.50,
                                   1.0, 1.18, 1.50, 1.68, 2.0,
                                   2.02, 2.16, 2.25, 2.39, 2.79,
                                   3.0, 3.32, 3.50, 3.75, 4.00]

        ETAMAX_INTERP_ETAMAX_VALUES = [6.95, 6.95, 6.98, 7.05, 7.26,
                                       7.56, 7.84, 8.00, 8.55, 8.95,
                                       8.47, 8.00, 6.84, 6.00, 4.32,
                                       4.00, 3.05, 2.74, 3.00, 3.10,
                                       2.73, 2.00, 1.58, 1.20, 0.78]
        self.k_interpolator = interpolate.interp1d(
            K_INTERP_XI_VALUES,
            K_INTERP_K_VALUES,
            bounds_error=False,
            fill_value=(K_INTERP_K_VALUES[0], K_INTERP_K_VALUES[-1]),
            kind='cubic')  # important! xi is log here
        self.log_etamax_interpolator = interpolate.interp1d(
            ETAMAX_INTERP_XI_VALUES,
            ETAMAX_INTERP_ETAMAX_VALUES,
            bounds_error=False,
            fill_value=(
                ETAMAX_INTERP_ETAMAX_VALUES[0],
                ETAMAX_INTERP_ETAMAX_VALUES[-1]),
            kind='cubic')  # important! xi is log here
        # new stuff
        self.grid = grid.Grid(self.wind, n_r=N_R, n_z=N_Z, n_disk=N_DISK)
        #self.grid.update_all(init=True)
        #self.integrator = integration.Integrator(self.wind.R_g,
        #        self.grid.density_grid,
        #        self.grid.grid_r_range,
        #        self.grid.grid_z_range,
        #        self.grid.mdot_grid,
        #        self.grid.uv_fraction_grid,
        #        self.grid.grid_disk_range,
        #        epsrel=1e-3,
        #        epsabs=0)

    def compute_mass_accretion_rate_grid(self, lines):
        """
        Returns mass accretion rate at radius r, taking into account the escaped wind.
        Also updates it from the grid file.
        """
        new_mdot_list = self.grid.mdot_grid.copy()
        lines_escaped = np.array(lines)[np.where([line.escaped for line in lines])[0]]
        accumulated_wind = 0.
        for line in lines_escaped[::-1]:
            r_0 = line.r_0
            width = line.line_width
            r_f_arg = np.searchsorted(self.grid.grid_disk_range, r_0 + width/2.)
            mdot_w = self.wind.compute_line_mass_loss(line) 
            mdot_w_normalized = mdot_w / self.qsosed.mass_accretion_rate 
            accumulated_wind += mdot_w_normalized
            new_mdot_list[0:r_f_arg] = self.mdot_0 - accumulated_wind 
        new_mdot_list = np.maximum(new_mdot_list, 0.)
        self.grid.mdot_grid = np.array(new_mdot_list)

    def optical_depth_uv(self, r, z, r_0, tau_dr, tau_dr_0):
        """
        UV optical depth.

        Args:
            r: radius in R_g units.
            z: height in R_g units.
            r_0: initial streamline radius.
            tau_dr: charact. optical depth
            tau_dr_0: initial charact. optical depth

        Returns:
            UV optical depth at point (r,z) 
        """
        if "old_taus" not in self.wind.modes:
            tau = tau_uv_c(r, z, self.grid.density_grid) * const.SIGMA_T * self.wind.R_g
            return tau
        else: 
            delta_r_0 = abs(r_0 - self.wind.r_init)
            delta_r = abs(r - r_0 - self.dr/2)
            distance = np.sqrt(r**2 + z**2)
            sec_theta = distance / r
            tau_uv = sec_theta * (delta_r_0 * tau_dr_0 + delta_r * tau_dr)
            tau_uv = min(tau_uv, 50)
            if tau_uv < 0:
                print("warning")
            tau_uv = max(tau_uv,0)
            try:
                assert tau_uv >= 0, "UV optical depth cannot be negative!"
            except AssertionError:
                print(f"r: {r} \n z : {z} \n r_0 : {r_0}\n tau_dr: {tau_dr} \n tau_dr_0: {tau_dr_0} \n\n")
                raise AssertionError 
            return tau_uv

    def compute_equilibrium_temperature(self, n, xi, Tx=1e8):
        try:
            root = optimize.root_scalar(cooling_total,
                    args=(n, xi, Tx),
                    bracket=(10, 1e9),
                    method='bisect') 
        except ValueError:
        #    print(xi)
            xi *= 5
            root = self.compute_equilibrium_temperature(n, xi, Tx)
            return root
        assert root.converged
        return root.root

    def compute_temperature(self, n, R, xi, Tx=1e8):
        if "compute_temperature" in self.wind.modes:
            xi = max(xi, 1e-10)
            eq_temp = self.compute_equilibrium_temperature(n, xi, Tx)
            disk_temp = self.qsosed.disk_nt_temperature4(R)**(1./4.)
            return max(disk_temp, eq_temp)
        else:
            return self.wind.T


    def ionization_parameter(self, r, z, tau_x, rho):
        """
        Computes Ionization parameter.

        Args:
            r: radius in R_g units.
            z: height in R_g units.
            tau_x: X-Ray optical depth at the point (r,z)
            rho_shielding: density of the atmosphere that contributes
                           to shielding the X-Rays.

        Returns:
            ionization parameter.
        """
        if "old_taus" in self.wind.modes:
            DENSITY_FLOOR = 1e2
            if r < self.wind.r_init:
                rho = DENSITY_FLOOR
                tau_x = tau_x / rho * DENSITY_FLOOR
            distance_2 = r**2. + z**2.
            xi = self.xray_luminosity * np.exp(-tau_x) \
                / (rho * distance_2 * self.wind.R_g**2)
            assert xi >= 0, "Ionization parameter cannot be negative!"
            xi += 1e-20  # to avoid overflow
            return xi
        else:
            tau_x = self.grid.tau_x_grid.get_value(r,z)
            d2 = r**2. + z**2.
            xi = self.xray_luminosity * np.exp(-tau_x) /  ( rho * d2 * self.wind.R_g**2)
            return xi + 1e-15 # to avoid roundoff issues

    def ionization_radius_kernel(self, rx):
        """
        Auxiliary function to compute ionization radius.

        Args:
            rx: Candidate for radius where material becomes non ionized.

        Returns:
            difference between current ion. parameter and target one.
        """
        tau_x =  max(min(self.wind.tau_dr_0 * (rx - self.wind.r_init), 50), 0)
        xi = self.ionization_parameter(rx, 0, tau_x, self.wind.rho_shielding)
        ionization_difference = const.IONIZATION_PARAMETER_CRITICAL - xi
        return ionization_difference

    def ionization_radius(self):
        """
        Computes the disc radius at which xi = xi_0 using the bisect method.
        """
        try:
            r_x = optimize.root_scalar(
                self.ionization_radius_kernel,
                bracket=[self.wind.disk_r_min, self.wind.disk_r_max])
        except:
            print("ionization radius outside the disc.")
            if(self.ionization_radius_kernel(self.wind.disk_r_min) > 0):
                print("ionization radius is below r_min, nothing is ionized.")
                r_x = self.wind.disk_r_min
                return r_x
            else:
                print(
                    "ionization radius is very large, atmosphere is completely ionized.")
                r_x = self.wind.disk_r_max
                return r_x
        assert r_x.converged is True
        r_x = r_x.root
        assert r_x > 0, "Got non physical ionization radius!"
        return r_x

    def opacity_x_r(self, r):
        """
        X-Ray opacity factor (respect to just Thomson cross-section).

        Args:
            r: radius in R_g

        Returns:
            opacity factor
        """
        if (r < self.r_x):
            return 1
        else:
            return 100

    def optical_depth_x(self, r, z, r_0, tau_dr, tau_dr_0, rho_shielding, es_only=False):
        """
        X-Ray optical depth at a distance d.

        Args:
            r: radius in R_g units.
            z: height in R_g units.
            r_0: initial streamline radius.
            tau_dr: charact. optical depth
            tau_dr_0: initial charact. optical depth
            rho_shielding: atmosphere density contributing to shield the X-Rays.

        Returns:
            X-Ray optical depth at the point (r,z)
        """
        if "old_taus" not in self.wind.modes:
            tau_x = self.grid.tau_x_grid.get_value(r,z)
            return tau_x
        else:
            if es_only:
                delta_r_0 = max(r_0 - self.wind.r_init - self.dr/2.,0)
                distance = np.sqrt(r ** 2 + z ** 2)
                sec_theta = distance / r
                delta_r = abs(r - r_0 - self.dr / 2)
                tau_x = sec_theta * (tau_dr_0 * delta_r_0 + tau_dr * delta_r)
                assert tau_x >= 0
                tau_x = min(tau_x,50)
                return tau_x

            tau_x_0 = max(self.r_x - self.wind.r_init - self.dr/2.,0)
            tau_x_0 += max(100 * (r_0 - self.dr/2. - self.r_x), 0)
            distance = np.sqrt(r ** 2 + z ** 2)
            sec_theta = distance / r
            delta_r = abs(r - r_0 - self.dr / 2)
            tau_x = sec_theta * (tau_dr_0 * tau_x_0 + tau_dr *
                                 self.opacity_x_r(r) * delta_r)
            tau_x = min(tau_x, 50)
            if tau_x < 0:
                print("warning")
            tau_x = max(tau_x,0)

            assert tau_x >= 0, "X-Ray optical depth cannot be negative!"
            return tau_x

    def force_multiplier_k(self, xi):
        """
        Auxiliary function required for computing force multiplier.

        Args: 
            xi: Ionisation Parameter.

        Returns:
            Factor k in the force multiplier formula.
        """
        if "analytical_fm" in self.wind.modes:
            k = 0.03 + 0.385 * np.exp(-1.4 * xi**(0.6))
        else:
            k = self.k_interpolator(np.log10(xi))
        assert k >= 0, "k cannot be negative!"
        return k

    def force_multiplier_eta_max(self, xi):
        """
        Auxiliary function required for computing force multiplier.

        Args:
            xi: Ionisation Parameter.

        Returns:
            Factor eta_max in the force multiplier formula.
        """
        if "analytical_fm" in self.wind.modes:
            if(np.log10(xi) < 0.5):
                aux = 6.9 * np.exp(0.16 * xi**(0.4))
                eta_max = 10**aux
            else:
                aux = 9.1 * np.exp(-7.96e-3 * xi)
                eta_max = 10**aux
        else:
            eta_max = 10**(self.log_etamax_interpolator(np.log10(xi)))
        assert eta_max >= 0, "Eta Max cannot be negative!"
        return eta_max

    def sobolev_optical_depth(self, tau_dr, dv_dr, v_thermal):
        """
        Returns differential optical depth times a factor that compares thermal velocity with spatial velocity gradient.

        Args:
            tau_dr : Charact. optical depth.
            dv_dr : Velocity spatial gradient (dv is in c units, dr is in R_g units).
            T : Wind temperature.

        Returns:
            sobolev optical depth.
        """
        sobolev_length = v_thermal / np.abs(dv_dr)
        sobolev_optical_depth = tau_dr * sobolev_length
        assert sobolev_optical_depth >= 0, "Sobolev optical depth cannot be negative!"
        return sobolev_optical_depth

    def force_multiplier(self, t, xi):
        """
        Computes the force multiplier, following Stevens & Kallman (1990).

        Args:
            t: Sobolev optical depth.
            xi : Ionisation Parameter.

        Returns:
            fm : force multiplier.
        """
        TAU_MAX_TOL = 0.001
        k = self.force_multiplier_k(xi)
        eta_max = self.force_multiplier_eta_max(xi)
        tau_max = t * eta_max
        alpha = 0.6
        if (tau_max < TAU_MAX_TOL):
            aux = (1. - alpha) * (tau_max ** alpha)
        else:
            aux = ((1. + tau_max)**(1. - alpha) - 1.) / \
                ((tau_max) ** (1. - alpha))
        fm = k * t**(-alpha) * aux
        #assert fm >= 0, "Force multiplier cannot be negative!"
        fm = max(0,fm)
        return fm

    def force_radiation(self, r, z, fm, tau_uv, return_error=False, no_tau_z=False, no_tau_uv=False, **kwargs):
        """
        Computes the radiation force at the point (r,z)

        Args:
            r: radius in R_g units.
            z: height in R_g units.
            fm: force_multiplier
            tau_uv: UV optical depth.

        Returns:
            radiation force at the point (r,z) boosted by fm and attenuated by e^tau_uv.
        """
        if "uv_interp" in self.wind.modes:
            if self.wind.first_iter:
                i_aux = self.integrator.integrate_notau(r, z)
            else:
                i_aux = self.integrator.integrate(r,z)
        else:
            i_aux = self.integrator.integrate_notau(r, z)
            if "no_tau_uv" in self.wind.modes:
                abs_uv = 1
            if "no_tau_z" in self.wind.modes:
                d = np.sqrt(r**2 + z**2)
                sin_theta = z / d
                cos_theta = r / d
                tau_uv = tau_uv * np.array([cos_theta, sin_theta])
                abs_uv = np.exp(-tau_uv)
            else:
                abs_uv = np.exp(-tau_uv)
            i_aux = np.array(i_aux) * abs_uv
        constant = (1 + fm) * self.FORCE_RADIATION_CONSTANT
        force = constant * np.asarray([i_aux[0], 0.,i_aux[1]])
        return force
        #if('old_integral' in self.wind.modes):
        #    i_aux = integration.qwind_old_integration(r, z)
        #if('non_relativistic' in self.wind.modes):
        #    i_aux = integration.qwind_integration_dblquad(
        #        r, z, self.wind.disk_r_min, self.wind.disk_r_max, **kwargs)
        #else:
        #    i_aux = self.integrator.integrate_notau(r,z)

        #self.int_hist.append(i_aux)
        #if no_tau_z == True:
        #    d = np.sqrt(r**2 + z**2)
        #    sin_theta = z / d
        #    cos_theta = r / d
        #    tau_uv = tau_uv * np.array([cos_theta, 0, sin_theta])
        #    abs_uv = np.exp(-tau_uv)
        #elif no_tau_uv == True:
        #    abs_uv = 1
        #else:
        #    abs_uv = np.exp(-tau_uv)
        #constant = (1 + fm) * self.FORCE_RADIATION_CONSTANT
        #force = abs_uv * constant  * np.asarray([i_aux[0],
        #                                         0.,
        #                                         i_aux[1],
        #                                         ])
        #assert force[2] >= 0
        #if return_error:
        #    error = constant * np.array(error)
        #    return [force, error]
        #return force

    
