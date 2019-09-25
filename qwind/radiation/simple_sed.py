"""
This module handles the radiation transfer aspects of Qwind.
"""

import numpy as np
from scipy import integrate, interpolate, optimize

import qwind.constants as const
from qwind.pyagn import sed
from qwind import integration


class SimpleSED:
    """
    This class handles all the calculations involving the radiation field, i.e., radiative opacities, optical depths, radiation force, etc.
    Original implementation of RE2010.
    """

    def __init__(self, wind):
        self.wind = wind
        self.lines_r_min = self.wind.lines_r_min
        self.lines_r_max = self.wind.lines_r_max
        self.dr = (self.lines_r_max - self.lines_r_min) / (self.wind.nr - 1)
        self.r_init = self.lines_r_min + 0.5 * self.dr
        self.wind.tau_dr_0 = self.wind.tau_dr(self.wind.rho_shielding)
        self.sed_class = sed.SED(
            M=wind.M / const.M_SUN,
            mdot=wind.mdot,
            astar=wind.spin,
            number_bins_fractions=100,
        )
        self.uv_fraction = self.sed_class.uv_fraction
        self.xray_fraction = self.sed_class.xray_fraction
        self.xray_luminosity = self.wind.mdot * \
            self.wind.eddington_luminosity * self.xray_fraction
        self.r_x = self.ionization_radius()
        self.FORCE_RADIATION_CONSTANT = 3. * self.wind.mdot / \
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

    def optical_depth_uv(self, r, z, r_0, tau_dr, tau_dr_0):
        """
        UV optical depth.

        Args:
            r: radius in Rg units.
            z: height in Rg units.
            r_0: initial streamline radius.
            tau_dr: charact. optical depth
            tau_dr_0: initial charact. optical depth

        Returns:
            UV optical depth at point (r,z) 
        """
        delta_r_0 = abs(r_0 - self.r_init)
        delta_r = abs(r - r_0)
        distance = np.sqrt(r**2 + z**2)
        sec_theta = distance / r
        tau_uv = sec_theta * (delta_r_0 * tau_dr_0 + delta_r * tau_dr)
        tau_uv = min(tau_uv, 50)
        assert tau_uv >= 0, "UV optical depth cannot be negative!"
        return tau_uv

    def ionization_parameter(self, r, z, tau_x, rho_shielding):
        """
        Computes Ionization parameter.

        Args:
            r: radius in Rg units.
            z: height in Rg units.
            tau_x: X-Ray optical depth at the point (r,z)
            rho_shielding: density of the atmosphere that contributes
                           to shielding the X-Rays.

        Returns:
            ionization parameter.
        """
        distance_2 = r**2. + z**2.
        xi = self.xray_luminosity * \
            np.exp(-tau_x) / (rho_shielding *
                              distance_2 * self.wind.RG**2)  # / 8.2125
        assert xi > 0, "Ionization parameter cannot be negative!"
        xi += 1e-20  # to avoid overflow
        return xi

    def ionization_radius_kernel(self, rx):
        """
        Auxiliary function to compute ionization radius.

        Args:
            rx: Candidate for radius where material becomes non ionized.

        Returns:
            difference between current ion. parameter and target one.
        """
        tau_x = max(min(self.wind.tau_dr_0 * (rx - self.lines_r_min), 50), 0)
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
            r: radius in Rg

        Returns:
            opacity factor
        """
        if (r < self.r_x):
            return 1
        else:
            return 100

    def optical_depth_x(self, r, z, r_0, tau_dr, tau_dr_0, rho_shielding):
        """
        X-Ray optical depth at a distance d.

        Args:
            r: radius in Rg units.
            z: height in Rg units.
            r_0: initial streamline radius.
            tau_dr: charact. optical depth
            tau_dr_0: initial charact. optical depth
            rho_shielding: atmosphere density contributing to shield the X-Rays.

        Returns:
            X-Ray optical depth at the point (r,z)
        """
        tau_x_0 = self.r_x - self.r_init
        if (self.r_x < r_0):
            tau_x_0 += max(100 * (r_0 - self.r_x), 0)
        distance = np.sqrt(r ** 2 + z ** 2)
        sec_theta = distance / r
        delta_r = abs(r - r_0)
        tau_x = sec_theta * (tau_dr_0 * tau_x_0 + tau_dr *
                             self.opacity_x_r(r) * delta_r)
        tau_x = min(tau_x, 50)
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

    def sobolev_optical_depth(self, tau_dr, dv_dr):
        """
        Returns differential optical depth times a factor that compares thermal velocity with spatial velocity gradient.

        Args:
            tau_dr : Charact. optical depth.
            dv_dr : Velocity spatial gradient (dv is in c units, dr is in Rg units).
            T : Wind temperature.

        Returns:
            sobolev optical depth.
        """
        sobolev_length = self.wind.v_thermal / np.abs(dv_dr)
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
        #XI_UPPER_LIM = 1e4
        # if ( xi > XI_UPPER_LIM):
        #    return 0
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
        assert fm >= 0, "Force multiplier cannot be negative!"
        return fm

    def force_radiation(self, r, z, fm, tau_dr, tau_uv, return_error=False):
        """
        Computes the radiation force at the point (r,z)

        Args:
            r: radius in Rg units.
            z: height in Rg units.
            fm: force_multiplier
            tau_uv: UV optical depth.

        Returns:
            radiation force at the point (r,z) boosted by fm and attenuated by e^tau_uv.
        """

        i_aux = integration.qwind_integration_dblquad(
            r, z, self.wind.disk_r_min, self.wind.disk_r_max)
        error = i_aux[2:4]
        self.int_error_hist.append(error)

        self.int_hist.append(i_aux)
        abs_uv = np.exp(-tau_uv)
        constant = (1 + fm) * self.FORCE_RADIATION_CONSTANT
        #d = np.sqrt(r**2 + z**2)
        #cost = z / d
        #sint = r / d
        force = constant * abs_uv * np.asarray([i_aux[0],
                                                0.,
                                                i_aux[1]])
        if return_error:
            error = constant * np.array(error)
            return [force, error]
        return force
