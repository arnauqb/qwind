"""
This module handles the radiation transfer aspects of Qwind.
"""

import numpy as np
from scipy import integrate, interpolate, optimize
import qwind.constants as const
from qwind.integration.integration import IntegratorSimplesed as integrator


class Radiation:
    """
    This class handles all the calculations involving the radiation field, i.e., radiative opacities, optical depths, radiation force, etc.
    Original implementation of RE2010.
    """

    def __init__(self, wind):
        self.wind = wind
        self.xray_fraction = self.wind.f_x
        self.uv_fraction = 1 - self.wind.f_x
        self.dr = (self.wind.lines_r_max - self.wind.lines_r_min) / (self.wind.nr - 1)
        self.wind.r_init = self.wind.lines_r_min + self.dr / 2.0
        self.wind.tau_dr_0 = self.wind.tau_dr(self.wind.rho_shielding)
        self.xray_luminosity = (
            self.wind.mdot * self.wind.eddington_luminosity * self.xray_fraction
        )
        self.r_x = self.ionization_radius()
        self.FORCE_RADIATION_CONSTANT = (
            3.0 * self.wind.mdot / (8.0 * np.pi * self.wind.eta) * self.uv_fraction
        )
        self.int_hist = []
        self.int_error_hist = []

        # interpolation values for force multiplier #
        K_INTERP_XI_VALUES = [
            -4,
            -3,
            -2.26,
            -2.00,
            -1.50,
            -1.00,
            -0.42,
            0.00,
            0.22,
            0.50,
            1.0,
            1.5,
            1.8,
            2.0,
            2.18,
            2.39,
            2.76,
            3.0,
            3.29,
            3.51,
            3.68,
            4.0,
        ]
        K_INTERP_K_VALUES = [
            0.411,
            0.411,
            0.400,
            0.395,
            0.363,
            0.300,
            0.200,
            0.132,
            0.100,
            0.068,
            0.042,
            0.034,
            0.033,
            0.021,
            0.013,
            0.048,
            0.046,
            0.042,
            0.044,
            0.045,
            0.032,
            0.013,
        ]
        ETAMAX_INTERP_XI_VALUES = [
            -3,
            -2.5,
            -2.00,
            -1.50,
            -1.00,
            -0.5,
            -0.23,
            0.0,
            0.32,
            0.50,
            1.0,
            1.18,
            1.50,
            1.68,
            2.0,
            2.02,
            2.16,
            2.25,
            2.39,
            2.79,
            3.0,
            3.32,
            3.50,
            3.75,
            4.00,
        ]

        ETAMAX_INTERP_ETAMAX_VALUES = [
            6.95,
            6.95,
            6.98,
            7.05,
            7.26,
            7.56,
            7.84,
            8.00,
            8.55,
            8.95,
            8.47,
            8.00,
            6.84,
            6.00,
            4.32,
            4.00,
            3.05,
            2.74,
            3.00,
            3.10,
            2.73,
            2.00,
            1.58,
            1.20,
            0.78,
        ]
        self.k_interpolator = interpolate.interp1d(
            K_INTERP_XI_VALUES,
            K_INTERP_K_VALUES,
            bounds_error=False,
            fill_value=(K_INTERP_K_VALUES[0], K_INTERP_K_VALUES[-1]),
            kind="cubic",
        )  # important! xi is log here
        self.log_etamax_interpolator = interpolate.interp1d(
            ETAMAX_INTERP_XI_VALUES,
            ETAMAX_INTERP_ETAMAX_VALUES,
            bounds_error=False,
            fill_value=(
                ETAMAX_INTERP_ETAMAX_VALUES[0],
                ETAMAX_INTERP_ETAMAX_VALUES[-1],
            ),
            kind="cubic",
        )  # important! xi is log here
        self.integrator = integrator(
            self.wind.R_g,
            self.wind.disk_r_min,
            self.wind.disk_r_max,
            0,
            self.wind.epsrel,
            self.wind.spin,
            self.wind.disk_r_min,
        )

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
        delta_r_0 = abs(r_0 - self.wind.lines_r_range[0])
        delta_r = abs(r - r_0 - self.dr/2)
        distance = np.sqrt(r**2 + z**2)
        sec_theta = distance / r
        tau_uv = sec_theta * (delta_r_0 * tau_dr_0 + delta_r * tau_dr)
        tau_uv = min(tau_uv, 50)
        return tau_uv

    def ionization_parameter(self, r, z, tau_x, rho):
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
        d = np.sqrt(r**2 + z**2) * self.wind.R_g
        xi = self.xray_luminosity * np.exp(-tau_x) / (rho * d**2)
        return max(xi, 1e-20)

    def ionization_radius_kernel(self, rx_log):
        """
        Auxiliary function to compute ionization radius.

        Args:
            rx: Candidate for radius where material becomes non ionized.

        Returns:
            difference between current ion. parameter and target one.
        """
        rx = np.exp(rx_log)
        if rx < self.wind.r_init:
            return np.log(
                self.xray_luminosity
                / (
                    const.IONIZATION_PARAMETER_CRITICAL
                    * 1e2
                    * (rx * self.wind.R_g) ** 2
                )  # 1e2 = vacuum
            )
        else:
            return (
                np.log(
                    self.xray_luminosity
                    / (
                        const.IONIZATION_PARAMETER_CRITICAL
                        * self.wind.rho_shielding
                        * (rx * self.wind.R_g) ** 2
                    )
                )
                - self.wind.rho_shielding
                * (rx - self.wind.r_init)
                * const.SIGMA_T
                * self.wind.R_g
            )

    def ionization_radius(self):
        """
        Computes the disc radius at which xi = xi_0 using the bisect method.
        """
        r_x = optimize.root_scalar(
            self.ionization_radius_kernel, bracket=(-40, 40), xtol=1e-4, rtol=1e-4
        )
        return np.exp(r_x.root)

    def opacity_x_r(self, r):
        """
        X-Ray opacity factor (respect to just Thomson cross-section).

        Args:
            r: radius in Rg

        Returns:
            opacity factor
        """
        if r < self.r_x:
            return 1
        else:
            return 100

    def optical_depth_x(
        self, r, z, r_0, tau_dr, tau_dr_0, es_only=False
    ):
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
        if es_only:
            delta_r_0 = max(r_0 - self.wind.r_init - self.dr/2.,0)
            distance = np.sqrt(r ** 2 + z ** 2)
            sec_theta = distance / r
            delta_r = abs(r - r_0 - self.dr / 2)
            tau_x = sec_theta * (tau_dr_0 * delta_r_0 + tau_dr * delta_r)
            tau_x = min(tau_x, 50)
            tau_x = max(tau_x,1e-20)
            return tau_x

        tau_x_0 = max(self.r_x - self.wind.r_init - self.dr/2.,0)
        tau_x_0 += max(100 * (r_0 - self.dr/2. - self.r_x), 0)
        distance = np.sqrt(r ** 2 + z ** 2)
        sec_theta = distance / r
        delta_r = abs(r - r_0 - self.dr / 2)
        tau_x = sec_theta * (tau_dr_0 * tau_x_0 + tau_dr *
                             self.opacity_x_r(r) * delta_r)
        tau_x = min(tau_x, 50)
        tau_x = max(tau_x,1e-20)
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
            k = 0.03 + 0.385 * np.exp(-1.4 * xi ** (0.6))
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
            if np.log10(xi) < 0.5:
                aux = 6.9 * np.exp(0.16 * xi ** (0.4))
                eta_max = 10 ** aux
            else:
                aux = 9.1 * np.exp(-7.96e-3 * xi)
                eta_max = 10 ** aux
        else:
            eta_max = 10 ** (self.log_etamax_interpolator(np.log10(xi)))
        assert eta_max >= 0, "Eta Max cannot be negative!"
        return eta_max

    def sobolev_optical_depth(self, tau_dr, dv_dr, v_thermal):
        """
        Returns differential optical depth times a factor that compares thermal velocity with spatial velocity gradient.

        Args:
            tau_dr : Charact. optical depth.
            dv_dr : Velocity spatial gradient (dv is in c units, dr is in Rg units).
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
        if tau_max < TAU_MAX_TOL:
            aux = (1.0 - alpha) * (tau_max ** alpha)
        else:
            aux = ((1.0 + tau_max) ** (1.0 - alpha) - 1.0) / (
                (tau_max) ** (1.0 - alpha)
            )
        fm = k * t ** (-alpha) * aux
        # assert fm >= 0, "Force multiplier cannot be negative!"
        fm = max(0, fm)
        return fm

    def force_radiation(
        self,
        r,
        z,
        fm,
        tau_uv,
        no_tau_z=False,
        no_tau_uv=False,
    ):
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
        i_aux = self.integrator.integrate(r, z)
        self.int_hist.append(i_aux)
        if no_tau_z == True:
            d = np.sqrt(r ** 2 + z ** 2)
            sin_theta = z / d
            cos_theta = r / d
            tau_uv = tau_uv * np.array([cos_theta, 0, sin_theta])
            abs_uv = np.exp(-tau_uv)
        elif no_tau_uv == True:
            abs_uv = 1
        else:
            abs_uv = np.exp(-tau_uv)
        constant = (1 + fm) * self.FORCE_RADIATION_CONSTANT
        force = abs_uv * constant * np.asarray([i_aux[0], 0.0, i_aux[1],])
        assert force[2] >= 0
        return force
