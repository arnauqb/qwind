"""
IN DEVELOPMENT.
"""

import numpy as np
from scipy import integrate, interpolate, optimize

import qwind.constants as const
from pyagn import sed
from qwind import aux_numba
#from qwind.compiled_functions import *


class QSOSED:
    """
    New QSOSED with proper physics!
    """

    def __init__(self, wind):
        self.wind = wind
        self.sed_class = sed.SED(
            M=wind.M / const.Ms, mdot=wind.mdot, astar=wind.spin)
        self.r_in = self.sed_class.warm_radius
        if (self.wind.rho_shielding is None):
            self.wind.rho_shielding = self.wind.density_ss(self.r_in)
        # print(self.wind.rho_shielding)
        self.wind.tau_dr_0 = self.wind.tau_dr(self.wind.rho_shielding)
        self.wind.r_in = self.r_in
        self.r_out = self.sed_class.gravity_radius
        self.wind.r_out = self.r_out
        self.dr = (self.r_out - self.r_in) / (self.wind.nr - 1)
        self.r_init = self.r_in  # + 0.5 * self.dr
        self.wind.r_init = self.r_in  # + 0.5 * self.dr
        self.uv_fraction, self.xray_fraction = self.sed_class.uv_fraction, self.sed_class.xray_fraction
        self.uv_fraction_list = self.sed_class.compute_uv_fractions(
            1e26, include_corona=True)[0]
        self.r_range_interp = np.linspace(
            self.sed_class.corona_radius, self.sed_class.gravity_radius, len(self.uv_fraction_list))
        self.xray_luminosity = self.wind.mdot * \
            self.wind.eddington_luminosity * self.xray_fraction
        self.r_x = self.ionization_radius()
        self.force_radiation_constant = 3. * \
            self.wind.mdot / (8. * np.pi * self.wind.eta)
        aux_numba.r_range_interp = self.r_range_interp
        aux_numba.fraction_uv_list = self.uv_fraction_list
        # self.uv_fraction_interpolator = lambda r: interpolate_point_grid_1d(
        #    r, np.array(self.uv_fraction_list)[::5], self.r_range_interp[::5], 0)
        self.uv_fraction_interpolator = interpolate.interp1d(
            x=self.r_range_interp, y=self.uv_fraction_list,
            bounds_error=False, fill_value=0, kind='cubic')
        self.int_hist = []

        # interpolation values for force multiplier #
        k_interp_xi_values = [-4, -3, -2.26, -2.00, -1.50, -1.00,
                              -0.42, 0.00, 0.22, 0.50, 1.0,
                              1.5, 1.8, 2.0, 2.18, 2.39,
                              2.76, 3.0, 3.29, 3.51, 3.68, 4.0]
        k_interp_k_values = [0.411, 0.411, 0.400, 0.395, 0.363, 0.300,
                             0.200, 0.132, 0.100, 0.068, 0.042,
                             0.034, 0.033, 0.021, 0.013, 0.048,
                             0.046, 0.042, 0.044, 0.045, 0.032,
                             0.013]
        etamax_interp_xi_values = [-3, -2.5, -2.00, -1.50, -1.00,
                                   -0.5, -0.23, 0.0, 0.32, 0.50,
                                   1.0, 1.18, 1.50, 1.68, 2.0,
                                   2.02, 2.16, 2.25, 2.39, 2.79,
                                   3.0, 3.32, 3.50, 3.75, 4.00]
        etamax_interp_etamax_values = [6.95, 6.95, 6.98, 7.05, 7.26,
                                       7.56, 7.84, 8.00, 8.55, 8.95,
                                       8.47, 8.00, 6.84, 6.00, 4.32,
                                       4.00, 3.05, 2.74, 3.00, 3.10,
                                       2.73, 2.00, 1.58, 1.20, 0.78]

        self.k_interpolator = interpolate.interp1d(
            k_interp_xi_values,
            k_interp_k_values,
            bounds_error=False,
            fill_value=(k_interp_k_values[0], k_interp_k_values[-1]),
            kind='linear')  # important! xi is log here
        self.log_etamax_interpolator = interpolate.interp1d(
            etamax_interp_xi_values,
            etamax_interp_etamax_values,
            bounds_error=False,
            fill_value=(
                etamax_interp_etamax_values[0], etamax_interp_etamax_values[-1]),
            kind='linear')  # important! xi is log here

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
        delta_r_0 = abs(r_0 - self.r_in)
        delta_r = abs(r - r_0)
        distance = np.sqrt((r - self.r_in)**2 + z**2)
        sec_theta = distance / r
        tau_uv = sec_theta * (delta_r_0 * tau_dr_0 + delta_r * tau_dr)
        tau_uv = min(tau_uv, 50)
        assert tau_uv >= 0, "UV optical depth cannot be negative!"
        return tau_uv

    def ionization_parameter(self, r, z, tau_x, rho_shielding):
        """rtolnumber
        Computes Ionization parameter.

        Args:
            r: radius in Rg units.
            z: height in Rg units.
            tau_x: X-Ray optical depth at the point (r,z)
            rho_shielding: density of the atmosphere that contributes to shielding the X-Rays.

        Returns:
            ionization parameter.
        """
        distance_2 = r**2. + z**2.
        xi = self.xray_luminosity * \
            np.exp(-tau_x) / (rho_shielding *
                              distance_2 * self.wind.Rg**2)  # / 8.2125
        #print(r, z, tau_x, rho_shielding)
        assert xi > 0, "Ionization parameter cannot be negative!"
        xi += 1e-20  # to avoid overflows
        return xi

    def ionization_radius_kernel(self, rx):
        """
        Auxiliary function to compute ionization radius.

        Args:
            rx: Candidate for radius where material becomes non ionized.

        Returns:
            difference between current ion. parameter and target one.
        """
        tau_x = self.wind.tau_dr_0 * (rx - self.r_in)
        xi = self.ionization_parameter(rx, 0, tau_x, self.wind.rho_shielding)
        ionization_difference = const.IONIZATION_PARAMETER_CRITICAL - xi
        return ionization_difference

    def ionization_radius(self):
        """
        Computes the disc radius at which xi = xi_0 using the bisect method.
        """
        try:
            #r_x = optimize.bisect(self.ionization_radius_kernel, self.r_in, self.r_out)
            r_x = optimize.root_scalar(self.ionization_radius_kernel, bracket=[
                                       self.r_in, self.r_out])
        except ValueError:
            print("ionization radius outside the disc.")
            if(self.ionization_radius_kernel(self.wind.r_min) < 0):
                print("ionization radius is below r_min, nothing is ionized.")
                r_x = self.wind.r_min
            else:
                print(
                    "ionization radius is very large, atmosphere is completely ionized.")
                r_x = self.wind.r_max
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
        tau_x = self.wind.tau_dr_shielding
        tantheta = z / r
        d = np.sqrt(r**2 + z**2)
        if (r <= self.r_x):
            tau_x = tau_x * np.sqrt(r**2 + z**2)
        else:
            dp = self.r_x * np.sqrt(1 + tantheta**2)
            dp2 = d - dp
            tau_x = tau_x * (dp + 100 * dp2)
        #tau_x_0 = self.r_x - self.r_in
        # if (self.r_x < r_0):
        #    tau_x_0 += 100 * (r_0 - self.r_x)
        #distance = np.sqrt(r ** 2 + z ** 2)
        #sec_theta = distance / r
        #delta_r = abs(r - r_0)
        # tau_x = sec_theta * (tau_dr_0 * tau_x_0 + tau_dr *
        #                     self.opacity_x_r(r) * delta_r)
        assert tau_x >= 0, "X-Ray optical depth cannot be negative!"
        tau_x = min(tau_x, 50)
        return tau_x

    def force_multiplier_k(self, xi):
        """
        Auxiliary function required for computing force multiplier.

        Args: 
            xi: Ionisation Parameter.

        Returns:
            Factor k in the force multiplier formula.
        """
        if "interp_fm" in self.wind.modes:
            k = self.k_interpolator(np.log10(xi))
        else:
            k = 0.03 + 0.385 * np.exp(-1.4 * xi**(0.6))
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
        if("interp_fm" in self.wind.modes):
            eta_max = 10**(self.log_etamax_interpolator(np.log10(xi)))
        else:
            if(np.log10(xi) < 0.5):
                aux = 6.9 * np.exp(0.16 * xi**(0.4))
                eta_max = 10**aux
            else:
                aux = 9.1 * np.exp(-7.96e-3 * xi)
                eta_max = 10**aux
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

    def force_radiation(self, r, z, fm, tau_dr, tau_uv):
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

        if("constant_uv" in self.wind.modes):
            i_aux = aux_numba.integration_quad_nointerp(
                r, z, tau_dr, self.wind.r_min, self.wind.r_max)
            self.int_hist.append(i_aux)
            i_aux = np.array(i_aux) * self.sed_class.uv_fraction
        else:
            i_aux = aux_numba.integration_quad(
                r, z, self.wind.tau_dr_shielding, self.wind.r_min, self.wind.r_max)
            self.int_hist.append(i_aux)

        # try:
        #    assert i_aux[0] >= 0
        #    assert i_aux[1] >= 0
        # except:
        #    if ('old_integral' in self.wind.modes):
        #        pass
        #    else:
        #        raise "Negative radiation force!"

        #force_r = (i_aux[0][0] + fm * i_aux[0][1]) * abs_uv * self.force_radiation_constant
        #force_z = (i_aux[1][0] + fm * i_aux[1][1]) * abs_uv * self.force_radiation_constant
        #force = [force_r, 0., force_z]
        #abs_uv = np.exp(-tau_uv)
        force = (1 + fm) * self.force_radiation_constant * \
            np.asarray([i_aux[0], 0., i_aux[1]])
        return force
