"""
This module handles the radiation transfer aspects of Qwind.
"""

import numpy as np
from scipy import integrate, interpolate, optimize
from qwind.integration import qwind2 as integration 
from numba import njit
from qsosed import sed
from qwind import grid
from skimage.draw import line

import qwind.constants as const
# interpolation values for force multiplier #
_K_INTERP_XI_VALUES = [-4, -3, -2.26, -2.00, -1.50, -1.00,
                      -0.42, 0.00, 0.22, 0.50, 1.0,
                      1.5, 1.8, 2.0, 2.18, 2.39,
                      2.76, 3.0, 3.29, 3.51, 3.68, 4.0]
_K_INTERP_K_VALUES = [0.411, 0.411, 0.400, 0.395, 0.363, 0.300,
                     0.200, 0.132, 0.100, 0.068, 0.042,
                     0.034, 0.033, 0.021, 0.013, 0.048,
                     0.046, 0.042, 0.044, 0.045, 0.032,
                     0.013]
_ETAMAX_INTERP_XI_VALUES = [-3, -2.5, -2.00, -1.50, -1.00,
                           -0.5, -0.23, 0.0, 0.32, 0.50,
                           1.0, 1.18, 1.50, 1.68, 2.0,
                           2.02, 2.16, 2.25, 2.39, 2.79,
                           3.0, 3.32, 3.50, 3.75, 4.00]

_ETAMAX_INTERP_ETAMAX_VALUES = [6.95, 6.95, 6.98, 7.05, 7.26,
                               7.56, 7.84, 8.00, 8.55, 8.95,
                               8.47, 8.00, 6.84, 6.00, 4.32,
                               4.00, 3.05, 2.74, 3.00, 3.10,
                               2.73, 2.00, 1.58, 1.20, 0.78]


def cooling_g_compton(T, xi, Tx=1e8):
    g_compton = 8.9e-36 * xi * (Tx - 4 * T)
    return g_compton

def cooling_g_photoi(T, xi, Tx=1e8):
    g_photo = 1.5e-21 * xi**(1./4.) * T**(0.5) * ( 1. - T/Tx)
    return g_photo

def cooling_brem(T, xi):
    lbl = 3.3e-27 * T**(0.5) + 1.7e-18 / (xi * T**(0.5)) * np.exp(-1.3e5 / T) + 1e-24
    return lbl

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
        self.qsosed = sed.SED(self.wind.M / const.M_SUN, self.wind.mdot, number_bins_fractions=grid.N_1D_GRID)
        if "qsosed_geometry" in self.wind.modes:
            self.uv_radial_flux_fraction = self.qsosed.compute_uv_fractions(return_all = False)
            self.xray_fraction = self.qsosed.xray_fraction
            self.uv_fraction = self.qsosed.uv_fraction
            self.FORCE_RADIATION_CONSTANT = 3. / (8. * np.pi * self.wind.eta)
            self.wind.disk_r_min = self.qsosed.warm_radius
            self.wind.disk_r_max = self.qsosed.gravity_radius
            #update grid
            #integration.UV_FRACTION_GRID_RANGE = np.linspace(self.wind.disk_r_min, self.wind.disk_r_max, 500)
            grid.UV_FRACTION_GRID = self.uv_radial_flux_fraction
        else:
            self.xray_fraction = self.wind.f_x 
            self.uv_fraction = 1 - self.wind.f_x
            self.FORCE_RADIATION_CONSTANT = 3. / (8. * np.pi * self.wind.eta) * self.uv_fraction
            grid.UV_FRACTION_GRID = np.ones(grid.N_1D_GRID)
        self.mdot_0 = wind.mdot
        grid.GRID_1D_RANGE = np.linspace(self.wind.disk_r_min, self.wind.disk_r_max, grid.N_1D_GRID)
        grid.MDOT_GRID = self.mdot_0 * np.ones_like(grid.GRID_1D_RANGE)
        #integration.MDOT_GRID = self.mdot_0 * np.ones(500)
        #self.mdot_grid = integration.MDOT_GRID
        #integration.MDOT_GRID_RANGE = np.linspace(self.wind.disk_r_min, self.wind.disk_r_max, 500)
        self.dr = (self.wind.lines_r_max - self.wind.lines_r_min) / (self.wind.nr - 1)
        self.wind.tau_dr_0 = self.wind.tau_dr(self.wind.rho_shielding)
        self.xray_luminosity = self.wind.mdot * self.wind.eddington_luminosity * self.xray_fraction
        self.int_hist = []
        self.int_error_hist = []


        self.k_interpolator = interpolate.interp1d(
            _K_INTERP_XI_VALUES,
            _K_INTERP_K_VALUES,
            bounds_error=False,
            fill_value=(_K_INTERP_K_VALUES[0], _K_INTERP_K_VALUES[-1]),
            kind='cubic')  # important! xi is log here
        self.log_etamax_interpolator = interpolate.interp1d(
            _ETAMAX_INTERP_XI_VALUES,
            _ETAMAX_INTERP_ETAMAX_VALUES,
            bounds_error=False,
            fill_value=(
                _ETAMAX_INTERP_ETAMAX_VALUES[0],
                _ETAMAX_INTERP_ETAMAX_VALUES[-1]),
            kind='cubic')  # important! xi is log here

        # Interpolation grids #
        self.initialize_all_grids(first_iter=True)
        self.integrator = integration.Integrator(self)
        

    def compute_mass_accretion_rate_grid(self, lines):
        """
        Returns mass accretion rate at radius r, taking into account the escaped wind.
        Also updates it from the grid file.
        """
        new_mdot_list = integration.MDOT_GRID.copy()
        lines_escaped = np.array(lines)[np.where([line.escaped for line in lines])[0]]
        for line in lines_escaped:
            r_0 = line.r_0
            width = line.line_width
            r_f_arg = np.searchsorted(integration.GRID_1D_RANGE, r_0 + width/2.)
            mdot_w = self.wind.compute_line_mass_loss(line) 
            mdot_w_normalized = mdot_w / self.qsosed.mass_accretion_rate 
            new_mdot_list[0:r_f_arg] -= mdot_w_normalized
        grid.MDOT_GRID = np.array(new_mdot_list)
        self.mdot_grid.grid = grid.MDOT_GRID

        #for r in r_range:
        #    if r > self.wind.escaped_radii[0] and r < self.wind.escaped_radii[-1]:
        #        mdot_wind = self.wind.mdot_w * self.wind.eta * const.C**2 / self.qsosed.eddington_luminosity 
        #        mdot = self.mdot_0 - mdot_wind 
        #    else:
        #        mdot = self.mdot_0
        #    mdot_list.append(mdot)
        #return np.array(mdot_list)

    def _optical_depth_uv_integrand(self, t, r, z, grid_r_range, grid_z_range, density_grid):
        """
        Auxiliary function used to compute the UV optical depth.
        """
        r = t * r
        z = t * z
        r_arg = np.searchsorted(grid_r_range, r, side="left")
        z_arg = np.searchsorted(grid_z_range, z, side="left")
        r_arg = min(r_arg, grid_r_range.shape[0] - 1)
        z_arg = min(z_arg, grid_z_range.shape[0] - 1)
        density = density_grid[r_arg, z_arg]
        return density 

        
    def optical_depth_uv(self,r, z, r_0, tau_dr, tau_dr_0):
        """
        UV optical depth.
        
        Args:
            r: radius in Rg units.
            z: height in Rg units.
        
        Returns:
            UV optical depth at point (r,z) 
        """
        if "uv_interp" in self.wind.modes:
            #grid_r_range = grid.GRID_R_RANGE 
            #grid_z_range = grid.GRID_Z_RANGE 
            #density_grid = grid.DENSITY_GRID 
            #line_element = np.sqrt(r**2 + z**2) * self.wind.RG * const.SIGMA_T
            #tau_uv_int = integrate.quad(self._optical_depth_uv_integrand,
            #                            a=0,
            #                            b=1,
            #                            args=(r, z, grid_r_range, grid_z_range, density_grid),
            #                            epsabs=0,
            #                            epsrel=1e-2)[0]
            #return tau_uv_int * line_element
            dtau_grid = self.density_grid.grid * const.SIGMA_T * self.wind.RG
            r_arg, z_arg = self.density_grid.get_arg(r,z) 
            line_coordinates = line(0,0, r_arg, z_arg)
            tau = dtau_grid[line_coordinates].sum() / len(line_coordinates[0]) * np.sqrt(r**2 + z**2)
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
        tau_x = self.tau_x_grid.get_value(r,z)
        d2 = r**2. + z**2.
        xi = self.xray_luminosity * np.exp(-tau_x) /  ( rho * d2 * self.wind.RG**2)
        return xi + 1e-15 # to avoid roundoff issues

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
        tau_x = self.tau_x_grid.get_value(r,z)
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
        if (tau_max < TAU_MAX_TOL):
            aux = (1. - alpha) * (tau_max ** alpha)
        else:
            aux = ((1. + tau_max)**(1. - alpha) - 1.) / \
                ((tau_max) ** (1. - alpha))
        fm = k * t**(-alpha) * aux
        #assert fm >= 0, "Force multiplier cannot be negative!"
        fm = max(0,fm)
        return fm

    
    def compute_equilibrium_temperature(self, n, xi, Tx=1e8):
        try:
            root = optimize.root_scalar(cooling_total, args=(n, xi, Tx), bracket=(10, 1e9), method='bisect') 
        except ValueError:
        #    print(xi)
            xi *= 5
            root = self.compute_equilibrium_temperature(n, xi, Tx)
            return root
        assert root.converged
        return root.root

    def compute_temperature(self, n, R, xi, Tx=1e8):
        xi = max(xi, 1e-10)
        eq_temp = self.compute_equilibrium_temperature(n, xi, Tx)
        disk_temp = self.qsosed.disk_nt_temperature4(R)**(1./4.)
        return max(disk_temp, eq_temp)

    def force_radiation(self, r, z, fm, tau_uv, return_error=False, no_tau_z=False, no_tau_uv=False, **kwargs):
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
        i_aux = self.integrator.integrate(r,
                                          z,
                                          self.wind.disk_r_min,
                                          self.wind.disk_r_max,
                                          **kwargs)
        error = i_aux[2:4]
        self.int_error_hist.append(error)
        self.int_hist.append(i_aux)
        if no_tau_z == True:
            d = np.sqrt(r**2 + z**2)
            sin_theta = z / d
            cos_theta = r / d
            tau_uv = tau_uv * np.array([cos_theta, 0, sin_theta])
            abs_uv = np.exp(-tau_uv)
        elif no_tau_uv == True:
            abs_uv = 1
        else:
            abs_uv = np.exp(-tau_uv)
        constant = abs_uv * (1 + fm) * self.FORCE_RADIATION_CONSTANT
        force = constant  * np.asarray([i_aux[0],
                                        0.,
                                        i_aux[1],
                                        ])
        assert force[2] >= 0
        if return_error:
            error = constant * np.array(error)
            return [force, error]
        return force

    def initialize_all_grids(self, first_iter=True):
        self.density_grid = grid.DensityGrid(self.wind.rho_shielding)
        self.ionization_grid = grid.IonizationParameterGrid(self.xray_luminosity, self.wind.RG)
        self.tau_x_grid = grid.OpticalDepthXrayGrid(self.wind.RG)
        if first_iter:
            self.mdot_grid = grid.Grid1D(self.mdot_0)
            self.uv_fraction_grid = grid.Grid1D(1)
            self.uv_fraction_grid.grid = grid.UV_FRACTION_GRID
        self.update_all_grids(init=True)

    def update_all_grids(self, init=False, end=False):
        """
        Updates all grids after one iteration (density, ionization parameter, optical depth)
        """
        print("updating grids...")
        if not init:
            self.density_grid.update_grid(self.wind)
        self.ionization_grid.update_grid(self.density_grid, self.tau_x_grid)
        self.tau_x_grid.update_grid(self.density_grid, self.ionization_grid)
        print("...")
        self.ionization_grid.update_grid(self.density_grid, self.tau_x_grid)
        self.tau_x_grid.update_grid(self.density_grid, self.ionization_grid)
        self.ionization_grid.update_grid(self.density_grid, self.tau_x_grid)

        

