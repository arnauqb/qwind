import numpy as np
import aux_numba 
import constants as const
from scipy import optimize, integrate

"""
This module handles the radiation transfer aspects of Qwind.
"""

class Qwind():
    """
    Original implementation from RE2010.
    """

    def __init__(self, wind):
        self.wind = wind
        self.r_x = self.ionization_radius()
        self.force_radiation_constant =  3. * self.wind.mdot / (8. * np.pi * self.wind.eta) * (1 - self.wind.fx)
        self.int_hist = []

    def optical_depth_uv(self, r, z, r_0, tau_dr, tau_dr_0):
        """
        UV optical depth
        """
        
        tau_uv_0 = (r - self.wind.r_init)
        distance = np.sqrt(r**2 + z**2)
        delta_r = r - r_0
        sec_theta = distance / r
        tau_uv = sec_theta *  ( tau_dr_0 * tau_uv_0  +  delta_r * tau_dr )
        return tau_uv
    
    def ionization_parameter(self, r, z, tau_x, rho_shielding):
        """
        Computes Ionization parameter.
        """

        distance_2 = r**2. + z**2. 
        xi = self.wind.xray_luminosity * np.exp(-tau_x) / ( rho_shielding * distance_2 * self.wind.Rg**2)
        return xi

    def ionization_radius_kernel(self, rx):
        """
        Auxiliary function to compute ionization radius.
        """

        ionization_difference = const.ionization_parameter_critical - self.ionization_parameter(rx, 0, 0, self.wind.rho_shielding)
        return ionization_difference

    def ionization_radius(self):
        """
        Computes the disc radius at which xi = xi_0 using the bisect method.
        """

        r_x = optimize.bisect(self.ionization_radius_kernel, self.wind.r_in, self.wind.r_out)
        return r_x 

    def opacity_x_r(self, r):
        """
        X-Ray opacity as a function of radius.
        """

        if ( r < self.r_x):
            return 1 
        else:
            return 100 

    def optical_depth_x(self, r, z, r_0, tau_dr, tau_dr_0, rho_shielding):
        """
        X-Ray optical depth at a distance d.
        """

        tau_x_0 = (self.r_x - self.wind.r_init) 
        if ( self.r_x < r):
            tau_x_0 += 100 * ( r - self.r_x)

        distance = np.sqrt(r**2+z**2)
        sec_theta = distance / r
        delta_r = r - r_0
        tau_x = sec_theta * ( tau_dr_0 * tau_x_0 + tau_dr * self.opacity_x_r(r) * delta_r)
        return tau_x

    def k(self, xi):
        """
        Auxiliary function required for computing force multiplier.
        
        Parameters
        -----------
        xi: float
            Ionisation Parameter.
        """

        return 0.03 + 0.385 * np.exp(-1.4 * xi**(0.6))

    def eta_max(self, xi):
        """
        Auxiliary function required for computing force multiplier.
        
        Parameters
        -----------
        xi: float
            Ionisation Parameter.
        """
        
        if(np.log10(xi) < 0.5):
            aux = 6.9 * np.exp(0.16 * xi**(0.4))
            return 10**aux
        else:
            aux = 9.1 * np.exp(-7.96e-3 * xi)
            return 10**aux

    def sobolev_optical_depth(self, tau_dr, dv_dr):
        """
        Returns differential optical depth times a factor that compares thermal velocity with spatial velocity gradient.
        Required by ForceMultiplier.
        
        Parameters
        -----------
        tau_dr : float
            Differential optical depth.
        dv_dr : float
            Velocity spatial gradient.
        T : float
            Wind temperature.
        """
        sobolev_length = self.wind.v_thermal / np.abs(dv_dr)
        sobolev_optical_depth = tau_dr * sobolev_length
        return sobolev_optical_depth 

    def force_multiplier(self, t, xi):
        """
        Computes Force multiplier.
        
        Parameters
        -----------
        t : float
            Sobolev Optical Depth.
        xi : float
            Ionisation Parameter.
        """

        k = self.k(xi)
        eta_max = self.eta_max(xi)
        tau_max = t * eta_max
        alpha = 0.6
        if (tau_max < 0.001):
            aux = (1. - alpha) * (tau_max ** alpha)
        else:
            aux = ((1. + tau_max)**(1. - alpha) - 1.) / \
                ((tau_max) ** (1. - alpha))
        return k * t**(-alpha) * aux

    def force_radiation(self, r, z, fm, tau_uv, fm_mean):
        """
        Computes radiation force at point (r,z)
        """

        if('old_integral' in self.wind.modes):
            i_aux = aux_numba.qwind_integration( r, z, tau_uv)
        else:
            i_aux = aux_numba.qwind_integration_dblquad(r, z, tau_uv, self.wind.r_min, self.wind.r_max)

        self.int_hist.append(i_aux)
        force = ( 1 + fm ) * self.force_radiation_constant * np.asarray([i_aux[0], 0., i_aux[1]])
        return force

