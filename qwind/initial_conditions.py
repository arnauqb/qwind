from qwind import constants
import numpy as np
from scipy.optimize import brentq


class InitialConditions:

    def __init__(self, wind):
        self.wind = wind
        self.mu = 0.5

    def T_simple(self, r):
        a = 3 * constants.G * self.wind.M * self.wind.radiation.qsosed.mass_accretion_rate #(1-np.sqrt(self.wind.disk_r_min/r))
        b = 8 * np.pi * (r*self.wind.R_g)**3 * constants.SIGMA_SB
        T = (a/b)**(1./4.)
        return T

    def gravity(self, z, r):
        r = r * self.wind.R_g
        z = z * self.wind.R_g
        grav = constants.G * self.wind.M * z / (r**3)
        return grav
    
    def gas_pressure(self, z, r):
        T = self.wind.radiation.qsosed.disk_core_temperature(r)
        cs2 = constants.K_B * T / (self.mu * constants.M_P)
        r = r * self.wind.R_g
        z = z * self.wind.R_g
        press = cs2 / z
        return press
    
    def rad_pressure(self, z,r):
        T = self.T_simple(r)
        rad_pressure = constants.SIGMA_T / constants.M_P * constants.SIGMA_SB / constants.C * T**4 
        return rad_pressure
        
    def equation_to_solve(self, z,r):
        res = - self.gravity(z,r) + self.gas_pressure(z,r) + self.rad_pressure(z,r)
        return res
    
    def z0(self, r):
        height = brentq(self.equation_to_solve, 1e-5, 50, args=(r),xtol=1e-16, rtol=1e-14) 
        return height
    
    def edd_ratio(self, z,r):
        return self.rad_pressure(z,r) / self.gravity(z,r)

    def thermal_velocity(self, T):
        return np.sqrt(constants.K_B * T / (self.mu * constants.M_P))

    def mdot_cak(self, r):
        alpha = 0.6
        k = 0.03
        z_0 = self.z0(r)
        gamma = self.edd_ratio(z_0,r)
        T = self.T_simple(r)
        r = r * self.wind.R_g
        z_0 = z_0 * self.wind.R_g
        const = 1 / (constants.SIGMA_T / constants.M_P * self.thermal_velocity(T)) * constants.G * self.wind.M  * alpha * (1-alpha)**((1-alpha)/alpha)
        a = z_0 / r**3
        b = (k * gamma)**(1/alpha) * (1-gamma)**(-(1-alpha)/alpha)
        return const * a * b

    def number_density(self, r):
        T = self.T_simple(r)
        v_th = self.thermal_velocity(T)
        mdot = self.mdot_cak(r)
        rho = mdot / v_th
        n = rho / (self.mu * constants.M_P)
        return n

    def velocity(self,r):
        T = self.T_simple(r)
        return self.thermal_velocity(T)





