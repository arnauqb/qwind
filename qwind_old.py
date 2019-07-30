import numpy as np
import scipy
import astropy.constants as astroconst
from astropy import units as u
import scipy.constants as const
import scipy.integrate
import scipy.optimize
from joblib import Parallel, delayed
import time
from multiprocessing import Process, Value, Pool
#from profilehooks import profile
import utils
import os
import numba as nb
from numba import jitclass, jit

# check backend to import appropiate progress bar #
def tqdm_dump(array):
    return array
backend = utils.type_of_script()
if(backend == 'jupyter'):
    from tqdm import tqdm_notebook as tqdm
else:
    tqdm = tqdm_dump
   #asd = open("asd.txt", "w")

@jit(nopython=True)
def _integrate_dblquad_kernel_r(r_d, phi_d, r, z, abs_uv, tau_c):
    ff0 = (1. - np.sqrt(6. / r_d)) / r_d ** 3.
    delta = _Distance_gas_disc(r_d, phi_d, r, z)
    cos_gamma = (r - r_d * np.cos(phi_d)) / delta
    sin_gamma = z / delta
    darea = r_d  # * deltar * deltaphi
    # test #
    #uu = abs_uv
    #abs_uv = np.exp(-tau_c * delta)
    #asd.write("%f \t %f \n"%(uu, abs_uv))
    ff = 2. * ff0 * darea * sin_gamma / delta ** 2. * abs_uv
    return ff * cos_gamma  # [ff*cos_gamma, ff*sin_gamma]


@jit(nopython=True)
def _integrate_dblquad_kernel_z(r_d, phi_d, r, z, abs_uv, tau_c):
    ff0 = (1. - np.sqrt(6. / r_d)) / r_d ** 3.
    delta = _Distance_gas_disc(r_d, phi_d, r, z)
    sin_gamma = z / delta
    # test #
    #abs_uv = np.exp(-tau_c * delta)
    darea = r_d  # * deltar * deltaphi
    ff = 2. * ff0 * darea * sin_gamma / delta ** 2. * abs_uv
    return ff * sin_gamma


def _integrate_dblquad(r, z, abs_uv, tau_c,Rmin, Rmax):
    r_int = scipy.integrate.dblquad(
        _integrate_dblquad_kernel_r,
        0,
        np.pi,
        Rmin,
        Rmax,
        args=(
            r,
            z,
            abs_uv,
            tau_c))[0]
    z_int = scipy.integrate.dblquad(
        _integrate_dblquad_kernel_z,
        0,
        np.pi,
        Rmin,
        Rmax,
        args=(
            r,
            z,
            abs_uv,
            tau_c))[0]
    return [r_int, z_int]


@jit(nopython=True)
def _Distance_gas_disc(r_d, phi_d, r, z):
    return np.sqrt(
        r ** 2. +
        r_d ** 2. +
        z ** 2. -
        2. *
        r *
        r_d *
        np.cos(phi_d))


@jit(nopython=True)
def _Force_integral_kernel(r_d, phi_d, deltaphi, deltar, r, z, abs_uv):
    ff0 = (1. - np.sqrt(6. / r_d)) / r_d ** 3.
    delta = _Distance_gas_disc(r_d, phi_d, r, z)
    cos_gamma = (r - r_d * np.cos(phi_d)) / delta
    sin_gamma = z / delta
    darea = r_d * deltar * deltaphi
    ff = 2. * ff0 * darea * sin_gamma / delta ** 2. * abs_uv
    return [ff * cos_gamma, ff * sin_gamma]


@jit(nopython=True)
def integration(rds, phids, deltards, deltaphids, r, z, abs_uv):
    integral = [0., 0.]
    for i in range(0, len(deltards)):
        for j in range(0, len(deltaphids)):
            aux = _Force_integral_kernel(
                rds[i], phids[j], deltaphids[j], deltards[i], r, z, abs_uv)
            integral[0] += aux[0]
            integral[1] += aux[1]
    return integral


class streamline:

    ## constants ##
    G = astroconst.G.cgs.value
    Ms = astroconst.M_sun.cgs.value
    c = astroconst.c.cgs.value
    m_p = astroconst.m_p.cgs.value
    k_B = astroconst.k_B.cgs.value
    Ryd = u.astrophys.Ry.cgs.scale
    sigma_sb = astroconst.sigma_sb.cgs.value
    sigma_t = const.physical_constants['Thomson cross section'][0] * 1e4
    year = u.yr.cgs.scale

    def __init__(
            self,
            r0=375.,
            r_init=236.84,
            dr = 100,
            z0=1.,
            M=2e8,
            mdot=0.5,
            a=0.,
            rho=2e8,
            v_z0=5e7,
            v_r0=0.,
            fx=0.15,
            Rin=200.,
            Rout=1600.,
            Rmin=6.,
            Rmax=1400.,
            beta=0.,
            T=2e6,
            mu=1.,
            modes=[],
            save_dir="Results",
            dt=4.096 / 10,
            intsteps=1.,
            eta = 0.06):

        self.save_dir = save_dir
        try:
            os.mkdir(save_dir)
        except BaseException:
            pass

        # array containing different modes for debugging #
        self.modes = modes

        # black hole and disc variables #
        self.M = M * self.Ms
        self.mdot = mdot
        self.a = [0, 0, 0]  # / u.s**2
        self.eta = eta#0.06 #0.0313 #0.06
        self.T = T  # * u.K
        self.mu = mu
        self.v_th = self.ThermalVelocity() / self.c
        self.fx = fx
        self.xi0 = 1e5 / (4 * np.pi * self.Ryd * self.c)
        self.Rin = Rin
        self.Rout = Rout
        self.Rmin = Rmin
        self.Rmax = Rmax
        self.r_init = r_init
        self.beta = beta
        self.Rg = self.G * self.M / (self.c ** 2)
        self.Rs = 2 * self.Rg
        self.rho0 = rho  
        self.opacity = self.OpacityLaw(self.rho0 * self.mu * self.m_p, self.T)
        self.GE = 4 * np.pi * self.m_p * self.c**3 / (self.opacity)
        self.SE = self.GE / self.Rg
        self.norm = 3 * self.mdot * self.mu / (8. * np.pi * self.eta)
        self.rho = self.rho0
        self.Ledd = self.EddingtonLuminosity()
        self.Lbol = self.BolLuminosity()

        ## position variables ##
        ## coordinates of particle are (R, phi, z) ##
        ## all positions are in units of Rg, all velocities in units of c. ##
        self.dt = dt  # units of  Rg / c
        self.r = r0
        self.phi = 0
        self.z = z0
        self.x = [self.r, self.phi, self.z]
        self.d = np.sqrt(self.r**2 + self.z**2)
        self.t = 0  # in seconds
        self.r_0 = r0
        self.z_0 = z0
        self.v_r = v_r0 / self.c   
        self.v_r_hist = [self.v_r]
        self.v_phi = self.v_kepler()
        self.l = self.v_phi * self.r  # initial angular momentum
        self.v_phi0 = self.v_phi
        self.v_z0 = v_z0 / self.c  
        self.v_z = self.v_z0  
        self.v = [self.v_r, self.v_phi, self.v_z]
        self.v_T = np.sqrt(self.v_z ** 2 + self.v_r ** 2)
        self.dv_dr = 0  
        self.dr_e = 0  
        self.escaped = False # this variable tracks whether the wind has reached the escape velocity

        
        ## optical depths ##
        self.tau_c0 = self.mu * self.opacity * self.rho0 * self.Rg  # self.CharactOptDep(self.r_0)
        self.tau_c = self.tau_c0
        self.tau_c_hist = [self.tau_c]
        try:
            self.Rx = self.FindRx() #328.73#self.FindRx() #280.45 # self.FindRx()
        except:
            print("shielding very low! \n")
            self.Rx = 1400
        self.tau_X0 = self.XrayOptDep(self.r_0)
        self.tau_UV0 = self.UVOptDep(self.r_0)
        self.tau_UV = self.tau_UV0
        self.tau_X = self.tau_X0
        self.tau_eff = 0
        self.fm = 0
        self.xi = self.Update_xi()


        ## aux variables for integration  ##
        self.phids = np.linspace(0, np.pi, intsteps * 100 + 1)
        self.deltaphids = np.asarray(
            [self.phids[i + 1] - self.phids[i] for i in range(0, len(self.phids) - 1)])
        self.rds = np.geomspace(self.Rmin, self.Rmax, intsteps * 250 + 1)
        self.deltards = np.asarray(
            [self.rds[i + 1] - self.rds[i] for i in range(0, len(self.rds) - 1)])

        # force related variables #

        self.Fgrav = []
        self.Frad = []
        self.iter = []

    # history variables #
        # position and velocities histories # 
        self.x_hist = [self.x]
        self.d_hist = [self.d]
        self.t_hist = [0]
        self.r_hist = [r0]
        self.phi_hist = [0]
        self.z_hist = [z0]
        self.v_phi_hist = [self.v_phi]
        self.v_z_hist = [self.v_z]
        self.v_hist = [self.v]
        self.v_T_hist = [self.v_T]
        # radiation related histories #
        self.opacity_hist = [self.opacity]
        self.rho_hist = [self.rho0]
        self.dv_dr_hist = [0]
        self.v2_hist=[0]
        self.dvt_hist=[0]
        self.dr_e_hist = [self.dr_e]
        self.tau_UV_hist = [self.tau_UV0]
        self.tau_X_hist = [self.tau_X]
        self.tau_eff_hist = [0]
        self.taumax_hist = []
        self.fm_hist = [1]
        self.xi_hist = [self.xi]
        #force histories #
        self.int_hist = []
        self.a_hist = [self.a]


    def MassAccretionRate(self):
        ## check iconsistency with qwind code ##
        aux = self.mdot * self.Ledd / (self.eta * self.c**2)
        aux = aux * u.g / u.s
        return aux.to(u.M_sun / u.year)

    def norm2d(self, vector):
        return np.sqrt(vector[0] ** 2 + vector[-1] ** 2)

    def dist2d(self, x, y):
        # 2d distance in cyl coordinates #
        dr = y[0] - x[0]
        dz = y[2] - x[2]
        return np.sqrt(dr**2 + dz**2)

    def v_kepler(self):
        return np.sqrt(1. / (self.r))

    def v_esc(self):
        return np.sqrt(2. / self.d)

    def EddingtonLuminosity(self):
        """ Returns the Eddington Luminosity. """
        return self.GE * self.Rg

    def BolLuminosity(self):
        """ Bolumetric Luminosity """
        return self.mdot * self.Ledd

    def T4(self, r):
        rel = (1. - np.sqrt(6. / r)) / r**3
        return self.norm * rel

    def Radiance(self, r):
        """Computes Disc Radiance assuming stantard SS disc.
        Radius in Rg units"""
        return self.sigma_sb * self.T4(r)

    def RadianceNorm(self, r):
        return self.sigma_sb * self.T4(r) / self.SE

    def ThermalVelocity(self):
        """Thermal velocity for gas with molecular weight mu and temperature T"""
        return np.sqrt(self.k_B * self.T / (self.mu * self.m_p))

    def NumberDensity(self, r):
        """Gas density profile, SS has beta = 0"""
        return self.rho0 * (r / self.r_0)**(-self.beta)
        # return self.tau_c0 * r ** (-self.beta)

    def MassDensity(self, r):
        """Mass Gas density profile"""
        return self.m_p * self.NumberDensity(r)

    def ElectronNumberDensity(self, r):
        """Electron number density profile"""
        Y = 1  # 1./5. * ( 3. / self.mu)
        return Y * self.NumberDensity(r)

    ## ionization ##

    def OpacityLaw(self, rho, T, uvorxray = 0):
        return self.sigma_t 

    def CharactOptDep(self, r):
        """ Electron scattering characteristic Optical depth """
        return self.opacity * self.ElectronNumberDensity(r) * self.Rg

    def UVOptDep(self, r):
        """ UV electron optical depth"""
        if (self.beta == 0):
            return self.tau_c * (r - self.r_init)
        else:
            return scipy.integrate.quad(self.CharactOptDep, self.r_init, r)[0]

    def XRayOpacity(self):
        if (self.xi > self.xi0):
            return 1
        else:
            return 100

    def XRayOpacity_r(self,r):
        if (r > self.Rx):
            return 100
        else:
            return 1


    def XrayOptDep(self, r):
        """ X-ray optical depth """
        #return self.CharactOptDep(r) * (r-self.Rin)
        aux = (self.Rx - self.r_init)
        if (self.Rx < r):
            aux += 100 * (r - self.Rx)
        return self.CharactOptDep(r) * aux
    
    def XrayOptDep_sensible(self, r):
        """ X-ray optical depth """
        #return self.CharactOptDep(r) * (r-self.Rin)
        if ( r < self.r_init):
            return 0
        
        if ( r < self.Rx):
            aux = r - self.r_init
        else:
            aux = (self.Rx - self.r_init) + 100 * ( r - self.Rx)
        return self.CharactOptDep(r) * aux
    
    def OptDepth(self, r,z):
        """ X-ray optical depth """
        r_range = np.linspace(self.r_init, r, 50+1)
        aux = 0
        for i in range(0,len(r_range)-1):
            rp = r_range[i]
            xi = self.IonParameterNoAbs(rp,z)
            xi *= np.exp(-aux)
            aux += (1 + self.Force_multiplier_static(xi)) * self.CharactOptDep(r) * (r_range[i+1] - rp)
        return aux

    def FindRxKernel(self, rx):
        """ Auxiliary function for FindRx """
        self.Rx = rx
        return self.xi0 - self.IonParameterFindRx(rx)

    def FindRx(self):
        """ Finds radius at which xi = xi_0  """
        self.Rx = scipy.optimize.bisect(self.FindRxKernel, self.Rin, self.Rout)
        return self.Rx

    def NormIonParameter(self):
        """ Normalization ionization parameter. """
        return self.GE * self.opacity / (4 * np.pi * self.Ryd * self.c)

    def IonParameterFindRx(self, r):
        """ Dimensionless ionization parameter """
        #tau = self.XrayOptDep(r)
        tau = 0
        d2 = self.z**2 + r**2
        aux = self.fx * self.mdot / (self.CharactOptDep(r) * d2)
        return self.NormIonParameter() * aux   * np.exp(-tau)

    def IonParameterNoAbs(self, r,z):
        """ Dimensionless ionization parameter """
        #tau = self.XrayOptDep(r)
        #tau = self.OptDepth(r,z)
        d2 = z**2 + r**2
        aux = self.fx * self.mdot / (self.CharactOptDep(r) * d2)
        return self.NormIonParameter() * aux# * np.exp(-tau)
    
    def IonParameter(self, r,z):
        """ Dimensionless ionization parameter """
        #tau = self.XrayOptDep(r)
        tau = self.OptDepth(r,z)
        d2 = z**2 + r**2
        aux = self.fx * self.mdot / (self.CharactOptDep(r) * d2)
        return self.NormIonParameter() * aux * np.exp(-tau)
    
    def IonParameter_old(self, r,z):
        """ Dimensionless ionization parameter """
        tau = self.XrayOptDep(r)
        #tau = self.OptDepth(r,z)
        d2 = z**2 + r**2
        aux = self.fx * self.mdot / (self.CharactOptDep(r) * d2)
        return self.NormIonParameter() * aux * np.exp(-tau)
    

    ###############
    ## streaming ##
    ###############

    def Update_xi(self):
        """ Dimensionless ionization parameter """
        const = self.NormIonParameter() * self.fx * self.mdot
        self.xi = const * (1. / (self.tau_c * self.d**2)) * np.exp(- self.tau_X)
        return self.xi

    def Update_Tau_C(self):
        """ evolving tauc"""
        if(self.v_z < 0):
            self.rho_hist.append(self.rho)
            return self.tau_c
        radial = (self.r / self.r_0) ** (-2.)
        v_ratio = self.v_z0 / self.norm2d(self.v)
        self.tau_c = self.tau_c0 * radial * v_ratio
        self.rho = self.rho0 * radial * v_ratio
        self.rho_hist.append(self.rho)
        return self.tau_c

    def Update_Tau_UV(self):
        sectheta = self.d / self.r  # np.sqrt( 1. + tantheta**2. )
        Deltar = self.r - self.r_0
        self.tau_UV = sectheta * (self.tau_UV0 +  self.tau_c * Deltar)
        return self.tau_UV

    def Update_Tau_X(self):

        sectheta = self.d / self.r  # np.sqrt( 1. + tantheta**2. )
        Deltar = self.r - self.r_0
        self.tau_X = sectheta * (self.tau_X0 + self.XRayOpacity() * self.tau_c * Deltar)
        return self.tau_X

    def Update_reff(self):
        self.dr_e = np.abs(self.v_th / self.dv_dr)
        return self.dr_e

    def Update_EffectiveOpticalDepth(self):

        self.dr_e = self.Update_reff()
        self.dr_e_hist.append(self.dr_e)
        self.tau_eff = self.dr_e * self.tau_c
        return self.tau_eff

    def K(self, xi):
        """Required for computing force multiplier"""
        return 0.03 + 0.385 * np.exp(-1.4 * xi**(0.6))

    def etamax(self, xi):
        """Required for computing force multiplier"""
        if(np.log10(xi) < 0.5):
            aux = 6.9 * np.exp(0.16 * xi**(0.4))
            return 10**aux
        else:
            aux = 9.1 * np.exp(-7.96e-3 * xi)
            return 10**aux
        
    def Force_multiplier_static(self,xi):
        """Computes force multiplier given the effective optical depth and
        the ionization parameter"""
        xi = xi * 8.2125
        K = self.K(xi)
        etamax = self.etamax(xi)
        taumax = etamax
        alpha = 0.6
        if (taumax < 0.001):
            fm_tfac = (1. - alpha) * (taumax ** alpha)
        else:
            fm_tfac = ((1. + taumax)**(1. - alpha) - 1.) / \
                ((taumax) ** (1. - alpha))
        fm = K * fm_tfac
        return fm

    def Update_force_multiplier(self):
        """Computes force multiplier given the effective optical depth and
        the ionization parameter"""
        K = self.K(self.xi)
        etamax = self.etamax(self.xi)
        taumax = self.tau_eff * etamax
        alpha = 0.6
        self.tau_eff_hist.append(self.tau_eff)
        self.taumax_hist.append(taumax)
        if (taumax < 0.001):
            fm_tfac = (1. - alpha) * (taumax ** alpha)
        else:
            fm_tfac = ((1. + taumax)**(1. - alpha) - 1.) / \
                ((taumax) ** (1. - alpha))
        self.fm = K * self.tau_eff ** (- alpha) * fm_tfac
        self.fm_hist.append(self.fm)
        return self.fm

    ## radiation force part ##

    def Distance_gas_disc(self, r_d, phi_d):
        return np.sqrt(
            self.r ** 2 +
            r_d ** 2 +
            self.z ** 2 -
            2 *
            self.r *
            r_d *
            np.cos(phi_d))

    ############################# qwind integration method ####################

    def Distance_gas_disc(self, r_d, phi_d):
        return np.sqrt(
            self.r ** 2 +
            r_d ** 2 +
            self.z ** 2 -
            2 *
            self.r *
            r_d *
            np.cos(phi_d))

    def Force_integral_kernel_r(self, r_d, phi_d, deltaphi, deltar):
        ff0 = (1. - np.sqrt(6. / r_d)) / r_d ** 3.
        delta = self.Distance_gas_disc(r_d, phi_d)
        cos_gamma = (self.r - r_d * np.cos(phi_d)) / delta
        sin_gamma = self.z / delta
        dArea = r_d * deltar * deltaphi
        ff = 2. * ff0 * dArea * sin_gamma / delta ** 2. * np.exp(- self.tau_UV)
        return np.asarray([ff * cos_gamma, ff * sin_gamma])

    def Force_integral_r_phi(self, r_d, deltar):
        aux = np.asarray([0., 0.])
        for i in range(0, len(self.phids) - 1):
            deltaphi = self.phids[i + 1] - self.phids[i]
            # * deltaphi #self.Force_integral_kernel_r(r_d, self.phids[i]) * deltaphi
            aux += self.Force_integral_kernel_r(r_d,
                                                self.phids[i], deltaphi, deltar)
        return aux

    def Force_integral_r_rd(self):
        aux = np.asarray([0., 0.])
        for i in range(0, len(self.rds) - 1):
            deltar = self.rds[i + 1] - self.rds[i]
            aux += self.Force_integral_r_phi(self.rds[i], deltar)  # * deltar
        return aux

    ########################################################################

    def Force_radiation(self):

        if('oldint' in self.modes):
            i_aux = integration(self.rds,
                                self.phids,
                                self.deltards,
                                self.deltaphids,
                                self.r,
                                self.z,
                                np.exp(-self.tau_UV))
            self.int_hist.append(i_aux)
        else:
            i_aux = _integrate_dblquad(
                self.r, self.z, np.exp(-self.tau_UV), self.tau_c, self.Rmin, self.Rmax)
            self.int_hist.append(i_aux)

        constant = 3. * self.mdot / \
            (8. * np.pi * self.eta) * (1 + self.fm) * (1 - self.fx)
        return constant * np.asarray([i_aux[0], 0., i_aux[1]])  # integrals

    ##########################################################################

    ## gravity ##

    def Force_gravity(self):
        array = np.asarray([self.r / self.d, 0., self.z / self.d])
        grav = - 1. / (self.d**2) * array
        return grav

   ## kinematics ##

    def UpdatePositionsMyWay(self):
        # compute acceleration vector #
        fg = self.Force_gravity()
        fr = self.Force_radiation()
        self.Fgrav.append(fg)
        self.Frad.append(fr)
        self.a = fg
        if('gravityonly' in self.modes):
            self.a += 0
        else:
            self.a += fr

        self.a[0] += self.l**2 / self.r**3
        self.a_hist.append(self.a)

        # my derived way #

        # r #
        rnew = self.r + self.v_r * self.dt + 0.5 * self.a[0] * self.dt**2
        vrnew = self.v_r + self.a[0] * self.dt

        # z #
        znew = self.z + self.v_z * self.dt + 0.5 * self.a[2] * self.dt**2
        vznew = self.v_z + self.a[2] * self.dt

        # phi #
        phinew = self.phi + self.l / self.r**2 * self.dt
        vphinew = self.l / self.r

        self.r = rnew
        self.v_r = vrnew

        self.z = znew
        self.v_z = vznew

        self.phi = phinew
        self.v_phi = vphinew
        self.x = [self.r, self.phi, self.z]
        self.v = [self.v_r, self.v_phi, self.v_z]

        # compute dv_dr #
        v2 = self.norm2d(self.v_hist[-1])
        self.delta_r = self.dist2d(self.x, self.x_hist[-1])
        self.vtot = self.norm2d(self.v)
        dvr = self.v_r_hist[-1] - self.v_r
        dvz = self.v_z_hist[-1] - self.v_z
        dvt = (self.v[0] * dvr + self.v[2] * dvz) / v2
        self.dvt_hist.append(dvt)
        self.v2_hist.append(v2)
        if (abs(dvt) < 0.01 * v2):
            self.dv_dr = dvt / self.delta_r
        else:
            self.dv_dr = (self.vtot - v2) / self.delta_r
        self.dv_dr_hist.append(self.dv_dr)

        # append to history #

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

        # spherical radius velocity #

        self.v_T = np.sqrt(self.v_r ** 2 + self.v_z**2)
        self.v_T_hist.append(self.v_T)

        # finally update time #
        self.t = self.t + self.dt
        self.t_hist.append(self.t)

    def UpdatePositions(self):

        # compute acceleration vector #
        fg = self.Force_gravity()
        fr = self.Force_radiation()
        self.a = fg

        if('gravityonly' in self.modes):
            self.a += 0
        else:
            self.a += fr

        self.Fgrav.append(fg)
        self.Frad.append(fr)
        self.a_hist.append(self.a)

        r = self.r

        dx = np.asarray(self.v) * self.dt + 0.5 * \
            np.asarray(self.a) * self.dt**2.
        rnew = np.sqrt((r + dx[0])**2. + (dx[1])**2.)
        dphi = np.arctan2(dx[1], r + dx[0])

        self.r = rnew
        self.phi += dphi
        self.z += dx[2]

        vrn = self.v_r + self.a[0] * self.dt
        vphi = self.v_phi
        dvz = self.a[2] * self.dt
        sin_dphi = dx[1] / rnew
        cos_dphi = (r + dx[0]) / rnew

        dvr1 = self.v_r * (cos_dphi - 1.0)
        dvr2 = self.a[0] * self.dt * cos_dphi + vphi * sin_dphi
        dvr = dvr1 + dvr2

        dvp1 = vphi * (cos_dphi - 1.0)
        dvp2 = - vrn * sin_dphi
        dvp = dvp1 + dvp2
        self.v_r += dvr
        self.v_phi += dvp
        self.v_z += dvz
        self.x = [self.r, self.phi, self.z]
        self.v = [self.v_r, self.v_phi, self.v_z]

        # compute dv_dr #
        v2 = self.norm2d(self.v_hist[-1])
        self.delta_r = self.dist2d(self.x, self.x_hist[-1])
        self.vtot = self.norm2d(self.v)
        dvt = (self.v[0] * dvr + self.v[2] * dvz) / v2
        if (abs(dvt) < 0.01 * v2):
            self.dv_dr = dvt / self.delta_r
        else:
            self.dv_dr = (self.vtot - v2) / self.delta_r
        self.dv_dr_hist.append(self.dv_dr)

        # append to history #

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

        # spherical radius velocity #

        self.v_T = np.sqrt(self.v_r ** 2 + self.v_z**2)
        self.v_T_hist.append(self.v_T)

        # finally update time #
        self.t = self.t + self.dt
        self.t_hist.append(self.t)

    def UpdateRadiation(self):

        self.opacity = self.OpacityLaw(self.rho * self.mu * self.m_p, self.T)
        self.opacity_hist.append(self.opacity)
        self.Update_Tau_C()
        self.tau_c_hist.append(self.tau_c)
        # print(self.tau_c)
        self.Update_Tau_UV()
        self.tau_UV_hist.append(self.tau_UV)
        self.Update_Tau_X()
        self.tau_X_hist.append(self.tau_X)
        self.Update_xi()
        self.xi_hist.append(self.xi)
        self.Update_EffectiveOpticalDepth()
        self.Update_force_multiplier()

    def Step(self):
        # update positions and velocities #
        if ('originaltimestep' in self.modes):
            self.UpdatePositions()
        else:
            self.UpdatePositionsMyWay()
        # self.OldUpdatePositions()
        #print("Positions updated\n")
        # update radiation field #
        self.UpdateRadiation()
        #print("Radiation updated\n")

    # @profile(immediate=True)

    def iterate(self, niter=5000, debug=False):
        results = open(
            self.save_dir +
            "/results_" +
            "%06.1f" %
            self.r_0 +
            ".txt",
            "w")
        results.write(
            "R\tPHI\tZ\tv_R\tv_PHI\tv_Z\tv_esc\tv_T\ta_r\ta_z\tIr\tIz\n")
        for it in tqdm(range(0, niter)):
            self.Step()
            self.it = it
            self.iter.append(it)
            if (it == 99):
                # update time step #
                self.dt = self.dt * 10.

            # termination condition for a failed wind#
            if( ((self.z <= self.z_0) and (self.v_z < 0.0)) or ((self.z < 0.2 * np.max(self.z_hist)) and (self.v_z < 0.0)) ):
                print("Failed wind! \n")
                break
            
            # record when streamline escapes #
            if(self.v_T > self.v_esc() and (not self.escaped)):
                self.escaped = True
            a_t = np.sqrt(self.a[0]**2 + self.a[2]**2)

            #termination condition for an escaped wind #
            if(self.escaped and a_t < 1e-8):
                print("Wind escaped")
                break

            #save results every 10 iterations #
            if(it % 10 == 0):
                results.close()
                results = open(
                    self.save_dir + "/results_" + "%06.1f" %
                    self.r_0 + ".txt", "a")
            results.write("%e \t %e \t %e \t %e \t %e \t %e \t %e \t %e \t %e \t %e \t %e \t %e \n" % (self.r,
                                                                                                       self.phi,
                                                                                                       self.z,
                                                                                                       self.v_r,
                                                                                                       self.v_phi,
                                                                                                       self.v_z,
                                                                                                       self.v_esc(),
                                                                                                       self.v_T,
                                                                                                       self.a[0],
                                                                                                       self.a[2],
                                                                                                       self.int_hist[-1][0],
                                                                                                       self.int_hist[-1][1]))

        results.close()
