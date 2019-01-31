import numpy as np
import scipy.integrate
import utils
import constants as const
from wind import wind 
from aux_numba import integrate

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
            parent,
            r_0=375.,
            z_0=1.,
            rho_0=2e8,
            T=2e6,
            v_z_0=5e7,
            v_r_0=0.,
            dt=4.096 / 10,
            fm_mean = 0
            ):
        """
        Parameters
        ----------
        parent : object
            Parents class (wind object), to inherit global properties.
        r_0 : float
            Initial radius in Rg units.
        z_0: float
            Initial height in Rg units.
        rho_0 : float
            Initial number density. Units of 1/cm^3.
        T : float
            Initial stramline temperature.
        v_r_0 : float
            Initial radial velocity in units of cm/s.
        v_z_0 : float
            Initial vertical velocity in units of cm/s.
        dt : float
            Timestep in units of Rg/c.
        """
        
    
        self.wind = parent
        self.radiation = radiation_class

        self.fm_mean = fm_mean
        
        # black hole and disc variables #
        self.a = [0, 0, 0]  # / u.s**2
        self.T = T  # * u.K
        self.v_th = self.wind.ThermalVelocity(self.T) 
        self.rho_0 = rho_0
        self.rho = self.rho_0
        
        ## position variables ##
        ## coordinates of particle are (R, phi, z) ##
        ## all positions are in units of Rg, all velocities in units of c. ##
        self.dt = dt  # units of  Rg / c
        self.r = r_0
        self.phi = 0
        self.z = z_0
        self.x = [self.r, self.phi, self.z]
        self.d = np.sqrt(self.r**2 + self.z**2)
        self.t = 0  # in seconds
        self.r_0 = r_0
        self.z_0 = z_0
        self.v_r = v_r_0 / const.c   
        self.v_r_hist = [self.v_r]
        self.v_phi = self.wind.v_kepler(r_0)
        self.l = self.v_phi * self.r  # initial angular momentum
        self.v_phi_0 = self.v_phi
        self.v_z_0 = v_z_0 / const.c  
        self.v_z = self.v_z_0  
        self.v = [self.v_r, self.v_phi, self.v_z]
        self.v_T = np.sqrt(self.v_z ** 2 + self.v_r ** 2)
        self.dv_dr = 0  
        self.dr_e = 0  
        self.escaped = False # this variable tracks whether the wind has reached the escape velocity

        
        ## optical depths ##
        self.tau_dr = self.wind.tau_dr( self.rho) 
        self.tau_dr_hist= [self.tau_dr]
        self.tau_dr_0 = self.tau_dr
        self.tau_dr_shielding = self.wind.tau_dr(self.wind.rho_shielding)

        self.tau_uv = self.radiation.optical_depth_uv(self.r, self.z, self.r_0, self.tau_dr, self.tau_dr_0)
        self.tau_sx = self.radiation.optical_depth_soft_xray(self.r, self.z, self.r_0, self.tau_dr, self.tau_dr_0, self.wind.rho_shielding)
        self.tau_hx = self.radiation.optical_depth_hard_xray(self.r, self.z, self.r_0, self.tau_dr, self.tau_dr_0, self.wind.rho_shielding)

        self.tau_eff = 0
        self.fm = 1
        self.xi = self.radiation.ionization_parameter(self.r, self.z, self.tau_sx, self.wind.rho_shielding)#self.wind.Xi(self.d, self.z / self.r) 

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
        self.v_T_hist = [self.v_T]

        # radiation related histories #
        self.rho_hist = [self.rho]
        self.tau_dr_hist = [self.tau_dr]
        self.dvt_hist = [0]
        self.v2_hist=[0]
        self.dv_dr_hist = [0]
        self.dr_e_hist = [self.dr_e]
        self.tau_uv_hist = [self.tau_uv]
        self.tau_x_hist = [self.tau_sx]
        self.tau_sx_hist = [self.tau_sx]
        self.tau_hx_hist = [self.tau_hx]
        self.tau_eff_hist = [0]
        self.taumax_hist = []
        self.fm_hist = [1]
        self.xi_hist = [self.xi]

        #force histories #
        self.a_hist = [self.a]
        

    
    ###############
    ## streaming ##
    ###############

    def UpdateDensity(self):
        """
        Updates density of streamline.
        """
        #if(self.v_z < 0):
        #    self.rho_hist.append(self.rho)
        #    return self.rho
        radial = (self.r / self.r_0) ** (-2.)
        v_ratio = self.v_z_0 / np.linalg.norm(np.asarray(self.v)[[0,2]]) ##self.wind.norm2d(self.v)# 
        self.rho = self.rho_0 * radial * v_ratio
        # save to grid #
        self.rho_hist.append(self.rho)
        return self.rho

    def Force_gravity(self):
        """
        Computes gravity force.
        """
        
        array = np.asarray([self.r / self.d, 0., self.z / self.d])
        grav = - 1. / (self.d**2) * array
        return grav

   ## kinematics ##

    def UpdatePositions(self):
        """
        Updates position of streamline.
        """
        # compute acceleration vector #
        fg = self.Force_gravity()
        fr = self.radiation.force_radiation(self.r, self.z, self.fm, self.tau_uv, self.fm_mean)#self.Force_radiation()
        self.Fgrav.append(fg)
        self.Frad.append(fr)
        self.a = fg
        if('gravityonly' in self.wind.modes):
            self.a += 0
        else:
            self.a += fr

        self.a[0] += self.l**2 / self.r**3
        self.a_hist.append(self.a)

        # update r #
        rnew = self.r + self.v_r * self.dt + 0.5 * self.a[0] * self.dt**2
        vrnew = self.v_r + self.a[0] * self.dt

        # update z #
        znew = self.z + self.v_z * self.dt + 0.5 * self.a[2] * self.dt**2
        vznew = self.v_z + self.a[2] * self.dt

        # update phi #
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
        v2 = np.linalg.norm(np.asarray(self.v_hist[-1])[[0,2]]) #
        #v2 = self.wind.norm2d(self.v_hist[-1])
        self.delta_r = np.linalg.norm(np.asarray(self.x)[[0,2]] - np.asarray(self.x_hist[-1])[[0,2]])#self.wind.dist2d(self.x, self.x_hist[-1])#
        self.vtot = np.linalg.norm(np.asarray(self.v)[[0,2]]) #self.wind.norm2d(self.v) #
        dvr = self.v_r_hist[-1] - self.v_r
        dvz = self.v_z_hist[-1] - self.v_z
        dvt = (self.v[0] * dvr + self.v[2] * dvz) / v2
        self.dvt_hist.append(dvt)
        self.v2_hist.append(v2)
        if (abs(dvt) < 0.01 * v2):
            self.dv_dr = dvt / self.delta_r
        else:
            self.dv_dr =(self.vtot - v2) / self.delta_r
            
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
        """
        Updates all parameters related to the radiation field, given the new streamline position.
        """

        self.rho = self.UpdateDensity()
        self.tau_dr = self.wind.tau_dr(self.rho)
        self.tau_dr_hist.append(self.tau_dr)
        
        self.tau_eff = self.radiation.sobolev_optical_depth(self.tau_dr, self.dv_dr)
        self.dr_e_hist.append(self.tau_eff/self.tau_dr)
        self.tau_eff_hist.append(self.tau_eff)
        
        self.xi = self.radiation.ionization_parameter(self.r,self.z, self.tau_sx, self.wind.rho_shielding)
        self.xi_hist.append(self.xi)

        self.tau_uv = self.radiation.optical_depth_uv(self.r, self.z, self.r_0, self.tau_dr, self.tau_dr_0)
        self.tau_uv_hist.append(self.tau_uv)

        self.tau_sx = self.radiation.optical_depth_soft_xray(self.r, self.z, self.r_0, self.tau_dr, self.tau_dr_0, self.wind.rho_shielding)
        self.tau_hx = self.radiation.optical_depth_hard_xray(self.r, self.z, self.r_0, self.tau_dr, self.tau_dr_0, self.wind.rho_shielding)
        self.tau_sx_hist.append(self.tau_sx)
        self.tau_x_hist.append(self.tau_sx)
        self.tau_hx_hist.append(self.tau_hx)
        
        self.fm = self.radiation.force_multiplier(self.tau_eff, self.xi)
        self.fm_hist.append(self.fm)
        
        return 0

    def Step(self):
        """
        Performs time step.
        """
        # update positions and velocities #
        self.UpdatePositions()
        # update radiation field #
        self.UpdateRadiation()

    # @profile(immediate=True)

    def iterate(self, niter=5000):
        """
        Iterates the streamline
        
        Paramters
        ---------
        
        niter : int
            Number of iterations
        """
        
        results = open(
            self.wind.save_dir +
            "/results_" +
            "%06.1f" %
            self.r_0 +
            ".txt",
            "w")
        results.write(
            "R\tPHI\tZ\tv_R\tv_PHI\tv_Z\tv_esc\tv_T\ta_r\ta_z\tIr\tIz\n")
        for it in tqdm(range(0, niter)):
            #execute time step #
            self.Step()
            
            # record number of iterations #
            self.it = it
            self.iter.append(it)
            
            if (it == 99):
                # update time step #
                self.dt = self.dt * 10.

            # termination condition for a failed wind #
            if( ((self.z <= self.z_0) and (self.v_z < 0.0)) or ((self.z < 0.2 * np.max(self.z_hist)) and (self.v_z < 0.0)) ):
                print("Failed wind! \n")
                break
            
            # record when streamline escapes #
            if(self.v_T > self.wind.v_esc(self.d) and (not self.escaped)):
                self.escaped = True
            a_t = np.sqrt(self.a[0]**2 + self.a[2]**2)

            #termination condition for an escaped wind #
            if(self.escaped and a_t < 1e-8):
                print("Wind escaped")
                break
            if(self.d > 3000):
                print("out of grid \n")
                break
            #save results every 10 iterations #
            if(self.it % 100 == 0):
                results.close()
                results = open(self.wind.save_dir + "/results_" + "%06.1f" %self.r_0 + ".txt", "a")
                #results = ope
            #results.close()
            #results = open(self.wind.save_dir + "/results_" + "%4.2f" %self.r_0 + ".txt", "a")
            results.write("%e \t %e \t %e \t %e \t %e \t %e \t %e \t %e \t %e \t %e \t %e \t %e \n" % (self.r,
                                                                                                       self.phi,
                                                                                                       self.z,
                                                                                                       self.v_r,
                                                                                                       self.v_phi,
                                                                                                       self.v_z,
                                                                                                       self.wind.v_esc(self.d),
                                                                                                       self.v_T,
                                                                                                       self.a[0],
                                                                                                       self.a[2],
                                                                                                       self.radiation.int_hist[-1][0],
                                                                                                       self.radiation.int_hist[-1][1]))
        results.close()
