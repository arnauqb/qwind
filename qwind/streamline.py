import numpy as np
import scipy.integrate
from qwind import utils
from qwind import constants as const

# check backend to import appropiate progress bar #
def tqdm_dump(array):
    return array
backend = utils.type_of_script()
if(backend == 'jupyter'):
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm
    #tqdm = tqdm_dump
IT = 1
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
            dt = 4.096 / 10.
            ):
        """
        Parameters
        ----------
        parent : object
            Parents class (wind object), to inherit global continueproperties.
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
        
        # black hole and disc variables #
        self.a = np.array([0, 0, 0])  # / u.s**2
        self.T = T  # * u.K
        self.v_th = self.wind.thermal_velocity(self.T) 
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
        self.tau_x = self.radiation.optical_depth_x(self.r, self.z, self.r_0, self.tau_dr, self.tau_dr_0, self.wind.rho_shielding)

        self.tau_eff = 0
        self.fm = 0
        self.xi = self.radiation.ionization_parameter(self.r, self.z, self.tau_x, self.wind.rho_shielding)#self.wind.Xi(self.d, self.z / self.r) 

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
        self.tau_x_hist = [self.tau_x]
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
        if(self.v_z < 0):
            self.rho_hist.append(self.rho)
            return self.rho
        radial = (self.r / self.r_0) ** (-2.)
        v_ratio = self.v_z_0 / np.linalg.norm(np.asarray(self.v)[[0,2]]) 
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


    def UpdatePositionsOld(self):
        
        # compute acceleration vector #
        fg = self.Force_gravity()
        fr = self.radiation.force_radiation(self.r, self.z, self.fm, self.tau_uv)
        self.a = fg + fr
        self.Fgrav.append(fg)
        self.Frad.append(fr)
        self.a_hist.append(self.a)
        
        r = self.r
        
        dx = np.asarray(self.v) * self.dt + 0.5 * np.asarray(self.a) * self.dt**2.
        rnew = np.sqrt( (r + dx[0])**2. + (dx[1])**2. )
        dphi = np.arctan2( dx[1] , r+dx[0]) 
        
        self.r = rnew
        self.phi += dphi
        self.z += dx[2]

        # debug
        #global IT
        #self.r = self.r_0 + IT * 0.1
        #self.z = 1 + IT * 0.1
        #IT += 1

        vrn  = self.v_r  +  self.a[0] * self.dt;
        vphi =  self.v_phi;
        dvz = self.a[2] * self.dt;
        sin_dphi = dx[1] / rnew;
        cos_dphi = ( r + dx[0] ) / rnew;

        dvr1 = self.v_r * ( cos_dphi - 1.0 );
        dvr2 = self.a[0] * self.dt  * cos_dphi + vphi * sin_dphi;
        dvr = dvr1 + dvr2;
        #print("dvr1: %e "%dvr1)
        #print("dvr2: %e "%dvr2)
        #print("vphi: %e"%vphi)
        #print("a_r: %e"%self.a[0])
        #print("cos: %e "%cos_dphi)
        #print("sin: %e "%sin_dphi)
        
        dvp1 = vphi * ( cos_dphi - 1.0 );
        dvp2 =  - vrn  * sin_dphi;
        dvp = dvp1 + dvp2;
        self.v_r   += dvr;
        self.v_phi += dvp;
        self.v_z   += dvz;
        self.x = [self.r, self.phi, self.z]
        self.v = [self.v_r, self.v_phi, self.v_z]

        # compute dv_dr #
        v2 = np.linalg.norm(np.asarray(self.v_hist[-1])[[0,2]]) #
        #v2 = self.wind.norm2d(self.v_hist[-1])
        self.delta_r = np.linalg.norm(np.asarray(self.x)[[0,2]] - np.asarray(self.x_hist[-1])[[0,2]])#self.wind.dist2d(self.x, self.x_hist[-1])#
        #print(self.x, print(self.x_hist[-1]))
        self.vtot = np.linalg.norm(np.asarray(self.v)[[0,2]]) #self.wind.norm2d(self.v) #
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
        
    def norm2d(self, vector):
        return np.sqrt(vector[0] ** 2 + vector[-1] ** 2)

    def UpdatePositions(self):
        """
        Updates position of streamline.
        """
        # compute acceleration vector #
        fg = self.Force_gravity()
        fr = self.radiation.force_radiation(self.r, self.z, self.fm, self.tau_uv)
        self.Fgrav.append(fg)
        self.Frad.append(fr)
        self.a = np.copy(fg)
        if('gravityonly' in self.wind.modes):
            self.a += 0.
        else:
            self.a += fr

        self.a[0] += self.l**2 / self.r**3
        self.a_hist.append(self.a)

        # update r #
        rnew = self.r + self.v_r * self.dt + 0.5 * self.a[0] * self.dt**2
        vrnew = self.v_r + self.a[0] * self.dt
        #print("V: %e \n Fr_grava: %e \n"%(self.v_r, self.a[0] ))

        # update z #
        znew = self.z + self.v_z * self.dt + 0.5 * self.a[2] * self.dt**2
        vznew = self.v_z + self.a[2] * self.dt

        # update phi #
        phinew = self.phi + self.l / self.r**2 * self.dt
        vphinew = self.l / self.r

        self.r = rnew
        #self.r = self.r_debug
        self.v_r = vrnew
        #global IT
        #self.v_r = 0.0002 * IT

        self.z = znew
        #self.z = self.z_debug
        # debug
        #self.r = self.r_0 + IT * 0.1
        #self.z = 1 + IT * 0.1
        #IT += 1
        self.v_z = vznew
        #self.v_z = 0.001 + 0.0002 * IT
        self.phi = phinew
        self.v_phi = vphinew
        self.x = [self.r, self.phi, self.z]
        self.v = [self.v_r, self.v_phi, self.v_z]

        # compute dv_dr #
        v2 = np.linalg.norm(np.asarray(self.v_hist[-1])[[0,2]]) #
        #v2 = self.wind.norm2d(self.v_hist[-1])
        self.delta_r = np.linalg.norm(np.asarray(self.x)[[0,2]] - np.asarray(self.x_hist[-1])[[0,2]])#self.wind.dist2d(self.x, self.x_hist[-1])#
        #print(self.x, print(self.x_hist[-1]))
        self.vtot = np.linalg.norm(np.asarray(self.v)[[0,2]]) #self.wind.norm2d(self.v) #
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

        #self.v_r = self.v_r_debug
        #self.v_z = self.v_z_debug
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
        #print("rho: %e \n"%self.rho)
        self.tau_dr = self.wind.tau_dr(self.rho)
        self.tau_dr_hist.append(self.tau_dr)
        
        self.tau_eff = self.radiation.sobolev_optical_depth(self.tau_dr, self.dv_dr)
        self.tau_eff = self.tau_eff
        self.dr_e_hist.append(self.tau_eff/self.tau_dr)
        self.tau_eff_hist.append(self.tau_eff)
        #print("dr_e: %e"%self.dr_e_hist[-1])
        #print(self.tau_eff)
        

        self.tau_uv = self.radiation.optical_depth_uv(self.r, self.z, self.r_0, self.tau_dr, self.tau_dr_0)
        self.tau_uv_hist.append(self.tau_uv)

        self.tau_x = self.radiation.optical_depth_x(self.r, self.z, self.r_0, self.tau_dr, self.tau_dr_0, self.wind.rho_shielding)
        self.tau_x_hist.append(self.tau_x)
        self.xi = self.radiation.ionization_parameter(self.r,self.z, self.tau_x, self.rho)
        self.xi_hist.append(self.xi)
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


    def iterate(self, niter=5000):
        """
        Iterates the streamline
        
        Paramters
        ---------
        
        niter : int
            Number of iterations
        """
        #debug 
        #trajectory = np.loadtxt("trajectory.txt")
        #r_range, z_range, v_r_range, v_z_range = trajectory
        for it in tqdm(range(0, niter)):
            #execute time step #
            #self.r_debug = r_range[it+1]
            #self.z_debug = z_range[it+1]
            #self.v_r_debug = v_r_range[it+1] 
            #self.v_z_debug = v_z_range[it+1] 
            self.Step()
            # record number of iterations #
            self.it = it
            self.iter.append(it)
            
            if (it == 99):
                # update time step #
                print("updating time step from %f to %f"%(self.dt, self.dt*10))
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
