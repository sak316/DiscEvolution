import astropy.units as u
import numpy as np

# constants
sig_SB = 5.6704e-5 * u.erg / u.cm**2 / u.s / u.K**4 # Stefan-Boltzmann constant, cgs
G = 6.67430e-8 * u.cm**3 / (u.g * u.s**2) # Gravitational constant, cgs

class DiskWindEvolution_Standalone(object):
    '''This class contains the equations for the disk wind evolution.
    Mainly stuff from Chambers et al. 2019, supplemented with Alessi et al. 2022.'''
    def __init__(self, sigma0, r0, T0, v0, fw, K, Tevap, rexp, k0):

        super(DiskWindEvolution_Standalone, self).__init__()

        self.sigma0 = sigma0
        self.r0 = r0
        self.T0 = T0
        self.v0 = v0
        self.fw = fw
        self.K = K
        self.Tevap = Tevap
        self.rexp = rexp
        self.k0 = k0
    
    def __call__(self, star, R, t):
        '''Returns the surface density, temperature, total mass, and accretion rate at a given time t and radius R.'''
        Mstar = star.M * u.Msun

        s0 = np.sqrt(self.rexp / self.r0)

        # part of Equation 37
        V = self.Tevap/self.T0

        Atop = 9*(1-self.fw)*G*Mstar*self.k0*(self.sigma0**2)*self.v0
        Abottom = 32*sig_SB*(self.r0**2)*(self.T0**4)
        A = np.sqrt(Atop/Abottom)

        x = (R/self.r0)**(1/2)

        # Equation 39
        p0 = A*(V**(1/2)) * ((A**2 + 1) / (A**2 + V**3))**(1/6)
        J = self.fw / (1-self.fw)
        n = -1 - (2/5) * ((1+J)**2 + 8*J*self.K)**(1/2)
        b = ((1-J)/2) + (1/2) * ((1+J)**2 + 8*J*self.K)**(1/2)
        tau = (8 * self.r0 * s0**(5/2)) / (25 * self.v0 * (1-self.fw))

        # Equation 38
        p1 = p0 * (1 + (t/tau))**n * (x**b)
        p2 = np.exp((1/s0)**(5/2) - (x/s0)**(5/2)*(1 + t/tau)**(-1))
        p = p1 * p2

        # Equation 36 
        sigma_1 = (self.sigma0/A) * p * x**(-5/2)
        sigma_2_top = 1 + V**(-2)*p*(x**(-9/2))
        sigma_2_bottom = 1 +p*(x**(-5/2))
        sigma = sigma_1 * (sigma_2_top/sigma_2_bottom)**(1/4)

        # Equation 37 more
        sig = A * sigma / self.sigma0

        # Equation 36 Temperature
        Ttop = sig**2 + 1
        Tbottom = sig**2 + (V**(3))*(x**(3))
        T = self.Tevap * (Ttop/Tbottom)**(1/3)


        # Calculating the total mass and accretion rate using equations from Alessi et al. 2022

        dA = np.pi * (R[1:]**2 - R[:-1]**2)
        dM = sigma[1:] * dA
        Mtot = np.sum(dM)


        v_in = self.v0 * (T[0]/self.T0)**(1/2) # alessi equation 13
        Macc = 2*np.pi*R[0]*sigma[0]*v_in # alessi equation 12


        # Fixing the units
        sigma = sigma.decompose().to(u.g / u.cm**2)
        T= T.decompose().to(u.K)
        Mtot = Mtot.decompose().to(u.Msun)
        Macc = Macc.decompose().to(u.Msun/u.yr)

        return sigma, T, Mtot, Macc