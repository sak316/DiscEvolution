import sys
import os

# Get the absolute path of the parent directory of the current script.
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Add the parent directory to sys.path.
sys.path.append(parent_dir)


from astropy import units as u
import numpy as np
import h5py
import time
import pickle
import pprint
import datetime

from DiscEvolution.grid import Grid
from DiscEvolution.star import SimpleStar
from DiscEvolution.eos import LocallyIsothermalEOS, IrradiatedEOS
from DiscEvolution.dust import SingleFluidDrift, DustGrowthTwoPop
from DiscEvolution.planet_formation import Planets, PebbleAccretionHill, Bitsch2015Model

from config import Config, Constants
from DiscEvolution.disc_winds import DiskWindEvolution

def calculate_alpha(Config, star):
    """
    Calculate the alpha viscosity parameter from the disk parameters
    """
    cs0_val = ((Constants.gamma)*(Constants.boltz)*Config.T0/(Constants.mu*Constants.mH))**0.5 # sound speed from ideal gas law at ref temp; cs^2=gamma*kt/(mu*mH)
    cs0_val = cs0_val.to(u.cm/u.s)
    Omega_ref = star.Omega_k(Config.r0.value) * (2*np.pi) / u.yr # keplerian frequency at reference radius, in proper units given docs of func
    alpha_turb = (1 - Config.fw)*Config.r0*Config.v0*Omega_ref / (cs0_val**2) 
    alpha_wind = Config.fw*Config.v0 / (cs0_val)
    alpha = alpha_turb 
    alpha = alpha.decompose()

    return alpha # placeholder value



def setup_model(Config):
    """
    Set up the disk model using the parameters in the Config class
    """
    grid = Grid(Config.rmin, Config.rmax, Config.nr)
    star = SimpleStar()

    alpha = calculate_alpha(Config, star)


    if Config.eos == "LocallyIsothermalEOS":
        # Placeholders until we can calcualte them
        h0 = 1/30 # aspect ratio at 1AU
        q = -0.25 #power law index of sound speed
        eos = LocallyIsothermalEOS(star, h0, q, alpha)

    elif Config.eos == "IrradiatedEOS":
        eos = IrradiatedEOS(star, alpha)

    
    eos.set_grid(grid)

    wind = DiskWindEvolution_Standalone(Config.sigma0, Config.r0, Config.T0, 
                                        Config.v0, Config.fw, Config.K, Config.Tevap, 
                                        Config.rexp, Config.k0)

    Sigma, _, _, _ = wind(star, grid.Rc*u.AU, 0.*u.Myr) # initial conditions for the accretion disk


    # Physics 

    disc = DustGrowthTwoPop(grid, star, eos, Config.initial_frac, Sigma=Sigma.value, feedback=Config.feedback)
    
    drift = SingleFluidDrift(settling=True)
    PebAcc = PebbleAccretionHill(disc)

    planets = Planets(Nchem = 0)
    for mass, semi_major_axis in Config.planet_params:
        planets.add_planet(0, semi_major_axis, mass, 0)


    

    planet_model = Bitsch2015Model(disc, pb_gas_f=0.0) # planet formation model

    return disc, drift, PebAcc, planets, planet_model, wind



def run_model(Config, model, output_dir):
    """
    Run the disk model using the parameters in the Config class
    """
    disc, drift, PebAcc, planets, planet_model, wind = model

    times_ref = []
    dust_ref = []
    gas_ref = []
    peb_Mdot = []
    largepop_size = []
    Rp, Mp, Menv = [], [], []

    times = np.arange(Config.t_initial, Config.t_final, Config.t_interval)

    t = 0
    n = 0

    # Find the index of the value closest to the reference radius
    index = np.abs(disc.grid.Rc - Config.r0.value).argmin()

    Peb_Onset = np.zeros(len(disc.grid.Rc)-1) # array to store pebble accretion onset; 


    # Get the current time
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    full_output_dir = f'{output_dir}/{current_time}_{Config.planet_params}'
    os.makedirs(full_output_dir, exist_ok=True)
    # os.makedirs(f'{output_dir}/{current_time}_{Config.planet_params}', exist_ok=True)
    


    for ti in times:
        while t < ti:
            dt = 0.5*drift.max_timestep(disc)
            # update all the physics, unsure how the order affects things
            new_sigma = wind(disc.star, disc.R*u.AU, t*u.yr)[0] # get the new surface density
            disc.set_surface_density(new_sigma.value) # update total disc surface density using the analytical solution, need to remove the units

            disc.update(dt) # updates the dust sizes i.e. perform grain growth. Also updates star and EOS
            drift(dt, disc) # update the dust fraction i.e. evolves the dust

            PebAcc.set_disc(disc) # set the disc for pebble accretion after disc has evolved
            PebAcc.update() # updates the pebble accretion rates

            planet_model.integrate(dt, planets)
            Rp.append(planets[0].R)
            Mp.append(planets[0].M_core)
            Menv.append(planets[0].M_env)


            t = np.minimum(t + dt, ti)
            n += 1

        # # calculate pebble accretion onset in Mearth

            # compute eta like in dust.py  , based on radial pressure gradient
            Om_k   = disc.Omega_k
            # Average to cell edges:        
            Om_kav  = 0.5*(Om_k      [1:] + Om_k      [:-1])
            rho = disc.midplane_gas_density
            dPdr = np.diff(disc.P) / disc.grid.dRc
            eta = - dPdr / (0.5*(rho[1:] + rho[:-1] + 1e-300)*Om_kav)

            Stokes = disc.Stokes(disc._a[1]) # Stokes number for large dust population
            
            Stokes_av = 0.5*(Stokes[1:] + Stokes[:-1])  # Average Stokes to cell edges
            Stokes_av = np.squeeze(Stokes_av)
            # remove last element of stokes_av from array
            Stokes_av = Stokes_av[:-1]

    
            Peb_Onset = 2.5e-4*(Stokes_av/0.1)*((eta/0.002)**3)*disc.star.M

            m_planet = Config.planet_params[0][0] # planet mass in earth masses
            r_planet = Config.planet_params[0][1] # planet semi-major axis in AU

            # append the values to the arrays
            times_ref.append(t)
            dust_ref.append(disc.Sigma_D[1][index])
            gas_ref.append(disc.Sigma_G[index])
            peb_Mdot.append(PebAcc.computeMdot(r_planet, m_planet)) # pebble mass accretion rate for 0.1 earth mass planet at 30 AU
            largepop_size.append(disc._a[1][index]) # dust size for large population

            if (n % 1000) == 0:
                print('Nstep: {}'.format(n))
                print('Time: {} '.format(t ))
                print('dt: {} '.format(dt))

        print('Nstep: {}'.format(n))
        print('Time: {} '.format(t))


        disc_data = {
            'R': disc.grid.Rc,
            'T': disc.T,
            'Sigma_G': disc.Sigma_G,
            'Sigma_D_small': disc.Sigma_D[0],
            'Sigma_D_large': disc.Sigma_D[1],
            'M_iso': PebAcc.M_iso(disc.grid.Rc),
            'Peb_Onset': Peb_Onset
        }


        with h5py.File(f'{full_output_dir}/disk_data_{t}.h5', 'a') as file:
            # Create datasets in the file
            for key, value in disc_data.items():
                file.create_dataset(key, data=np.array(value))



    sim_data = {
        'times': times_ref,
        'dust': dust_ref,
        'gas': gas_ref,
        'peb_Mdot': peb_Mdot,
        'largepop_size': largepop_size,
        'Rp': Rp,
        'Mp': Mp,
        'Menv': Menv
    }

    # Create a new HDF5 file in the new directory
    with h5py.File(f'{full_output_dir}/sim_data.h5', 'a') as file:
        # Create datasets in the file
        for key, value in sim_data.items():
            file.create_dataset(key, data=np.array(value))

    # Assuming config is an instance of your config class
    with open(f'{full_output_dir}/Config.pkl', 'wb') as file:
        pickle.dump(Config, file)

    #  also pickle the fully evolved pebble accretion model
    with open(f'{full_output_dir}/PebAcc.pkl', 'wb') as file:
        pickle.dump(PebAcc, file)



    # Create a text file with a human-readable representation of the Config object
    with open(f'{full_output_dir}/Config.txt', 'w') as file:
        file.write(pprint.pformat(vars(Config)))


if __name__ == '__main__':
    

    # Create a new directory for the output
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_dir = f'draz_plot_{current_time}'

    # three planets to go through
    R = [30]
    M = [0.01]
    # R = np.logspace(-1, 3, 3) # 10^1 to 10^100 :( fix
    # M = np.logspace(-8, 2, 3)

    planet_params = [(m, r) for (m, r) in zip(M, R)] # zip the two lists together
    # planet_params = [(m, r) for m in M for r in R]
    for p in planet_params:
        Config.planet_params = [p]
        print(Config.planet_params)

        model = setup_model(Config)
        run_model(Config, model, output_dir)  # Pass the output directory to the run_model function