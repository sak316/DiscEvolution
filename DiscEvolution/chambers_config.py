import astropy.units as u


class Constants:
    boltz = 1.3806e-16 *u.erg / u.K # boltzmann constant
    mH = 1.67e-24 *u.g # mass of hydrogen atom in grams
    G = 6.67430e-8 * u.cm**3 / (u.g * u.s**2)
    mu = 2.34 
    gamma = 1.4


class Config:

    # Chambers/Pudritz disk model parameters
    # -------------------------
    r0 = 1*u.au # Reference radius in AU
    sigma0 = 3450*u.g / u.cm**2 # Surface density at reference radius, g/cm^2
    T0 = 150 * u.K # temperature at reference radius, K
    v0 = 30 * u.cm / u.s # inflow velocity at reference radius, cm/s
    fw = 0 # fraction of v0 caused by disk wind
    K = 0 # controls mass loss rate via disk wind
    Tevap = 1500 * u.K # Dust evaporation temperature, K
    rexp = 15*u.au # Initial exponential turnover distance
    k0 = 0.1 * u.cm**2 / u.g # Dust opacity, cm^2/g

    # Grid parameters
    # -------------------------
    rmin = 0.05 # Minimum radius, AU
    rmax = 1000 # Maximum radius, AU
    nr = 1000 # Number of radial grid points

    # Simulation parameters
    # -------------------------
    t_initial = 0 # inital time, years
    t_final = 1e6 # final time, years
    t_interval = 1e4 # interval to output disk properties, years

    # Dust growth parameters
    # -------------------------
    initial_frac = 0.01
    feedback = False

    eos = "LocallyIsothermalEOS" # Equation of state to use

    # Planets
    # -------------------------
    planet_params = [(0.1, 30)] # ordered pairs of planet mass (in earth masses) and semi-major axis (in AU)