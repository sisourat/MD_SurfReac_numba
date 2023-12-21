import json
import numpy as np
from utils import angtoau, evtoau, fstoau
from scipy.constants import pi, e, m_u, m_e

def read_input(inputfile):
    # Opening JSON file
    f = open(inputfile)
    # returns JSON object as
    # a dictionary
    data = json.load(f)
    # Closing file
    f.close()

#Parameters for surface-atom potential
    eunit = data['potentialA']['eunit']
    runit = data['potentialA']['runit']
    D_a = data['potentialA']['D_a']
    Delta_a = data['potentialA']['Delta_a']
    alpha_a = data['potentialA']['alpha_a']
    if(eunit.lower()=='ev'):
        D_a = D_a * evtoau
    elif(not eunit.lower()=='au'):
        raise TypeError("Only ev or au are allowed for energy unit")
    if(runit.lower()=='angstrom'):
        alpha_a = alpha_a / angtoau
    elif(not runit.lower()=='au'):
        raise TypeError("Only angstrom or au are allowed for length unit")

# Parameters for atom-atom potential
    eunit = data['potentialM']['eunit']
    runit = data['potentialM']['runit']
    D_m = data['potentialM']['D_m']
    Delta_m = data['potentialM']['Delta_m']
    alpha_m = data['potentialM']['alpha_m']
    if (eunit.lower() == 'ev'):
        D_m = D_m * evtoau
    elif (not eunit.lower() == 'au'):
        raise TypeError("Only ev or au are allowed for energy unit")
    if (runit.lower() == 'angstrom'):
        alpha_m = alpha_m / angtoau
    elif (not runit.lower() == 'au'):
        raise TypeError("Only angstrom or au are allowed for length unit")

# Masses of the atoms
    munit = data['masses']['munit']
    mt = data['masses']['mt'] * m_u / m_e
    mp = data['masses']['mp'] * m_u / m_e
    if (not munit.lower() == 'amu'):
        raise TypeError("Only amu is allowed for mass unit")

    runit = data['sysparam']['runit']
    r_rec = data['sysparam']['r_rec']
    r0 = data['sysparam']['r0']
    z0 = data['sysparam']['z0']
    if (runit.lower() == 'angstrom'):
        r_rec = r_rec * angtoau
        r0 = r0 * angtoau
        z0 = z0 * angtoau
    elif (not runit.lower() == 'au'):
        raise TypeError("Only angstrom or au are allowed for length unit")

    mtot = mt + mp
    mu = mt * mp / mtot
    k_a = 2.0 * D_a * alpha_a ** 2
    omega_a = np.sqrt(k_a / mt)
    # Parameters for the dynamics

    try:
        tunit =  data['param']['tunit']
        tmax =  data['param']['tmax']
        dt =  data['param']['dt']
        dz = data['param']['dz']
        dZ = data['param']['dZ']
        drho = data['param']['drho']
        ntraj = int(data['param']['ntraj'])

        if (tunit.lower() == 'fs'):
            tmax = tmax*fstoau
            dt = dt*fstoau
        elif (not tunit.lower() == 'au'):
            raise TypeError("Only fs or au are allowed for time unit")

        if (runit.lower() == 'angstrom'):
            dz = dz * angtoau
            dZ = dZ * angtoau
            drho = drho * angtoau
        elif (not runit.lower() == 'au'):
            raise TypeError("Only angstrom or au are allowed for length unit")


    except:
        print("Some numerical parameters are not given, use default values")
        pass

    tlist = np.arange(0, tmax + 1, dt)

    eunit = data['dynamics']['eunit']
    ecoll = data['dynamics']['ecoll']
    if (eunit.lower() == 'ev'):
        ecoll = ecoll * evtoau
    elif (not eunit.lower() == 'au'):
        raise TypeError("Only angstrom or au are allowed for length unit")

    runit = data['dynamics']['runit']
    nb = data['dynamics']['nb']
    bmin = data['dynamics']['bmin']
    bmax = data['dynamics']['bmax']
    zp_i = data['dynamics']['zp_i']
    if (runit.lower() == 'angstrom'):
        zp_i = zp_i * angtoau
        bmin = bmin * angtoau
        bmax = bmax * angtoau
    elif (not runit.lower() == 'au'):
        raise TypeError("Only angstrom or au are allowed for length unit")

    blist = np.linspace(bmin,bmax,nb)

    return {"D_a":  D_a, "Delta_a":  Delta_a, "alpha_a":  alpha_a,
            "D_m":  D_m, "Delta_m":  Delta_m, "alpha_m":  alpha_m,
            "k_a": k_a, "omega_a": omega_a,
            "mt":   mt, "mp": mp, "mtot": mtot, "mu": mu, "r0": r0, "z0": z0,
            "tmax": tmax, "dt": dt, "dz": dz, "dZ": dZ, "drho": drho, "ntraj": ntraj,
            "blist": blist, "ecoll": ecoll, "zp_i": zp_i, "tlist": tlist, "r_rec": r_rec}
