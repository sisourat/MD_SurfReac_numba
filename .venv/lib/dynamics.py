import sys
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import scipy.integrate as si
from icecream import ic

from potential import pot3d as pot
from utils import angtoau, evtoau, fstoau

def initial_gen_target(zt, ekt, freq, mass, number):
    # Generate a number of initial conditions (zt,vzt)_0 for target atom
    vz = np.sqrt(2.0 * ekt / mass)
    stdz = np.sqrt(1.0 / (2.0 * freq * mass))
    stdv = np.sqrt((2.0 * freq) / mass)
    z_0 = np.random.normal(zt, stdz, number)
    vz_0 = (2 * np.random.randint(0, 2, number) - 1) * np.random.normal(vz, stdv, number)
    return z_0, vz_0

def initial_gen_proj(zp, b, theta, ekp, delta_rho, delta_ek, mass, number):
    # Generate a number of initial conditions (z2, rho, vz2, vrho)_0 for projectile atom
    z_0 = zp * np.ones(number)

    stdrho = delta_rho / 2.0
    rho_0 = np.random.normal(b, stdrho, number)

    stdek = delta_ek / 2.0
    ek_0 = np.abs(np.random.normal(ekp, stdek, number))
    # Absolute value to avoid negative Kinetic energies when ek_i<delta_ek (Non gaussian distrib tho)
    v_0 = np.sqrt(2 * ek_0 / mass)
    vz_0 = - v_0 * np.cos(theta)
    vrho_0 = - v_0 * np.sin(theta)
    return z_0, rho_0, vz_0, vrho_0

########################################################################################################################

def diff_sys(y0, tlist, mu, mtot, drho, dz, dZ, D_a, Delta_a, alpha_a, D_m, Delta_m, alpha_m, mt, mp, z0, r0):
    # Defines the Differential system that will be solved to find the trajectory
    # Takes y0 = (pos,vit) and computes its derivative dy0/dt = (vit,acc)
    rho, z, Z, vrho, vz, vZ = y0
    Arho = -(pot(rho + .5 * drho, z, Z, D_a, Delta_a, alpha_a, D_m, Delta_m, alpha_m, mt, mp, mtot, z0, r0)
        - pot(rho - .5 * drho, z, Z, D_a, Delta_a, alpha_a, D_m, Delta_m, alpha_m, mt, mp, mtot, z0, r0)) / (mu * drho)
    Az = -(pot(rho, z + .5 * dz, Z, D_a, Delta_a, alpha_a, D_m, Delta_m, alpha_m, mt, mp, mtot, z0, r0)
          - pot(rho, z - .5 * dz, Z, D_a, Delta_a, alpha_a, D_m, Delta_m, alpha_m, mt, mp, mtot, z0, r0)) / (mu * dz)
    AZ = -(pot(rho, z, Z + .5 * dZ, D_a, Delta_a, alpha_a, D_m, Delta_m, alpha_m, mt, mp, mtot, z0, r0)
         - pot(rho, z, Z - .5 * dZ, D_a, Delta_a, alpha_a, D_m, Delta_m, alpha_m, mt, mp, mtot, z0, r0)) / (mtot * dZ)

    return np.array([vrho, vz, vZ, Arho, Az, AZ])

########################################################################################################################

def solv_traj(data, zp_0, vzp_0, rho_0, vrho_0, zt_0, vzt_0):
    t = data['tlist']
    mt = data['mt']
    mp = data['mp']
    mtot = data['mtot']
    mu = data['mu']
    drho = data['drho']
    dz = data['dz']
    dZ = data['dZ']

    D_a=data['D_a']
    Delta_a=data['Delta_a']
    alpha_a=data['alpha_a']
    D_m=data['D_m']
    Delta_m=data['Delta_m']
    alpha_m=data['alpha_m']
    r0 = data['r0']
    z0 = data['z0']

    r_rec = data['r_rec']
    recomb = 0

    rho_in = rho_0
    z_in = zp_0 - zt_0
    Z_in = (mp * zp_0 + mt * zt_0) / mtot
    vrho_in = vrho_0
    vz_in = vzp_0 - vzt_0
    vZ_in = (mp * vzp_0 + mt * vzt_0) / mtot

    y0 = np.array([rho_in, z_in, Z_in, vrho_in, vz_in, vZ_in])
    y = si.odeint(diff_sys, y0, t, args=(mu, mtot, drho, dz, dZ, D_a, Delta_a, alpha_a, D_m, Delta_m,
                                         alpha_m, mt, mp, z0, r0))

    rho, z, Z, vrho, vz, vZ = y.T
    zt = Z - (mp / mtot) * z
    zp = Z + (mt / mtot) * z
    r = np.sqrt(rho * rho + z * z)
    r_fin = r[-1]
    ic(r_fin,r_rec)
    if r_fin < r_rec:
        recomb = 1

    return (y, recomb)

def data_traj(data, y, t):
    mt = data['mt']
    mp = data['mp']
    mtot = data['mtot']
    mu = data['mu']
    rho, z, Z, vrho, vz, vZ = y.T
    zt = Z - (mp / mtot) * z
    zp = Z + (mt / mtot) * z
    r = np.sqrt(rho * rho + z * z)

    eZ = 0.5 * mtot * vZ * vZ
    er = 0.5 * mu * (vz * vz + vrho * vrho)
    potential = pot(rho, z, Z, D_a=data['D_a'],
              Delta_a=data['Delta_a'],
              alpha_a=data['alpha_a'] ,
              D_m=data['D_m'] ,
              Delta_m=data['Delta_m'],
              alpha_m=data['alpha_m'],
              mt=data['mt'], mp = data['mp'],
              mtot = data['mtot'],
              r0 = data['r0'],
              z0 = data['z0'])

    return (rho, z, Z, zt, zp, r, eZ, er, potential)

def solv_ntraj(data, zp_0_vect, vzp_0_vect, rho_0_vect, vrho_0_vect, zt_0_vect, vzt_0_vect):

    ecoll = data['ecoll']
    t = data['tlist']
    ntraj = data['ntraj']
    n_rec = 0

    eZ_fin = np.zeros(ntraj)
    er_fin = np.zeros(ntraj)
    evib_fin = np.zeros(ntraj)

    for i in range(ntraj):
        zt_0 = zt_0_vect[i]
        vzt_0 = vzt_0_vect[i]

        zp_0 = zp_0_vect[i]
        rho_0 = rho_0_vect[i]
        vzp_0 = vzp_0_vect[i]
        vrho_0 = vrho_0_vect[i]
        y, recomb = solv_traj(data, zp_0, vzp_0, rho_0, vrho_0, zt_0, vzt_0)

        n_rec += recomb

        rho, z, Z, zt, zp, r, eZ, er, V = data_traj(data, y, t)

        eZ_fin[i] = eZ[-1] * recomb
        er_fin[i] = er[-1] * recomb
        evib_fin[i] = (er[-1] + V[-1] - ecoll + data['D_m']) * recomb

    if n_rec == 0:
        eZ_fin_avg = 0.
        er_fin_avg = 0.
        evib_fin_avg = 0.
    else:
        eZ_fin_avg = eZ_fin.sum() / n_rec
        er_fin_avg = er_fin.sum() / n_rec
        evib_fin_avg = evib_fin.sum() / n_rec

    return (n_rec, eZ_fin_avg, er_fin_avg, evib_fin_avg)