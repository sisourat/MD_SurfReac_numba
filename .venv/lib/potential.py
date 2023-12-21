import numpy as np
from utils import angtoau, evtoau
from numba import njit

@njit()
def Ua(x,D_a,Delta_a,alpha_a,z0):
    return D_a / (4 * (1 + Delta_a)) * (
            (3 + Delta_a) * np.exp(-2 * alpha_a * (x - z0)) - (2 + 6 * Delta_a) * np.exp(
        -alpha_a * (x - z0)))

@njit()
def Um(x,D_m,Delta_m,alpha_m,r0):
    return D_m / (4 * (1 + Delta_m)) * (
                (3 + Delta_m) * np.exp(-2 * alpha_m * (x - r0)) - (2 + 6 * Delta_m) * np.exp(
            -alpha_m * (x - r0)))

@njit()
def Qa(x,D_a,Delta_a,alpha_a,z0):
    return D_a / (4 * (1 + Delta_a)) * (
                (1 + 3 * Delta_a) * np.exp(-2 * alpha_a * (x - z0)) - (6 + 2 * Delta_a) * np.exp(
            -alpha_a * (x - z0)))

@njit()
def Qm(x,D_m,Delta_m,alpha_m,r0):
    return D_m / (4 * (1 + Delta_m)) * (
                (1 + 3 * Delta_m) * np.exp(-2 * alpha_m * (x - r0)) - (6 + 2 * Delta_m) * np.exp(
            -alpha_m * (x - r0)))

@njit()
def pot3d(rho, z, Z, D_a, Delta_a, alpha_a, D_m, Delta_m, alpha_m, mt, mp, mtot, z0, r0):
    # rho
    # z = zp - zt
    # Z = (mt*zt + mp*zp)/M
    zt = Z - (mp / mtot) * z
    zp = Z + (mt / mtot) * z
    r = np.sqrt(rho * rho + z * z)
    return (Ua(zt, D_a, Delta_a, alpha_a, z0) + Ua(zp, D_a, Delta_a, alpha_a, z0) + Um(r, D_m, Delta_m, alpha_m, r0)
            - np.sqrt(Qm(r, D_m, Delta_m, alpha_m, r0) ** 2 + (Qa(zt, D_a, Delta_a, alpha_a, z0)
            + Qa(zp, D_a, Delta_a, alpha_a, z0)) ** 2 - (Qa(zt, D_a, Delta_a, alpha_a, z0) +
            Qa(zp, D_a, Delta_a, alpha_a, z0)) * Qm(r, D_m, Delta_m, alpha_m, r0)))

@njit()
def Vfar_a(z1, D_a, Delta_a, alpha_a, z0):
    return Ua(z1, D_a, Delta_a, alpha_a, z0) - np.abs(Qa(z1, D_a, Delta_a, alpha_a, z0))

@njit()
def Vfar_m(b, D_m, Delta_m, alpha_m, r0):
    return Um(b, D_m, Delta_m, alpha_m, r0) - np.abs(Qm(b, D_m, Delta_m, alpha_m, r0))




