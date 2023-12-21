import numpy as np

evtoau = 1.0/27.211324570273
angtoau = 1.0/0.5291772109217
fstoau = 1.0/0.02418884326585747
def copy_properties(src, dest):
    for attr in vars(src).keys():
        setattr(dest, attr, getattr(src, attr))


def Compute_CrossSec(b_vect, p_vect, delta_p_vect, au=True):
    if au:
        b_vect_angs = b_vect * 0.529177
    else:
        b_vect_angs = b_vect

    sig = 2 * np.pi * np.trapz(b_vect_angs * p_vect, b_vect_angs)
    delta_sig = np.pi * (np.trapz(b_vect_angs * (p_vect + delta_p_vect), b_vect_angs) - np.trapz(
        b_vect_angs * (p_vect - delta_p_vect), b_vect_angs))
    return sig, delta_sig