
from potential import *
from read_input import *
from  dynamics import *
from utils import *

data = read_input('input.json')
#print(data)

blist = data['blist']
mt = data['mt']
mp = data['mp']
zt_i = 0.0
zp_i = data['zp_i']

ekt_i = 0.0
ecoll = data['ecoll']
omega_a = data['omega_a']

ntraj = data['ntraj']
theta = 0.0 # normal incidence
delta_rho = 0.0
delta_ek = 0.0

p_rec = np.zeros(len(blist))
delta_p_rec = np.zeros(len(blist))

for ib, b in enumerate(blist):
   zt_0_vect, vzt_0_vect = initial_gen_target(zt_i,ekt_i,omega_a,mt,ntraj)
   zp_0_vect, rho_0_vect, vzp_0_vect, vrho_0_vect = initial_gen_proj(zp_i,b,theta,ecoll,
                                                    delta_rho,delta_ek,mp,ntraj)
   n_rec, eZ_fm, er_fm, evib_fm = solv_ntraj(data, zp_0_vect, vzp_0_vect, rho_0_vect, vrho_0_vect,
                                                   zt_0_vect, vzt_0_vect)

   p_rec[ib] = n_rec / ntraj
   if n_rec == 0:
       delta_p_rec[ib] = 0
   else:
       delta_p_rec[ib] = n_rec / ntraj * np.sqrt((ntraj - n_rec) / (ntraj * n_rec))

print(p_rec, delta_p_rec)
print("Cross sections:", Compute_CrossSec(blist, p_rec, delta_p_rec))

#print(pot3d(1.0,1.0,1.0,D_a=data['D_a'],
#              Delta_a=data['Delta_a'],
#              alpha_a=data['alpha_a'] ,
#              D_m=data['D_m'] ,
#              Delta_m=data['Delta_m'],
#              alpha_m=data['alpha_m'],
#              mt=data['mt'], mp = data['mp'],
#              mtot = data['mtot'],
#              r0 = data['r0'],
#              z0 = data['z0']))
