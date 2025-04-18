import numpy as np
from ex_ante_functions_jit import *
from pathlib import Path

cache_dir = Path("/home/konrad/.homo_moralis_cache_ex_ante")

N = 4000 # set to 100 for testing, 4000 for high resolution plots

m = 12.0
kap = 0.2
rho = 2.67
the = 2.0
av = 1.25
bv = 0.75
a0 = 0.5
b0 = 0.11
eps = 1e-10
param_arr = np.array([m,the,kap,a0,b0,av,rho,bv])

m_arr = np.linspace(0.5,50,N)
kap_arr = np.linspace(0.0,1.0,N)
b0_arr = np.linspace(0.01,0.4,N)
a0_arr = np.linspace(0.2,0.5,N)
bv_arr = np.linspace(0.4,0.98,N)
av_arr = np.linspace(0.6,1.06,N) # suitable boundaries for rho bv - av > 0 and av+a0 > bv+ b0
rho_arr = np.linspace(1.67,5,N) # suitable boundary for rho bv - av > 0

str1 = make_param_string_all(param_arr)
with open(cache_dir / "all_params.txt", "w") as f:
    f.write(str1)

amount_eq, amount_eq_winning = counts_equilibria_b(param_arr,eps,m_arr,kap_arr,0,2,N)
str1 = make_param_string(param_arr,0,2)
np.savez(cache_dir / 'eq_m_kap.npz', m_arr=m_arr, kap_arr=kap_arr, counts=amount_eq, counts_winning=amount_eq_winning)
with open(cache_dir / "params_m_kap.txt", "w") as f:
    f.write(str1)

amount_eq, amount_eq_winning = counts_equilibria_b(param_arr,eps,b0_arr,kap_arr,4,2,N)
str1 = make_param_string(param_arr,4,2)
np.savez(cache_dir / 'eq_b0_kap.npz', b0_arr=b0_arr, kap_arr=kap_arr, counts=amount_eq, counts_winning=amount_eq_winning)
with open(cache_dir / "params_b0_kap.txt", "w") as f:
    f.write(str1)

amount_eq, amount_eq_winning = counts_equilibria_b(param_arr,eps,rho_arr,kap_arr,6,2,N)
str1 = make_param_string(param_arr,6,2)
np.savez(cache_dir / 'eq_rho_kap.npz', rho_arr=rho_arr, kap_arr=kap_arr, counts=amount_eq, counts_winning=amount_eq_winning)
with open(cache_dir / "params_rho_kap.txt", "w") as f:
    f.write(str1)


