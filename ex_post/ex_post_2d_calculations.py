import numpy as np
from ex_post_funs_jit import *
from pathlib import Path

N = 4000 # set to 100 or less for testing, 4000 for high resolution plots

m = 40.0
kap = 0.14
rho = 5.0
thea = 2.9
theb = thea
av = 0.9
a0 = 0.85
b0 = 0.6
bv = b0*av/a0
eps = 1e-10
param_arr = np.array([m,thea,theb,kap,rho,a0,b0,av,bv])

m_arr = np.linspace(0.5,50,N)
kap_arr = np.linspace(0.0,1.0,N)
b0_arr = np.linspace(0.01,1.5,N)
a0_arr = np.linspace(0.01,1.5,N)
bv_arr = np.linspace(0.1,0.55,N)
av_arr = np.linspace(0.3,1.5,N)
rho_arr = np.linspace(1.0,8.0,N)
thea_arr = np.linspace(1.0,6.0,N)
theb_arr = np.linspace(1.0,6.0,N)

cache_dir = Path("/home/konrad/.homo_moralis_cache")
str1 = make_param_string_all(param_arr)
with open(cache_dir / "all_params.txt", "w") as f:
    f.write(str1)

counts,counts_a_winning,counts_b_winning,turnouts_a,turnouts_b,utilities_a,utilities_b = counts_turnouts_utilities(param_arr,eps,theb_arr,thea_arr,2,1,N)
str1 = make_param_string(param_arr,2,1)
np.savez(cache_dir / 'output_theb_thea.npz', theb_arr=theb_arr, thea_arr=thea_arr, counts=counts, counts_a_winning=counts_a_winning, counts_b_winning=counts_b_winning, turnouts_a=turnouts_a, turnouts_b=turnouts_b, utilities_a=utilities_a, utilities_b=utilities_b)
with open(cache_dir / "params_theb_thea.txt", "w") as f:
    f.write(str1)

counts,counts_a_winning,counts_b_winning,turnouts_a,turnouts_b,utilities_a,utilities_b = counts_turnouts_utilities(param_arr,eps,m_arr,kap_arr,0,3,N)
str1 = make_param_string(param_arr,0,3)
np.savez(cache_dir / 'output_m_kap.npz', m_arr=m_arr, kap_arr=kap_arr, counts=counts, counts_a_winning=counts_a_winning, counts_b_winning=counts_b_winning, turnouts_a=turnouts_a, turnouts_b=turnouts_b, utilities_a=utilities_a, utilities_b=utilities_b)
with open(cache_dir / "params_m_kap.txt", "w") as f:
    f.write(str1)

counts,counts_a_winning,counts_b_winning,turnouts_a,turnouts_b,utilities_a,utilities_b = counts_turnouts_utilities(param_arr,eps,b0_arr,kap_arr,6,3,N)
str1 = make_param_string(param_arr,6,3)
np.savez(cache_dir / 'output_b0_kap.npz', b0_arr=b0_arr, kap_arr=kap_arr, counts=counts, counts_a_winning=counts_a_winning, counts_b_winning=counts_b_winning, turnouts_a=turnouts_a, turnouts_b=turnouts_b, utilities_a=utilities_a, utilities_b=utilities_b)
with open(cache_dir / "params_b0_kap.txt", "w") as f:
    f.write(str1)

counts,counts_a_winning,counts_b_winning,turnouts_a,turnouts_b,utilities_a,utilities_b = counts_turnouts_utilities(param_arr,eps,b0_arr,a0_arr,6,5,N)
str1 = make_param_string(param_arr,6,5)
np.savez(cache_dir / 'output_b0_a0.npz', b0_arr=b0_arr, a0_arr=a0_arr, counts=counts, counts_a_winning=counts_a_winning, counts_b_winning=counts_b_winning, turnouts_a=turnouts_a, turnouts_b=turnouts_b, utilities_a=utilities_a, utilities_b=utilities_b)
with open(cache_dir / "params_b0_a0.txt", "w") as f:
    f.write(str1)

counts,counts_a_winning,counts_b_winning,turnouts_a,turnouts_b,utilities_a,utilities_b = counts_turnouts_utilities(param_arr,eps,rho_arr,kap_arr,4,3,N)
str1 = make_param_string(param_arr,4,3)
np.savez(cache_dir / 'output_rho_kap.npz', rho_arr=rho_arr, kap_arr=kap_arr, counts=counts, counts_a_winning=counts_a_winning, counts_b_winning=counts_b_winning, turnouts_a=turnouts_a, turnouts_b=turnouts_b, utilities_a=utilities_a, utilities_b=utilities_b)
with open(cache_dir / "params_rho_kap.txt", "w") as f:
    f.write(str1)

counts,counts_a_winning,counts_b_winning,turnouts_a,turnouts_b,utilities_a,utilities_b = counts_turnouts_utilities(param_arr,eps,b0_arr,bv_arr,6,8,N)
str1 = make_param_string(param_arr,6,8)
np.savez(cache_dir / 'output_b0_bv.npz', b0_arr=b0_arr, bv_arr=bv_arr, counts=counts, counts_a_winning=counts_a_winning, counts_b_winning=counts_b_winning, turnouts_a=turnouts_a, turnouts_b=turnouts_b, utilities_a=utilities_a, utilities_b=utilities_b)
with open(cache_dir / "params_b0_bv.txt", "w") as f:
    f.write(str1)