import numpy as np
from ex_post_funs_jit import *

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

counts,counts_a_winning,counts_b_winning,turnouts_a,turnouts_b,utilities_a,utilities_b = counts_turnouts_utilities(param_arr,eps,m_arr,kap_arr,0,3,N)
str1 = make_param_string(param_arr,0,3)
# plot_turnouts_utilities(m_arr,kap_arr,turnouts_a,turnouts_b,utilities_a,utilities_b,'$\kappa$','$m$','kappa','m',str1)
plot_turnouts_utilities_flat(m_arr,kap_arr,turnouts_a,turnouts_b,utilities_a,utilities_b,r'$\kappa$','$m$','kappa','m',str1)
plot_counts(m_arr,kap_arr,counts,counts_a_winning,counts_b_winning,r'$\kappa$','$m$','kappa','m',str1)

counts,counts_a_winning,counts_b_winning,turnouts_a,turnouts_b,utilities_a,utilities_b = counts_turnouts_utilities(param_arr,eps,b0_arr,kap_arr,6,3,N)
str1 = make_param_string(param_arr,6,3)
# plot_turnouts_utilities(b0_arr,kap_arr,turnouts_a,turnouts_b,utilities_a,utilities_b,'$\kappa$','$b_0$','kappa','b0',str1)
plot_turnouts_utilities_flat(b0_arr,kap_arr,turnouts_a,turnouts_b,utilities_a,utilities_b,r'$\kappa$','$b_0$','kappa','b0',str1)
plot_counts(b0_arr,kap_arr,counts,counts_a_winning,counts_b_winning,r'$\kappa$','$b_0$','kappa','b0',str1)

counts,counts_a_winning,counts_b_winning,turnouts_a,turnouts_b,utilities_a,utilities_b = counts_turnouts_utilities(param_arr,eps,b0_arr,a0_arr,6,5,N)
str1 = make_param_string(param_arr,6,5)
# plot_turnouts_utilities(b0_arr,a0_arr,turnouts_a,turnouts_b,utilities_a,utilities_b,'$a_0$','$b_0$','a0','b0',str1)
plot_turnouts_utilities_flat(b0_arr,a0_arr,turnouts_a,turnouts_b,utilities_a,utilities_b,'$a_0$','$b_0$','a0','b0',str1)
plot_counts(b0_arr,a0_arr,counts,counts_a_winning,counts_b_winning,'$a_0$','$b_0$','a0','b0',str1)

counts,counts_a_winning,counts_b_winning,turnouts_a,turnouts_b,utilities_a,utilities_b = counts_turnouts_utilities(param_arr,eps,bv_arr,av_arr,8,7,N)
str1 = make_param_string(param_arr,8,7)
# plot_turnouts_utilities(bv_arr,av_arr,turnouts_a,turnouts_b,utilities_a,utilities_b,'$a_v$','$b_v$','av','bv',str1)
plot_turnouts_utilities_flat(bv_arr,av_arr,turnouts_a,turnouts_b,utilities_a,utilities_b,'$a_v$','$b_v$','av','bv',str1)
plot_counts(bv_arr,av_arr,counts,counts_a_winning,counts_b_winning,'$a_v$','$b_v$','av','bv',str1)

counts,counts_a_winning,counts_b_winning,turnouts_a,turnouts_b,utilities_a,utilities_b = counts_turnouts_utilities(param_arr,eps,rho_arr,kap_arr,4,3,N)
str1 = make_param_string(param_arr,4,3)
# plot_turnouts_utilities(rho_arr,kap_arr,turnouts_a,turnouts_b,utilities_a,utilities_b,r'$\kappa$',r'$\rho$','kappa','rho',str1)
plot_turnouts_utilities_flat(rho_arr,kap_arr,turnouts_a,turnouts_b,utilities_a,utilities_b,r'$\kappa$',r'$\rho$','kappa','rho',str1)
plot_counts(rho_arr,kap_arr,counts,counts_a_winning,counts_b_winning,r'$\kappa$',r'$\rho$','kappa','rho',str1)
