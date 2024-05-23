import numpy as np
from ex_ante_functions_jit import *

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

turnouts = calc_turnouts_b(param_arr,eps,m_arr,kap_arr,0,2,N)
utilities = calc_utilities_b(turnouts,param_arr,m_arr,kap_arr,0,2,N)
str1 = make_param_string(param_arr,0,2)
plot_turnouts_utilities(m_arr,kap_arr,turnouts,utilities,'$\kappa$','$m$','kappa','m',str1)

turnouts = calc_turnouts_b(param_arr,eps,b0_arr,kap_arr,4,2,N)
utilities = calc_utilities_b(turnouts,param_arr,b0_arr,kap_arr,4,2,N)
str1 = make_param_string(param_arr,4,2)
plot_turnouts_utilities(b0_arr,kap_arr,turnouts,utilities,'$\kappa$','$b_0$','kappa','b0',str1)

amount_eq, amount_eq_winning = counts_equilibria_b(param_arr,eps,m_arr,kap_arr,0,2,N)
str1 = make_param_string(param_arr,0,2)
plot_counts(m_arr,kap_arr,amount_eq,amount_eq_winning,'$\kappa$','$m$','kappa','m',str1)

amount_eq, amount_eq_winning = counts_equilibria_b(param_arr,eps,b0_arr,kap_arr,4,2,N)
str1 = make_param_string(param_arr,4,2)
plot_counts(b0_arr,kap_arr,amount_eq,amount_eq_winning,'$\kappa$','$b_0$','kappa','b0',str1)

amount_eq, amount_eq_winning = counts_equilibria_b(param_arr,eps,b0_arr,a0_arr,4,3,N)
str1 = make_param_string(param_arr,4,3)
plot_counts(b0_arr,a0_arr,amount_eq,amount_eq_winning,'$a_0$','$b_0$','a0','b0',str1)

# amount_eq, amount_eq_winning = counts_equilibria_b(param_arr,eps,bv_arr,av_arr,7,5,N)
# str1 = make_param_string(param_arr,7,5)
# plot_counts(bv_arr,av_arr,amount_eq,amount_eq_winning,'$a_v$','$b_v$','av','bv',str1)

amount_eq, amount_eq_winning = counts_equilibria_b(param_arr,eps,rho_arr,kap_arr,6,2,N)
str1 = make_param_string(param_arr,6,2)
plot_counts(rho_arr,kap_arr,amount_eq,amount_eq_winning,'$\kappa$',r'$\rho$','kappa','rho',str1)


