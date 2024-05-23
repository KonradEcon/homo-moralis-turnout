import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from ex_ante_functions import *
import scipy.optimize as sco

viridis_map = mpl.colormaps.get_cmap('viridis') # getting some colors
viridis = viridis_map(np.linspace(0,1,7))

# plotting h function
xs=np.linspace(-1, 1, 200)

plt.plot(xs,h(xs,0.1), label='m=0.1',color=viridis[0])
plt.plot(xs,h(xs,2), label='m=2',color=viridis[1],linestyle="dashdot")
plt.plot(xs,h(xs,10), label='m=10',color=viridis[2],linestyle="dotted")
plt.plot(xs,h(xs,100), label='m=100',color=viridis[3],linestyle="--")
plt.legend()
plt.savefig("1d_plots/arctan_m.png",format="png",dpi=600,bbox_inches='tight')
plt.close()

# plotting multiple equilibria for m growing
# defining some baseline parameters
m = 12.0
kap = 0.2
rho = 2.67
the = 2.0
av = 1.25
bv = 0.75
a0 = 0.5
b0 = 0.1
eps = 1e-10

# plotting a three equilibria case

b = get_equilibria_b(m,the,kap,a0,b0,av,rho,bv,eps)

fig, ax = plt.subplots(figsize=(5,5))
ax.set_xlabel("$b^i$")
ax.set_ylabel("$EU_B^\kappa(a_0,a_0,b,b^i)$")
viridis_new = viridis_map(np.linspace(0,1,len(b)+1))
b_vec = np.linspace(b0,b0+bv,500)
for i in range(0,len(b)):
    print(b[i])
    ax.scatter(b[i],utility_b(m,the,kap,a0,b0,av,rho,bv,b[i],b[i]),color=viridis_new[i],marker="*",s=15)
    ax.plot(b_vec,utility_b(m,the,kap,a0,b0,av,rho,bv,b[i],b_vec),color=viridis_new[i],label=r"$b=b_" + str(i+1) + r"$")

fig.text(0.5,-0.01,r'$m='+ str(m) + r',\kappa='+ str(kap) + r',\rho=' + str(rho) + r',\theta=' + str(the) +  r',$',ha="center")
fig.text(0.5,-0.04,r'$a_0=' + str(a0) +r',a_v=' + str(av) + r',b_0=' + str(b0) + r',b_v=' + str(bv) + r'$',ha="center")
plt.legend()
plt.savefig("1d_plots/ex_ante_3_eqs.png",format="png",dpi=600,bbox_inches="tight")
plt.close()

# plotting equilibria with m growing

m_axis_log = np.linspace(-0.5,6,20000)
m_arr = np.zeros(0)
b_arr = np.zeros(0)
for i in range(0,len(m_axis_log)):
    m_new = np.exp(np.log(10)*m_axis_log[i])
    eqs = get_equilibria_b(m_new,the,kap,a0,b0,av,rho,bv,eps)
    m_arr = np.append(m_arr,m_new*np.ones(len(eqs)))
    b_arr = np.append(b_arr,eqs)

fig, ax = plt.subplots(figsize=(5,5))
ax.plot(m_arr,b0*np.ones(len(m_arr)),label='$b_0$',color=viridis[0],linestyle="--")
ax.plot(m_arr,a0*np.ones(len(m_arr)),label='$a_0$',color=viridis[0],linestyle="dotted")
ax.scatter(m_arr,b_arr,s=1,color=viridis[3])
ax.plot([np.NaN],[np.NaN],label='Equilibria',color=viridis[1])
ax.set_xscale('log')
ax.set_xlabel("$m$")
ax.set_ylabel("$b$")

fig.text(0.5,-0.01,r'$\kappa='+ str(kap) + r',\rho=' + str(rho) + r',\theta=' + str(the) +  r',$',ha="center")
fig.text(0.5,-0.04,r'$a_0=' + str(a0) +r',a_v=' + str(av) + r',b_0=' + str(b0) + r',b_v=' + str(bv) + r'$',ha="center")
plt.legend()
plt.savefig("1d_plots/ex_ante_limit.png",format="png",dpi=600,bbox_inches='tight')
plt.close()

# two equilibria case

b0 = 0.1
a0 = 0.45
bv = 0.4
av = 0.6
rho = 3.0
kap  =0.4
m = 14.0
the = 2.0

b = get_equilibria_b(m,the,kap,a0,b0,av,rho,bv,eps)

fig, ax = plt.subplots(figsize=(5,5))
ax.set_xlabel("$b^i$")
ax.set_ylabel("$EU_B^\kappa(a_0,a_0,b,b^i)$")
viridis_new = viridis_map(np.linspace(0,1,len(b)+1))
b_vec = np.linspace(b0,b0+bv,500)
for i in range(0,len(b)):
    print(b[i])
    ax.scatter(b[i],utility_b(m,the,kap,a0,b0,av,rho,bv,b[i],b[i]),color=viridis_new[i],marker="*",s=15)
    ax.plot(b_vec,utility_b(m,the,kap,a0,b0,av,rho,bv,b[i],b_vec),color=viridis_new[i],label=r"$b=b_" + str(i+1) + r"$")

fig.text(0.5,-0.01,r'$m='+ str(m) + r',\kappa='+ str(kap) + r',\rho=' + str(rho) + r',\theta=' + str(the) +  r',$',ha="center")
fig.text(0.5,-0.04,r'$a_0=' + str(a0) +r',a_v=' + str(av) + r',b_0=' + str(b0) + r',b_v=' + str(bv) + r'$',ha="center")

plt.legend()
plt.savefig("1d_plots/ex_ante_2_eqs.png",format="png",dpi=600,bbox_inches="tight")
plt.close()

# finding and plotting a knife-edge case for two equilibria with kappa = 1

# kappa = 1 and other parameters except bv selected to get a knife-edge case
m = 40.0
kap = 1.0
rho = 2.0
the = 6.0
a0 = 0.55
av = 0.75
b0 = 0.2

# finding the knife-edge case by varying bv
def difference(bv):
    roots = np.real(roots_aux_b(m,the,kap,a0,b0,av,rho,bv))
    return utility_b(m,the,kap,a0,b0,av,rho,bv,roots[2],roots[2]) - utility_b(m,the,kap,a0,b0,av,rho,bv,roots[0],roots[0])

bv_knife = sco.brentq(difference,0.5,1.0)
print(bv_knife)
vec_difference = np.vectorize(difference)
bv_vec = np.linspace(0.5,0.75,500)
plt.plot(bv_vec,vec_difference(bv_vec),color=viridis[0])
plt.scatter(bv_knife,0,color=viridis[0],marker="*",s=15,label=r"$b_v^*$")
plt.xlabel("$b_v$")
plt.ylabel(r"$EU_B^1(a_0,a_0,b_h,b_h)-EU_B^1(a_0,a_0,b_\ell,b_\ell$")
plt.legend()
plt.savefig("1d_plots/knife_edge_diff_between_local_maxes.png",format="png",dpi=600,bbox_inches='tight')
plt.close()


# plotting the knife-edge case
eqs = np.real(roots_aux_b(m,the,kap,a0,b0,av,rho,bv_knife))
eqs = np.array([eqs[0],eqs[2]])
fig, ax = plt.subplots(figsize=(5,5))
b_vec = np.linspace(b0,b0+0.59,500)
ax.plot(b_vec,utility_b(m,the,kap,a0,b0,av,rho,0.59,b_vec,b_vec),color=viridis[1],linestyle="dotted",label=r"$b_v = 0.59$")
b_vec = np.linspace(b0,b0+0.62,500)
ax.plot(b_vec,utility_b(m,the,kap,a0,b0,av,rho,0.62,b_vec,b_vec),color=viridis[2],linestyle="dashdot",label=r"$b_v = 0.62$")
b_vec = np.linspace(b0,b0+bv_knife,500)
ax.plot(b_vec,utility_b(m,the,kap,a0,b0,av,rho,bv_knife,b_vec,b_vec),color=viridis[0],label=r"$b_v \approx" + str(round(bv_knife,3)) + r"$")
for i in range(0,len(eqs)):
    ax.scatter(eqs[i],utility_b(m,the,kap,a0,b0,av,rho,bv_knife,eqs[i],eqs[i]),color=viridis[0],marker="*",s=15)
ax.plot(b_vec,len(b_vec)*[utility_b(m,the,kap,a0,b0,av,rho,bv_knife,eqs[0],eqs[0])],color=viridis[0],linestyle="--")
ax.set_xlabel("$b^i$")
ax.set_ylabel("$EU_B^1(a_0,a_0,b,b^i)$")

fig.text(0.5,-0.01,r'$m='+ str(m) + r',\kappa='+ str(kap) + r',\rho=' + str(rho) + r',\theta=' + str(the) +  r',$',ha="center")
fig.text(0.5,-0.04,r'$a_0=' + str(a0) +r',a_v=' + str(av) + r',b_0=' + str(b0) + r'$',ha="center")
plt.legend()
plt.savefig("1d_plots/ex_ante_knife_edge_kappa_is_1.png",format="png",dpi=600,bbox_inches='tight')
plt.close()

# plotting all equilibria when av starts exceeding rho bv

b0 = 0.4
a0 = 0.5
bv = 0.4
rho = 1.5
kap  =0.4
m = 14.0
the = 2.0

fig, ax = plt.subplots(figsize=(5,5))
av_arr = np.linspace(0.3,0.9,500)
av_a_eqs = np.zeros(0)
av_b_eqs = np.zeros(0)
a_eqs = np.zeros(0)
b_eqs = np.zeros(0)
for i in range(0,len(av_arr)):
    eqs_b = get_equilibria_b(m,the,kap,a0,b0,av_arr[i],rho,bv,eps)
    eqs_a = get_equilibria_a(m,the,kap,a0,b0,av_arr[i],rho,bv,eps)
    av_a_eqs = np.append(av_a_eqs,av_arr[i]*np.ones(len(eqs_a)))
    av_b_eqs = np.append(av_b_eqs,av_arr[i]*np.ones(len(eqs_b)))
    a_eqs = np.append(a_eqs,eqs_a)
    b_eqs = np.append(b_eqs,eqs_b)

ax.plot(av_arr,b0*np.ones(len(av_arr)),label='$b_0$',color=viridis[5],linestyle="--")
ax.plot(av_arr,a0*np.ones(len(av_arr)),label='$a_0$',color=viridis[0],linestyle="--")

ax.scatter(av_a_eqs,a_eqs,s=1,color=viridis[0])
ax.plot([np.NaN],[np.NaN],label='$a$ in equilibrium',color=viridis[0])
ax.scatter(av_b_eqs,b_eqs,s=1,color=viridis[5])
ax.plot([np.NaN],[np.NaN],label='$b$ in equilibrium',color=viridis[5])
fig.text(0.5,-0.01,r'$m='+ str(m) + r',\kappa='+ str(kap) + r',\rho=' + str(rho) + r',\theta=' + str(the) +  r',$',ha="center")
fig.text(0.5,-0.04,r'$a_0=' + str(a0) + r',b_0=' + str(b0) +r',b_v=' + str(bv) +  r'$',ha="center")
ax.set_xlabel("$a_v$")
ax.set_ylabel("$a,b$")
plt.legend()
plt.savefig("1d_plots/ex_ante_around_lambda_0.png",format="png",dpi=600,bbox_inches='tight')
plt.close()

# plotting majority coordination problem with high minority base
b0 = 0.5
bv = 0.5
av = 1.0
rho = 1.5
m = 10.0

fig, ax = plt.subplots(figsize=(5,5))
ax.plot(np.linspace(0.0,1.0,3),3*[b0],color="black",linestyle="--",label=r'$b_0$')
a0_vec = np.linspace(0.0,1.0,500)
j=0
for kap in [0.1,0.5,0.8]:
    a0_vec_plot = np.zeros(0)
    eqs = np.zeros(0)
    for i in range(0,len(a0_vec)):
        new_eqs = get_equilibria_a(m,the,kap,a0_vec[i],b0,av,rho,bv,eps)
        eqs = np.append(eqs,new_eqs)
        a0_vec_plot = np.append(a0_vec_plot,a0_vec[i]*np.ones(len(new_eqs)))
    ax.scatter(a0_vec_plot,eqs,s=1,color =viridis[3*j])
    ax.plot([np.NaN],[np.NaN],label=r'$\kappa = ' + str(kap) + r'$',color=viridis[3*j])
    j+=1

ax.set_xlabel("$a_0$")
ax.set_ylabel("a")
fig.text(0.5,-0.01,r'$m='+ str(m) + r',\rho=' + str(rho) + r',\theta=' + str(the) +  r',$',ha="center")
fig.text(0.5,-0.04,r'$a_v=' + str(av) + r',b_0=' + str(b0) +r',b_v=' + str(bv) +  r'$',ha="center")
plt.legend()
plt.savefig("1d_plots/ex_ante_majority_coordination_problem.png",format="png",dpi=600,bbox_inches='tight')
