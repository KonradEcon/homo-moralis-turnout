from ex_post_funs_simple import *
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.optimize as sco

viridis_map = mpl.colormaps.get_cmap('viridis') # getting some colors
viridis = viridis_map(np.linspace(0,1,8))
color_a = "black"
color_b = viridis[6]

def do_plot(m,kap,rho,thea,theb,av,bv,a0,b0,eps,N,name,approx=False,add=None,approx2=False):
    b_vec = np.linspace(b0,b0+bv,N)
    a_vec = np.linspace(a0,a0+av,N)

    x,y = find_group_br_a_vecb(m,thea,kap,a0,av,b_vec,eps)
    x[x==0] = np.NaN
    y[y==0] = np.NaN
    z,w = find_group_br_b_veca(m,theb,kap,rho,b0,bv,a_vec,eps)
    z[z==0] = np.NaN
    w[w==0] = np.NaN

    a,b = find_all_equilibria(m,thea,theb,kap,rho,a0,b0,av,bv,eps)

    if not add is None:
        a = np.append(a,add[0])
        b = np.append(b,add[1])

    fig, ax = plt.subplots(figsize=(5,5))
    ax.scatter(x,y,color=color_a,s=1)
    ax.scatter(z,w,color=color_b,s=1)
    ax.plot([np.NaN],[np.NaN],color=color_a,label="$A$-consistent")
    ax.plot([np.NaN],[np.NaN],color=color_b,label="$B$-consistent")
    ax.plot(np.linspace(max(a0,b0),min(b0+bv,a0+av),N),np.linspace(max(a0,b0),min(b0+bv,a0+av),N),color="black",label="b=a",linestyle="--")

    for i in range(0,len(a)):
        if i == 0:
            ax.scatter(a[i],b[i],color="black",marker="*",s=15,label="Equilibria")
        else:
            ax.scatter(a[i],b[i],color="black",marker="*",s=15)
        ax.annotate(r"$" + str(i+1) + r"$",(a[i]+0.005,b[i]+0.005),color="black")
        print(name)
        print("a=" + str(a[i]))
        print("b=" + str(b[i]))
    # uncomment to plot the ratio condition:
    # quadsol = a0 /2 + np.sqrt(a0**2/ 4 + (av**2/(rho*bv**2))*(b_vec**2 - b0*b_vec))
    # plt.plot(quadsol,b_vec,color="green",label=r"$\frac{b}{\rho a} = \frac{C_A'(a)}{C_B'(b)}$")
    ax.set_xlabel("$a$")
    if approx2:
        fig.text(0.5,-0.01,r'$m='+ str(m) + r',\kappa='+ str(kap) + r',\rho=' + str(rho) + r',\theta_A\approx' + str(round(thea,2)) + r',\theta_B=' + str(theb) + r',$',ha="center")
    else:
        fig.text(0.5,-0.01,r'$m='+ str(m) + r',\kappa='+ str(kap) + r',\rho=' + str(rho) + r',\theta_A=' + str(thea) + r',\theta_B=' + str(theb) + r',$',ha="center")
    if approx:
        fig.text(0.5,-0.05,r'$a_0=' + str(a0) +r',a_v=' + str(av) + r',b_0=' + str(b0) + r',b_v=\frac{b_0 a_v}{a_0}\approx' + str(round(bv,2)) + r'$',ha="center")
    else:
        fig.text(0.5,-0.04,r'$a_0=' + str(a0) +r',a_v=' + str(av) + r',b_0=' + str(b0) + r',b_v=' + str(bv) + r'$',ha="center")
    ax.set_ylabel("$b$")
    ax.legend()
    plt.savefig("1d_plots/" + name + ".png",format="png",dpi=600,bbox_inches="tight")
    plt.close()
    if len(a) > 0:
        fig, ax = plt.subplots(figsize=(5,5))
        ax.set_xlabel("$a^i$")
        ax.set_ylabel("$U_A^\kappa(a,a^i)$")
        viridis_map_new = mpl.colormaps.get_cmap('viridis')
        viridis_new = viridis_map_new(np.linspace(0,1,len(a)))
        if add is None:
            for i in range(0,len(a)):
                plt.scatter(a[i],utility_a(m,thea,kap,a0,av,a[i],b[i],a[i]),color=viridis_new[i],marker="*",s=15)
                plt.plot(a_vec,utility_a(m,thea,kap,a0,av,a[i],b[i],a_vec),color=viridis_new[i],label=r"$(a,b)=(a_" + str(i+1) + r",b_" + str(i+1) + r")$")
        else:
            for i in range(0,len(a)):
                plt.scatter(a[i],utility_a(m,thea,kap,a0,av,a[i],b[i],a[i]),color=viridis_new[0],marker="*",s=15)
                if i == 0:
                    plt.plot(a_vec,utility_a(m,thea,kap,a0,av,a[i],b[i],a_vec),color=viridis_new[i],label=r"$(a,b)=(a_" + str(i+1) + r",b_" + str(i+1) + r")$")
        plt.legend()
        plt.savefig("1d_plots/" + name + "_deviations_A.png",format="png",dpi=600,bbox_inches="tight")
        plt.close()
        fig, ax = plt.subplots(figsize=(5,5))
        ax.set_xlabel("$b^i$")
        ax.set_ylabel("$U_B^\kappa(b,b^i)$")
        for i in range(0,len(a)):
            plt.scatter(b[i],utility_b(m,theb,kap,rho,b0,bv,a[i],b[i],b[i]),color=viridis_new[i],marker="*",s=15)
            plt.plot(b_vec,utility_b(m,theb,kap,rho,b0,bv,a[i],b[i],b_vec),color=viridis_new[i],label=r"$(a,b)=(a_" + str(i+1) + r",b_" + str(i+1) + r")$")
        plt.legend()
        plt.savefig("1d_plots/" + name + "_deviations_B.png",format="png",dpi=600,bbox_inches="tight")
        plt.close()

N = 500
eps = 1e-10

m = 10.0
kap = 0.5
rho = 2.0
thea = 3.0
theb = thea
av = 1.5
bv = 0.9
a0 = 0.5
b0 = 0.3


do_plot(m,kap,rho,thea,theb,av,bv,a0,b0,eps,N,"ex_post_br_no_eq")

m =40.0
rho =  5.0
thea =  2.9
theb = thea
a0 = 0.85
b0 = 0.60
av = 0.90
kap = 0.14
bv = b0*av/a0

do_plot(m,kap,rho,thea,theb,av,bv,a0,b0,eps,N,"ex_post_br_5_eq",True)

m = 4.8
rho = 2.35
thea = 2.0
theb = thea
a0 = 0.5
b0 = 0.1
av = 1.25
bv = 1.25
kap = 0.1

do_plot(m,kap,rho,thea,theb,av,bv,a0,b0,eps,N,"ex_post_br_1_eq_kap_0_1")


kap = 0.3
do_plot(m,kap,rho,thea,theb,av,bv,a0,b0,eps,N,"ex_post_br_1_eq_kap_0_3")

m = 3.0
rho = 3.6
kap = 0.5
thea = 1.1
theb = 2.0
a0 = 0.5
b0 = 0.4
av = 1.4
bv = 1.1

do_plot(m,kap,rho,thea,theb,av,bv,a0,b0,eps,N,"minority_high_stake_majority_cost_advantage")

rho = 1
thea = 2.0
b0 = 0.7
av = 1.25
bv = 1.0
do_plot(m,kap,rho,thea,theb,av,bv,a0,b0,eps,N,"minority_larger_base")

rho = 1.5
kap = 1
m = 5.7
theb = 0.1
a0 = 0.35
b0 = 0.05
av = 1.2
bv = 1.75
def difference(thea):
    a1 = np.real(roots_aux_a(m,thea,kap,a0,av,b0+bv)[2])
    return utility_a(m,thea,kap,a0,av,a1,b0+bv,a1) - utility_a(m,thea,kap,a0,av,a0+av,b0+bv,a0+av)

thea_root = sco.brentq(difference,1.0,1.5)
print(thea_root)

do_plot(m,kap,rho,thea_root,theb,av,bv,a0,b0,eps,N,"multi_kappa_1",add=(a0+av,b0+bv),approx2=True)

# example for limit
kap = 0.2
thea = 3.0
theb = 3.0
a0 = 0.2
b0 = 0.5
av = 0.6
bv = 0.15
rho = 1.2

m_vec_ln = np.linspace(-0.5,6,20000)
a = []
b = []
m = []

for i in range(0,len(m_vec_ln)):
    m_new = np.exp(np.log(10)*m_vec_ln[i])
    a_new,b_new = find_all_equilibria(m_new,thea,theb,kap,rho,a0,b0,av,bv,eps)
    for j in range(0,len(a_new)):
        a.append(a_new[j])
        b.append(b_new[j])
        m.append(m_new)

fig, ax = plt.subplots(figsize=(5,5))
ax.set_xlabel(r"$m$")
ax.scatter(m,a,color="black",s=1)
ax.scatter(m,b,color=color_b,s=1)
ax.plot([np.NaN],[np.NaN],color="black",label="$a$")
ax.plot([np.NaN],[np.NaN],color=color_b,label="$b$")
ax.plot(np.exp(np.log(10)*m_vec_ln),len(m_vec_ln)*[a0],linestyle="--",color="black",label=r"$a_0$")
ax.plot(np.exp(np.log(10)*m_vec_ln),len(m_vec_ln)*[b0],linestyle="--",color=color_b,label=r"$b_0$")
ax.plot(np.exp(np.log(10)*m_vec_ln),len(m_vec_ln)*[a0+av],linestyle="dotted",color="black",label=r"$a_0+a_v$")
ax.plot(np.exp(np.log(10)*m_vec_ln),len(m_vec_ln)*[b0+bv],linestyle="dotted",color=color_b,label=r"$b_0+b_v$")
ax.legend(loc="upper right")
ax.set_ylabel(r"$a,b$")
ax.set_xscale("log")
fig.text(0.5,-0.01,r'$\kappa='+ str(kap) + r',\rho=' + str(rho) + r',\theta_A=' + str(thea) + r',\theta_B=' + str(theb) + r',$',ha="center")
fig.text(0.5,-0.04,r'$a_0=' + str(a0) +r',a_v=' + str(av) + r',b_0=' + str(b0) + r',b_v=' + str(bv) + r'$',ha="center")

plt.savefig("1d_plots/limit_ex_post.png",format="png",dpi=600,bbox_inches="tight")

# example for high turnout limit
kap = 0.2
thea = 1.5
theb = 1.5
a0 = 0.35
b0 = 0.25
av = 1.25
bv = 1.35
rho = 1.2

m_vec_ln = np.linspace(-0.5,6,1000)
a = []
b = []
m = []

for i in range(0,len(m_vec_ln)):
    m_new = np.exp(np.log(10)*m_vec_ln[i])
    a_new,b_new = find_all_equilibria(m_new,thea,theb,kap,rho,a0,b0,av,bv,eps)
    for j in range(0,len(a_new)):
        a.append(a_new[j])
        b.append(b_new[j])
        m.append(m_new)

fig, ax = plt.subplots(figsize=(5,5))
ax.scatter(m,a,color="black",s=4)
ax.scatter(m,b,color=color_b,s=1)
ax.plot([np.NaN],[np.NaN],color="black",linewidth=2.0,label="$a$")
ax.plot([np.NaN],[np.NaN],color=color_b,linewidth=1.0,label="$b$")
ax.set_xlabel(r"$m$")
ax.plot(np.exp(np.log(10)*m_vec_ln),len(m_vec_ln)*[a0+av],linestyle="dotted",color="black",label=r"$a_0+a_v = b_0 + b_v$")
ax.plot(np.exp(np.log(10)*m_vec_ln),len(m_vec_ln)*[a0],linestyle="--",color="black",label=r"$a_0$")
ax.plot(np.exp(np.log(10)*m_vec_ln),len(m_vec_ln)*[b0],linestyle="--",color=color_b,label=r"$b_0$")
ax.legend(loc="right")
ax.set_ylabel(r"$a,b$")
ax.set_xscale("log")
fig.text(0.5,-0.01,r'$\kappa='+ str(kap) + r',\rho=' + str(rho) + r',\theta_A=' + str(thea) + r',\theta_B=' + str(theb) + r',$',ha="center")
fig.text(0.5,-0.04,r'$a_0=' + str(a0) +r',a_v=' + str(av) + r',b_0=' + str(b0) + r',b_v=' + str(bv) + r'$',ha="center")

plt.savefig("1d_plots/limit_ex_post_high_turnout.png",format="png",dpi=600,bbox_inches="tight")

# example for partial underdog compensation

# kap = 0.15
# m = 42.0
# thea = 2.9
# theb = thea
# a0 = 0.85
# b0 = 0.60
# av = 0.90
# bv = b0*av / a0

# rho_vec = np.linspace(1,5.5,400)

# a = [[],[],[]]
# b = [[],[],[]]
# rho_list = [[],[],[]]

# for i in range(0,len(rho_vec)):
#     rho = rho_vec[i]
#     a_new,b_new = find_equilibria_interior(m,thea,theb,kap,rho,a0,b0,av,bv,eps)
#     if len(a_new) == 1:
#         a[0].append(a_new[0])
#         b[0].append(b_new[0])
#         rho_list[0].append(rho)
#     elif len(a_new) == 2:
#         a[0].append(a_new[1])
#         b[0].append(b_new[1])
#         rho_list[0].append(rho)
#         a[1].append(a_new[0])
#         b[1].append(b_new[0])
#         rho_list[1].append(rho)
#     else:
#         a[0].append(a_new[2])
#         b[0].append(b_new[2])
#         rho_list[0].append(rho)
#         a[2].append(a_new[1])
#         b[2].append(b_new[1])
#         rho_list[2].append(rho)
#         a[1].append(a_new[0])
#         b[1].append(b_new[0])
#         rho_list[1].append(rho)
    

# fig, ax = plt.subplots(figsize=(5,5))
# ax.set_xlabel(r"$\rho$")
# for i in range(0,3):
#     ax.plot(rho_list[i],np.array(a[i])/av,color=viridis[2*i],linestyle="--")
#     ax.plot(rho_list[i],np.array(b[i])/bv,color=viridis[2*i])
#     ax.plot(rho_list[i],np.array(rho_list[i])*np.array(a[i])/av,color=viridis[2*i],linestyle="dotted")

# ax.plot([np.NaN],[np.NaN],color="black",linestyle="--",label=r"$\frac{a}{a_v}$")
# ax.plot([np.NaN],[np.NaN],color="black",label=r"$\frac{b}{b_v}$")
# ax.plot([np.NaN],[np.NaN],color="black",linestyle="dotted",label=r"$\rho \frac{a}{a_v}$")    
# ax.set_ylabel(r"$a,b$")
# ax.legend()
# plt.savefig("1d_plots/partial_compensation.png",format="png",dpi=600,bbox_inches="tight")
# plt.close()
