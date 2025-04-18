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
    x[x==0] = np.nan
    y[y==0] = np.nan
    z,w = find_group_br_b_veca(m,theb,kap,rho,b0,bv,a_vec,eps)
    z[z==0] = np.nan
    w[w==0] = np.nan

    a,b = find_all_equilibria(m,thea,theb,kap,rho,a0,b0,av,bv,eps)

    if not add is None:
        a = np.append(a,add[0])
        b = np.append(b,add[1])

    fig, ax = plt.subplots(figsize=(5,5))
    ax.scatter(x,y,color=color_a,s=1)
    ax.scatter(z,w,color=color_b,s=1)
    ax.plot([np.nan],[np.nan],color=color_a,label="$A$-consistent")
    ax.plot([np.nan],[np.nan],color=color_b,label="$B$-consistent")
    ax.plot(np.linspace(max(a0,b0),min(b0+bv,a0+av),N),np.linspace(max(a0,b0),min(b0+bv,a0+av),N),color="black",label="b=a",linestyle="--")

    for i in range(0,len(a)):
        
        stability_test_b_minus = [b[i],b[i]]
        stability_test_b_plus = [b[i],b[i]]
        stability_test_a_minus = [a[i],a[i] - 1e-3]
        stability_test_a_plus = [a[i],a[i] + 1e-3]

        for k in range(0,100):
            cand_br = find_group_br_b(m,theb,kap,rho,b0,bv,stability_test_a_plus[-1],eps)
            cand_br[cand_br < b0] = b0
            cand_br[cand_br > b0+bv] = b0+bv
            if len(cand_br) > 0:
                stability_test_b_plus.append(cand_br[np.argmin(np.abs(cand_br-stability_test_b_plus[-1]))])
                stability_test_a_plus.append(stability_test_a_plus[-1])
            else:
                break

            cand_br = find_group_br_a(m,thea,kap,a0,av,stability_test_b_plus[-1],eps)
            cand_br[cand_br < a0] = a0
            cand_br[cand_br > a0+av] = a0+av
            if len(cand_br)> 0:
                stability_test_a_plus.append(cand_br[np.argmin(np.abs(cand_br-stability_test_a_plus[-1]))])
                stability_test_b_plus.append(stability_test_b_plus[-1])
            else:
                break

        for k in range(0,100):
            cand_br = find_group_br_b(m,theb,kap,rho,b0,bv,stability_test_a_minus[-1],eps)
            cand_br[cand_br < b0] = b0
            cand_br[cand_br > b0+bv] = b0+bv
            # print(cand_br)
            if len(cand_br) > 0:
                stability_test_b_minus.append(cand_br[np.argmin(np.abs(cand_br-stability_test_b_minus[-1]))])
                stability_test_a_minus.append(stability_test_a_minus[-1])
            else:
                break

            cand_br = find_group_br_a(m,thea,kap,a0,av,stability_test_b_minus[-1],eps)
            cand_br[cand_br < a0] = a0
            cand_br[cand_br > a0+av] = a0+av
            if len(cand_br) > 0:
                stability_test_a_minus.append(cand_br[np.argmin(np.abs(cand_br-stability_test_a_minus[-1]))])
                stability_test_b_minus.append(stability_test_b_minus[-1])
            else:
                break
            
        plotline_a_minus = [stability_test_a_minus[0],stability_test_a_minus[-1]]
        plotline_b_minus = [stability_test_b_minus[0],stability_test_b_minus[-1]]
        plotline_a_plus = [stability_test_a_plus[0],stability_test_a_plus[-1]]
        plotline_b_plus = [stability_test_b_plus[0],stability_test_b_plus[-1]]

        # if np.abs(plotline_a_minus[-1]-a[i]) < 1e-3 + 1e-4 and np.abs(plotline_b_minus[-1]-b[i]) < 1e-3+ 1e-4 and np.abs(plotline_a_plus[-1]-a[i]) < 1e-3+ 1e-4 and np.abs(plotline_b_plus[-1]-b[i]) < 1e-3+ 1e-4:
    
            
        ax.scatter(a[i],b[i],color="black",marker="*",s=15)
        # else:
            # ax.scatter(a[i],b[i],color="red",marker="*",s=15)

        if i == 0:
            ax.scatter([np.nan],[np.nan],color="black",marker="*",s=15,label="Equilibria")
            # ax.scatter([np.nan],[np.nan],color="red",marker="*",s=15,label="Unstable Equilibria")



        # ax.plot(plotline_a_minus,plotline_b_minus,color="red",linestyle="-")
        # ax.plot(plotline_a_plus,plotline_b_plus,color="yellow",linestyle="-")
        print(len(stability_test_a_minus))
        print(len(stability_test_a_plus))
        print("equilibrium ", a[i],b[i])
        print("stab_a_plus ",stability_test_a_plus[-1],stability_test_b_plus[-1])
        print("stab_a_minus",stability_test_a_minus[-1],stability_test_b_minus[-1])
        

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
        ax.set_ylabel("$EU_A^\kappa(a,a^i)$")
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
        ax.set_ylabel("$EU_B^\kappa(b,b^i)$")
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


# plot accessible A
for m in [5.0,10.0,20.0]:

    a= 1.2
    b = 1.2
    i=0
    fig, ((ax1, ax2),(ax3,ax4)) = plt.subplots(figsize=(10,10),ncols=2,nrows=2)
    for kap in [1.0,0.8,0.6,0.4,0.2]:
        ai_range = np.linspace(a0,a0+av,N)
        x_range = (1-kap)*a + kap*ai_range
        y = utility_a(m,thea,kap,a0,av,a,b,ai_range)
        if i > 0:
            ax2.plot(x_range,h((x_range-b)/(x_range + b),m),label=r"$\kappa = " + str(kap) + r"$",color=viridis[2*i-2])
            ax1.plot(x_range,y,label=r"$\kappa=" + str(kap) + r"$",color=viridis[2*i-2])
        else:
            ax2.plot(x_range,h((x_range-b)/(x_range + b),m),label=r"$\kappa = " + str(kap) + r"$",color="black")
            ax1.plot(x_range,y,label=r"$\kappa=" + str(kap) + r"$",color="black")
        # ax2.plot(x_range,h((x_range-b)/(x_range + b),m),label=r"$\kappa = " + str(kap) + r"$",color=viridis[2*i])
        # ax1.plot(x_range,y,label=r"$\kappa=" + str(kap) + r"$",color=viridis[2*i])
        i+=1

    kap_arr = np.linspace(0.00,1.0,40)
    br_arr = []
    akap_arr = []
    utility_arr = []
    for i in range(0,len(kap_arr)):
        roots = roots_utility_a(m,thea,kap_arr[i],a0,av,a,b)
        roots = roots[np.imag(roots)< eps]
        roots = np.real(roots)
        roots = roots[roots > a0]
        roots = roots[roots < a0+av]
        roots = [roots[i] for i in range(0,len(roots))]
        roots.append(a0)
        roots.append(a0+av)
        utilities = []
        for j in range(0,len(roots)):
            utilities.append(utility_a(m,thea,kap_arr[i],a0,av,a,b,roots[j]))
        
        br_arr.append(roots[np.argmax(utilities)])
        akap_arr.append(br_arr[i]*kap_arr[i] + (1-kap_arr[i])*a)
        utility_arr.append(np.max(utilities))

    akap_arr = np.array(akap_arr)
    utility_arr = np.array(utility_arr)

    akap_dir = akap_arr[1:]- akap_arr[0:-1]
    utility_dir = utility_arr[1:] - utility_arr[0:-1]

    ax1.quiver(akap_arr[0:-1],utility_arr[0:-1],akap_dir,utility_dir,scale=1.0,angles="xy",scale_units="xy",width=0.003,units="width")
    ax1.scatter([], [], marker=r'$\longrightarrow$', c="black",label=r"BR as $\kappa\uparrow$",s=100)
    ax1.set_xlabel("$a^\kappa$")
    ax1.set_ylabel("$EU_A^\kappa(a,a^i)$")
    ax1.set_title(r"$a=" + str(a) + r", b=" + str(b)+ r"$")
    ax1.legend()
    ax2.set_xlabel("$a^\kappa$")
    ax2.set_ylabel("benefit term")
    ax2.set_title(r"$a=" + str(a) + r", b=" + str(b)+ r"$")

    ax2.legend()

    a= 0.9
    b = 1.2
    i=0
    for kap in [1.0,0.8,0.6,0.4,0.2]:
        x_range = np.linspace(np.max([a0,(1-kap)*a + kap*a0]),np.min([a0+av,(1-kap)*(a-a0)+kap*av + a0]),N)
        ai_range = (-(1-kap)*a + x_range)/kap
        y = utility_a(m,thea,kap,a0,av,a,b,ai_range)
        if i > 0:
            ax4.plot(x_range,h((x_range-b)/(x_range + b),m),label=r"$\kappa = " + str(kap) + r"$",color=viridis[2*i-2])
            ax3.plot(x_range,y,label=r"$\kappa=" + str(kap) + r"$",color=viridis[2*i-2])
        else:
            ax4.plot(x_range,h((x_range-b)/(x_range + b),m),label=r"$\kappa = " + str(kap) + r"$",color="black")
            ax3.plot(x_range,y,label=r"$\kappa=" + str(kap) + r"$",color="black")
        i+=1

    kap_arr = np.linspace(0.00,1.0,320)
    br_arr = []
    akap_arr = []
    utility_arr = []
    for i in range(0,len(kap_arr)):
        roots = roots_utility_a(m,thea,kap_arr[i],a0,av,a,b)
        roots = roots[np.imag(roots)< eps]
        roots = np.real(roots)
        roots = roots[roots > a0]
        roots = roots[roots < a0+av]
        roots = [roots[i] for i in range(0,len(roots))]
        roots.append(a0)
        roots.append(a0+av)
        utilities = []
        for j in range(0,len(roots)):
            utilities.append(utility_a(m,thea,kap_arr[i],a0,av,a,b,roots[j]))
        
        br_arr.append(roots[np.argmax(utilities)])
        akap_arr.append(br_arr[i]*kap_arr[i] + (1-kap_arr[i])*a)
        utility_arr.append(np.max(utilities))

    akap_arr = np.array(akap_arr)
    utility_arr = np.array(utility_arr)

    akap_dir = akap_arr[1:]- akap_arr[0:-1]
    utility_dir = utility_arr[1:] - utility_arr[0:-1]

    if np.any(akap_dir > 0.1):
        cutoff_index = np.argmax(akap_dir > 0.1) + 1
        print(cutoff_index)
        akap_arr_1 = akap_arr[cutoff_index-1::-8][::-1]
        akap_arr_2 = akap_arr[cutoff_index::8]
        utility_arr_1 = utility_arr[cutoff_index-1::-8][::-1]
        utility_arr_2 = utility_arr[cutoff_index::8]

        akap_dir_1 = akap_arr_1[1:]- akap_arr_1[0:-1]
        utility_dir_1 = utility_arr_1[1:] - utility_arr_1[0:-1]
        akap_dir_2 = akap_arr_2[1:]- akap_arr_2[0:-1]
        utility_dir_2 = utility_arr_2[1:] - utility_arr_2[0:-1]

        ax3.quiver(akap_arr_1[:-1],utility_arr_1[:-1],akap_dir_1,utility_dir_1,scale=1.0,angles="xy",scale_units="xy",width=0.003,units="width")
        ax3.quiver(akap_arr_2[:-1],utility_arr_2[:-1],akap_dir_2,utility_dir_2,scale=1.0,angles="xy",scale_units="xy",width=0.003,units="width")

        ax3.plot([akap_arr[cutoff_index-1],akap_arr[cutoff_index]],[utility_arr[cutoff_index-1],utility_arr[cutoff_index]],linestyle="dotted",color="black")


    else:
        akap_arr = akap_arr[::4]
        utility_arr = utility_arr[::4]
        akap_dir = akap_arr[1:]- akap_arr[0:-1]
        utility_dir = utility_arr[1:] - utility_arr[0:-1]
        ax3.quiver(akap_arr[:-1],utility_arr[:-1],akap_dir,utility_dir,scale=1,angles="xy",scale_units="xy",width=0.003,units="width")
    ax3.scatter([], [], marker=r'$\longrightarrow$', c="black",label=r"BR as $\kappa\uparrow$",s=100)
    ax3.set_xlabel("$a^\kappa$")
    ax3.set_ylabel("$EU_A^\kappa(a,a^i)$")
    ax3.set_title(r"$a=" + str(a) + r", b=" + str(b)+ r"$")
    ax3.legend()
    ax4.set_xlabel("$a^\kappa$")
    ax4.set_ylabel("benefit term")
    ax4.set_title(r"$a=" + str(a) + r", b=" + str(b)+ r"$")

    ax4.legend()

    fig.text(0.5,0.05,r'$m='+ str(m) + r',\rho =' + str(rho) + r',\theta_A=' + str(thea) + r',\theta_B=' + str(theb) + r',a_0=' + str(a0) +r',a_v=' + str(av) + r',b_0=' + str(b0) + r',b_v=' + str(bv) + r'$' ,ha="center") 
    plt.savefig(f"1d_plots/accessible_A_m={m}.png",format="png",dpi=600,bbox_inches="tight")

a= 1.1
b = 1.2
kap = 0.5
thea = 2.0
i=0
fig, ax1 = plt.subplots(figsize=(5,5),ncols=1,nrows=1)
fig2, ax2 = plt.subplots(figsize=(5,5),ncols=1,nrows=1)
for m in [1.0,5.0,10.0,15.0]:
    ai_range = np.linspace(a0,a0+av,N)
    y = utility_a(m,thea,kap,a0,av,a,b,ai_range)
    ax1.plot(ai_range,y,label=r"$m=" + str(m) + r"$",color=viridis[2*i])
    i+=1

m_arr = np.linspace(1.0,15.0,25)
br_arr = []
utility_arr = []
for i in range(0,len(m_arr)):
    roots = roots_utility_a(m_arr[i],thea,kap,a0,av,a,b)
    roots = roots[np.imag(roots)< eps]
    roots = np.real(roots)
    roots = roots[roots > a0]
    roots = roots[roots < a0+av]
    roots = [roots[i] for i in range(0,len(roots))]
    roots.append(a0)
    roots.append(a0+av)
    utilities = []
    for j in range(0,len(roots)):
        utilities.append(utility_a(m_arr[i],thea,kap,a0,av,a,b,roots[j]))
    
    br_arr.append(roots[np.argmax(utilities)])
    utility_arr.append(np.max(utilities))

br_arr = np.array(br_arr)
utility_arr = np.array(utility_arr)

br_dir = br_arr[1:]- br_arr[0:-1]
utility_dir = utility_arr[1:] - utility_arr[0:-1]

ax1.quiver(br_arr[0:-1],utility_arr[0:-1],br_dir,utility_dir,scale=1.0,angles="xy",scale_units="xy",width=0.003,units="width")
ax1.scatter([], [], marker=r'$\longrightarrow$', c="black",label=r"BR as $m\uparrow$",s=100)

m= 4.8



i=0
for thea in [1.0,2.0,4.0,6.0]:
    ai_range = np.linspace(a0,a0+av,N)
    y = utility_a(m,thea,kap,a0,av,a,b,ai_range)
    ax2.plot(ai_range,y,label=r"$\theta_A=" + str(thea) + r"$",color=viridis[2*i])
    i+=1

thea_arr = np.linspace(1.0,6.0,25)
br_arr = []
utility_arr = []
for i in range(0,len(thea_arr)):
    roots = roots_utility_a(m,thea_arr[i],kap,a0,av,a,b)
    roots = roots[np.imag(roots)< eps]
    roots = np.real(roots)
    roots = roots[roots > a0]
    roots = roots[roots < a0+av]
    roots = [roots[i] for i in range(0,len(roots))]
    roots.append(a0)
    roots.append(a0+av)
    utilities = []
    for j in range(0,len(roots)):
        utilities.append(utility_a(m,thea_arr[i],kap,a0,av,a,b,roots[j]))
    
    br_arr.append(roots[np.argmax(utilities)])
    utility_arr.append(np.max(utilities))

br_arr = np.array(br_arr)
utility_arr = np.array(utility_arr)

br_dir = br_arr[1:]- br_arr[0:-1]
utility_dir = utility_arr[1:] - utility_arr[0:-1]
ax2.quiver(br_arr[0:-1],utility_arr[0:-1],br_dir,utility_dir,scale=1.0,angles="xy",scale_units="xy",width=0.003,units="width")
ax2.scatter([], [], marker=r'$\longrightarrow$', c="black",label=r"BR as $\theta_A\uparrow$",s=100)

ax1.set_xlabel(r"$a^i$")
ax2.set_xlabel(r"$a^i$")
ax1.set_ylabel("$EU_A^\kappa(a,a^i)$")
ax2.set_ylabel("$EU_A^\kappa(a,a^i)$")
ax1.legend()
ax2.legend()

thea = 2.0

fig2.text(0.5,-0.01, r'$m=' + str(m) + r',\kappa=' + str(kap) + r',\rho=' + str(rho) + r',\theta_B=' + str(theb) + ',$', ha="center")
fig2.text(0.5,-0.04, r'$a_0=' + str(a0) +r',a_v=' + str(av) + r',b_0=' + str(b0) + r',b_v=' + str(bv) + r'$' ,ha="center") 
fig.text(0.5,-0.01, r'$\kappa=' + str(kap) + r',\rho=' + str(rho) + r',\theta_A=' + str(thea) + r',\theta_B=' + str(theb) + ',$', ha="center")
fig.text(0.5,-0.04, r'$a_0=' + str(a0) +r',a_v=' + str(av) + r',b_0=' + str(b0) + r',b_v=' + str(bv) + r'$' ,ha="center")

ax1.set_title("$a=" + str(a) + r", b=" + str(b)+ r"$")
ax2.set_title("$a=" + str(a) + r", b=" + str(b)+ r"$")
fig.savefig("1d_plots/comp_stat_BR_m.png",format="png",dpi=600,bbox_inches="tight")
fig2.savefig("1d_plots/comp_stat_BR_thea.png",format="png",dpi=600,bbox_inches="tight")
plt.close()


i = 0
b0 = 0.1
m = 10.0
a = 0.7
fig, ax = plt.subplots(figsize=(5,5))

for rho in [1.0,3.0,6.0,9.0]:
    bi_range = np.linspace(b0,b0+bv,N)
    y = utility_b(m,theb,kap,rho,b0,bv,a,b,bi_range)
    ax.plot(bi_range,y,label=r"$\rho=" + str(rho) + r"$",color=viridis[2*i])
    i+=1

rho_arr = np.linspace(1.0,9.0,25)
br_arr = []
utility_arr = []

for i in range(0,len(rho_arr)):
    roots = roots_utility_b(m,theb,kap,rho_arr[i],b0,bv,a,b)
    roots = roots[np.imag(roots)< eps]
    roots = np.real(roots)
    roots = roots[roots > b0]
    roots = roots[roots < b0+bv]
    roots = [roots[i] for i in range(0,len(roots))]
    roots.append(b0)
    roots.append(b0+bv)
    utilities = []
    for j in range(0,len(roots)):
        utilities.append(utility_b(m,theb,kap,rho_arr[i],b0,bv,a,b,roots[j]))
    
    br_arr.append(roots[np.argmax(utilities)])
    utility_arr.append(np.max(utilities))

br_arr = np.array(br_arr)
utility_arr = np.array(utility_arr)

br_dir = br_arr[1:]- br_arr[0:-1]
utility_dir = utility_arr[1:] - utility_arr[0:-1]

ax.quiver(br_arr[0:-1],utility_arr[0:-1],br_dir,utility_dir,scale=1.0,angles="xy",scale_units="xy",width=0.003,units="width")
ax.scatter([], [], marker=r'$\longrightarrow$', c="black",label=r"BR as $\rho\uparrow$",s=100)

ax.set_xlabel(r"$b^i$")
ax.set_ylabel("$EU_B^\kappa(b,b^i)$")
ax.legend()

fig.text(0.5,-0.01, r'$m=' + str(m) + r',\kappa=' + str(kap) + r',\theta_A=' + str(thea) + r',\theta_B=' + str(theb) + ',$', ha="center")
fig.text(0.5,-0.04, r'$a_0=' + str(a0) +r',a_v=' + str(av) + r',b_0=' + str(b0) + r',b_v=' + str(bv) + r'$' ,ha="center")
ax.set_title("$a=" + str(a) + r", b=" + str(b)+ r"$")
plt.savefig("1d_plots/comp_stat_BR_rho.png",format="png",dpi=600,bbox_inches="tight")

a = 1.1
rho = 2.35
kap = 0.3
b0 = 0.1
m = 4.8
m_space = np.linspace(1.0,30.0,1000)

ms = []
a = []
b = []

for m in m_space:
    a_new,b_new = find_all_equilibria(m,thea,theb,kap,rho,a0,b0,av,bv,eps)
    for j in range(0,len(a_new)):
        ms.append(m)
        a.append(a_new[j])
        b.append(b_new[j])

fig, ax = plt.subplots(figsize=(5,5))
ax.set_xlabel(r"$m$")
ax.scatter(ms,a,color="black",s=1)
ax.scatter(ms,b,color=color_b,s=1)
ax.plot([np.nan],[np.nan],color="black",label="$a$")
ax.plot([np.nan],[np.nan],color=color_b,label="$b$")
ax.plot(m_space,len(m_space)*[a0],linestyle="--",color="black",label=r"$a_0$")
ax.plot(m_space,len(m_space)*[b0],linestyle="--",color=color_b,label=r"$b_0$")
ax.plot(m_space,len(m_space)*[a0+av],linestyle="dotted",color="black",label=r"$a_0+a_v$")
ax.plot(m_space,len(m_space)*[b0+bv],linestyle="dotted",color=color_b,label=r"$b_0+b_v$")
ax.legend(loc="upper right")

ax.set_ylabel(r"$a,b$")
fig.text(0.5,-0.01,r'$\kappa='+ str(kap) + r',\rho=' + str(rho) + r',\theta_A=' + str(thea) + r',\theta_B=' + str(theb) + r',$',ha="center")

fig.text(0.5,-0.04,r'$a_0=' + str(a0) +r',a_v=' + str(av) + r',b_0=' + str(b0) + r',b_v=' + str(bv) + r'$',ha="center")

plt.savefig("1d_plots/comp_statics_m.png",format="png",dpi=600,bbox_inches="tight")

fig, ax = plt.subplots(figsize=(5,5))
ax.set_xlabel(r"$m$")
bs = [0.6,0.9,1.2,1.5]
i=0
for b in bs:
    ms = []
    a = []

    for m in m_space:
        a_new = find_group_br_a(m,thea,kap,a0,av,b,eps)
        for j in range(0,len(a_new)):
            ms.append(m)
            a.append(a_new[j])

    ax.scatter(ms,a,color=viridis[2*i],s=1)
    ax.plot(np.nan,np.nan,color=viridis[2*i],label=r"$b=" + str(b) + r"$")
    i+=1

ax.legend()
ax.set_ylabel(r"$a$")
fig.text(0.5,-0.01,r'$\kappa='+ str(kap) + r',\rho=' + str(rho) + r',\theta_A=' + str(thea) + r',\theta_B=' + str(theb) + r',$',ha="center")

fig.text(0.5,-0.04,r'$a_0=' + str(a0) +r',a_v=' + str(av)  + r',b_0=' + str(b0) + r',b_v=' + str(bv) + r'$',ha="center")

plt.savefig("1d_plots/comp_statics_m_a_consistent.png",format="png",dpi=600,bbox_inches="tight")

m = 5.0
kap_space = np.linspace(0.1,0.9,1000)
kaps = []
a = []
b = []

for kap in kap_space:
    a_new,b_new = find_all_equilibria(m,thea,theb,kap,rho,a0,b0,av,bv,eps)
    for j in range(0,len(a_new)):
        kaps.append(kap)
        a.append(a_new[j])
        b.append(b_new[j])

fig, ax = plt.subplots(figsize=(5,5))
ax.set_xlabel(r"$\kappa$")
ax.scatter(kaps,a,color="black",s=1)
ax.scatter(kaps,b,color=color_b,s=1)
ax.plot([np.nan],[np.nan],color="black",label="$a$")
ax.plot([np.nan],[np.nan],color=color_b,label="$b$")
ax.plot(kap_space,len(kap_space)*[a0],linestyle="--",color="black",label=r"$a_0$")
ax.plot(kap_space,len(kap_space)*[b0],linestyle="--",color=color_b,label=r"$b_0$")
ax.plot(kap_space,len(kap_space)*[a0+av],linestyle="dotted",color="black",label=r"$a_0+a_v$")
ax.plot(kap_space,len(kap_space)*[b0+bv],linestyle="dotted",color=color_b,label=r"$b_0+b_v$")
ax.legend(loc="upper right")

ax.set_ylabel(r"$a,b$")
fig.text(0.5,-0.01,r'$m='+ str(m) + r',\rho=' + str(rho) + r',\theta_A=' + str(thea) + r',\theta_B=' + str(theb) + r',$',ha="center")

fig.text(0.5,-0.04,r'$a_0=' + str(a0) +r',a_v=' + str(av) + r',b_0=' + str(b0) + r',b_v=' + str(bv) + r'$',ha="center")

plt.savefig("1d_plots/comp_statics_kap.png",format="png",dpi=600,bbox_inches="tight")


m= 5.0
fig, ax = plt.subplots(figsize=(5,5))
ax.set_xlabel(r"$\kappa$")


bs = [0.6,0.9,1.2,1.5]
i=0
for b in bs:
    kaps = []
    a = []

    for kap in kap_space:
        a_new = find_group_br_a(m,thea,kap,a0,av,b,eps)
        for j in range(0,len(a_new)):
            kaps.append(kap)
            a.append(a_new[j])

    ax.scatter(kaps,a,color=viridis[2*i],s=1)
    ax.plot(np.nan,np.nan,color=viridis[2*i],label=r"$b=" + str(b) + r"$")
    i+=1

ax.legend()
ax.set_ylabel(r"$a$")
fig.text(0.5,-0.01,r'$m='+ str(m) + r',\rho=' + str(rho) + r',\theta_A=' + str(thea) + r',\theta_B=' + str(theb) + r',b=' + str(b) + r'$',ha="center")

fig.text(0.5,-0.04,r'$a_0=' + str(a0) +r',a_v=' + str(av) + r'$',ha="center")

plt.savefig("1d_plots/comp_statics_kap_a_consistent.png",format="png",dpi=600,bbox_inches="tight")


kap = 0.3
rho_space = np.linspace(1.0,5.0,1000)
rhos = []
a = []
b = []

for rho in rho_space:
    a_new,b_new = find_all_equilibria(m,thea,theb,kap,rho,a0,b0,av,bv,eps)
    for j in range(0,len(a_new)):
        rhos.append(rho)
        a.append(a_new[j])
        b.append(b_new[j])

fig, ax = plt.subplots(figsize=(5,5))
ax.set_xlabel(r"$\rho$")
ax.scatter(rhos,a,color="black",s=1)
ax.scatter(rhos,b,color=color_b,s=1)
ax.plot([np.nan],[np.nan],color="black",label="$a$")
ax.plot([np.nan],[np.nan],color=color_b,label="$b$")
ax.plot(rho_space,len(rho_space)*[a0],linestyle="--",color="black",label=r"$a_0$")
ax.plot(rho_space,len(rho_space)*[b0],linestyle="--",color=color_b,label=r"$b_0$")
ax.plot(rho_space,len(rho_space)*[a0+av],linestyle="dotted",color="black",label=r"$a_0+a_v$")
ax.plot(rho_space,len(rho_space)*[b0+bv],linestyle="dotted",color=color_b,label=r"$b_0+b_v$")
ax.legend(loc="upper right")

ax.set_ylabel(r"$a,b$")

fig.text(0.5,-0.01,r'$m='+ str(m) + r',\kappa=' + str(kap) + r',\theta_A=' + str(thea) + r',\theta_B=' + str(theb) + r',$',ha="center") 
fig.text(0.5,-0.04,r'$a_0=' + str(a0) +r',a_v=' + str(av) + r',b_0=' + str(b0) + r',b_v=' + str(bv) + r'$',ha="center") 

plt.savefig("1d_plots/comp_statics_rho.png",format="png",dpi=600,bbox_inches="tight")

fig, ax = plt.subplots(figsize=(5,5))
ax.set_xlabel(r"$\rho$")

a_s = [0.6,0.9,1.2,1.5]
i=0
for a in a_s:
    rhos = []
    b = []

    for rho in rho_space:
        b_new = find_group_br_b(m,theb,kap,rho,b0,bv,a,eps)
        for j in range(0,len(b_new)):
            rhos.append(rho)
            b.append(b_new[j])

    ax.scatter(rhos,b,color=viridis[2*i],s=1)
    ax.plot(np.nan,np.nan,color=viridis[2*i],label=r"$a=" + str(a) + r"$")
    i+=1

ax.legend()
ax.set_ylabel(r"$b$")
fig.text(0.5,-0.01,r'$m='+ str(m) + r',\kappa=' + str(kap) + r',\theta_A=' + str(thea) + r',\theta_B=' + str(theb) + r',a_0=' + str(a0) +r',a_v=' + str(av) + r'$',ha="center")

fig.text(0.5,-0.04,r'$b_0=' + str(b0) +r',b_v=' + str(bv) + r'$',ha="center")

plt.savefig("1d_plots/comp_statics_rho_b_consistent.png",format="png",dpi=600,bbox_inches="tight")






m = 15.0
rho_space = np.linspace(1.0,5.0,1000)
rhos = []
a = []
b = []

for rho in rho_space:
    a_new,b_new = find_all_equilibria(m,thea,theb,kap,rho,a0,b0,av,bv,eps)
    for j in range(0,len(a_new)):
        rhos.append(rho)
        a.append(a_new[j])
        b.append(b_new[j])

fig, ax = plt.subplots(figsize=(5,5))
ax.set_xlabel(r"$\rho$")
ax.scatter(rhos,a,color="black",s=1)
ax.scatter(rhos,b,color=color_b,s=1)
ax.plot([np.nan],[np.nan],color="black",label="$a$")
ax.plot([np.nan],[np.nan],color=color_b,label="$b$")
ax.plot(rho_space,len(rho_space)*[a0],linestyle="--",color="black",label=r"$a_0$")
ax.plot(rho_space,len(rho_space)*[b0],linestyle="--",color=color_b,label=r"$b_0$")
ax.plot(rho_space,len(rho_space)*[a0+av],linestyle="dotted",color="black",label=r"$a_0+a_v$")
ax.plot(rho_space,len(rho_space)*[b0+bv],linestyle="dotted",color=color_b,label=r"$b_0+b_v$")
ax.legend(loc="upper right")

ax.set_ylabel(r"$a,b$")

fig.text(0.5,-0.01,r'$m='+ str(m) + r',\kappa=' + str(kap) + r',\theta_A=' + str(thea) + r',\theta_B=' + str(theb) + r',$',ha="center") 

fig.text(0.5,-0.04,r'$a_0=' + str(a0) +r',a_v=' + str(av) + r',b_0=' + str(b0) + r',b_v=' + str(bv) + r'$',ha="center") 

plt.savefig("1d_plots/comp_statics_rho_high_m.png",format="png",dpi=600,bbox_inches="tight")

fig, ax = plt.subplots(figsize=(5,5))
ax.set_xlabel(r"$\rho$")
a_s = [0.6,0.9,1.2,1.5]
i=0
for a in a_s:
    rhos = []
    b = []

    for rho in rho_space:
        b_new = find_group_br_b(m,theb,kap,rho,b0,bv,a,eps)
        for j in range(0,len(b_new)):
            rhos.append(rho)
            b.append(b_new[j])

    ax.scatter(rhos,b,color=viridis[2*i],s=1)
    ax.plot(np.nan,np.nan,color=viridis[2*i],label=r"$a=" + str(a) + r"$")
    i+=1

ax.legend()
ax.set_ylabel(r"$b$")
fig.text(0.5,-0.01,r'$m='+ str(m) + r',\kappa=' + str(kap) + r',\theta_A=' + str(thea) + r',\theta_B=' + str(theb) + r',a_0=' + str(a0) +r',a_v=' + str(av) + r'$',ha="center")

fig.text(0.5,-0.04,r'$b_0=' + str(b0) +r',b_v=' + str(bv) + r'$',ha="center")

plt.savefig("1d_plots/comp_statics_rho_b_consistent_high_m.png",format="png",dpi=600,bbox_inches="tight")


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
ax.plot([np.nan],[np.nan],color="black",label="$a$")
ax.plot([np.nan],[np.nan],color=color_b,label="$b$")
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
ax.plot([np.nan],[np.nan],color="black",linewidth=2.0,label="$a$")
ax.plot([np.nan],[np.nan],color=color_b,linewidth=1.0,label="$b$")
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

kap = 0.15
m = 42.0
thea = 2.9
theb = thea
a0 = 0.85
b0 = 0.60
av = 0.90
bv = b0*av / a0

rho_vec = np.linspace(1,5.5,400)

a = [[],[],[]]
b = [[],[],[]]
rho_list = [[],[],[]]

for i in range(0,len(rho_vec)):
    rho = rho_vec[i]
    a_new,b_new = find_equilibria_interior(m,thea,theb,kap,rho,a0,b0,av,bv,eps)
    if len(a_new) == 1:
        a[0].append(a_new[0])
        b[0].append(b_new[0])
        rho_list[0].append(rho)
    elif len(a_new) == 2:
        a[0].append(a_new[1])
        b[0].append(b_new[1])
        rho_list[0].append(rho)
        a[1].append(a_new[0])
        b[1].append(b_new[0])
        rho_list[1].append(rho)
    else:
        a[0].append(a_new[2])
        b[0].append(b_new[2])
        rho_list[0].append(rho)
        a[2].append(a_new[1])
        b[2].append(b_new[1])
        rho_list[2].append(rho)
        a[1].append(a_new[0])
        b[1].append(b_new[0])
        rho_list[1].append(rho)
    

fig, ax = plt.subplots(figsize=(5,5))
ax.set_xlabel(r"$\rho$")
for i in range(0,3):
    ax.plot(rho_list[i],np.array(a[i])/av,color=viridis[2*i],linestyle="--")
    ax.plot(rho_list[i],np.array(b[i])/bv,color=viridis[2*i])
    ax.plot(rho_list[i],np.array(rho_list[i])*np.array(a[i])/av,color=viridis[2*i],linestyle="dotted")

ax.plot([np.nan],[np.nan],color="black",linestyle="--",label=r"$\frac{a}{a_v}$")
ax.plot([np.nan],[np.nan],color="black",label=r"$\frac{b}{b_v}$")
ax.plot([np.nan],[np.nan],color="black",linestyle="dotted",label=r"$\rho \frac{a}{a_v}$")    
ax.set_ylabel(r"$a,b$")
ax.legend()
plt.savefig("1d_plots/partial_compensation.png",format="png",dpi=600,bbox_inches="tight")
plt.close()
