import numba as nb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as plc

@nb.njit
def h(x,m):
    return np.arctan(m*x)/np.arctan(m)

@nb.njit
def utility_b(m,the,kap,a0,b0,av,rho,bv,b,bi):
    lam = rho*bv - av
    bkappa = (1-kap)*b + kap*bi
    return lam*h((bkappa-a0)/(bkappa+a0),m) - the*(bi-b0)**2/(2*bv)

@nb.njit
def utility_a(m,the,kap,a0,b0,av,rho,bv,a,ai):
    lam = av - rho * bv
    akappa = (1-kap)*a + kap*ai
    return lam*h((akappa-b0)/(akappa+b0),m) - the*(ai-a0)**2/(2*av)

@nb.njit
def roots_aux_b(m, the, kap, a0, b0, av, rho, bv):
    coeffs = np.empty(4,dtype=np.complex128)

    coeffs[0] = m**2 + 1
    coeffs[1] = -2*a0*m**2 + 2*a0 - b0*m**2 - b0
    coeffs[2] = a0**2*m**2 + a0**2 + 2*a0*b0*m**2 - 2*a0*b0
    coeffs[3] = -a0**2*b0*m**2 - a0**2*b0 + 2*a0*av*bv*kap*m/(the*np.arctan(m)) - 2*a0*bv**2*kap*m*rho/(the*np.arctan(m))

    return np.roots(coeffs)

@nb.njit
def roots_aux_a(m, the, kap, a0, b0, av, rho, bv):
    coeffs = np.empty(4,dtype=np.complex128)

    coeffs[0] = m**2 + 1
    coeffs[1] = -a0*m**2 - a0 - 2*b0*m**2 + 2*b0
    coeffs[2] = 2*a0*b0*m**2 - 2*a0*b0 + b0**2*m**2 + b0**2
    coeffs[3] = -a0*b0**2*m**2 - a0*b0**2 - 2*av**2*b0*kap*m/(the*np.arctan(m)) + 2*av*b0*bv*kap*m*rho/(the*np.arctan(m))

    return np.roots(coeffs)

@nb.njit
def roots_utility_a(m,the,kap,a0,b0,av,rho,bv,a):
    coeffs = np.empty(4,dtype=np.complex128)
    
    coeffs[0] = kap**2*m**2 + kap**2
    coeffs[1] = -2*a*kap**2*m**2 - 2*a*kap**2 + 2*a*kap*m**2 + 2*a*kap - a0*kap**2*m**2 - a0*kap**2 - 2*b0*kap*m**2 + 2*b0*kap
    coeffs[2] = a**2*kap**2*m**2 + a**2*kap**2 - 2*a**2*kap*m**2 - 2*a**2*kap + a**2*m**2 + a**2 + 2*a*a0*kap**2*m**2 + 2*a*a0*kap**2 - 2*a*a0*kap*m**2 - 2*a*a0*kap + 2*a*b0*kap*m**2 - 2*a*b0*kap - 2*a*b0*m**2 + 2*a*b0 + 2*a0*b0*kap*m**2 - 2*a0*b0*kap + b0**2*m**2 + b0**2
    coeffs[3] = -a**2*a0*kap**2*m**2 - a**2*a0*kap**2 + 2*a**2*a0*kap*m**2 + 2*a**2*a0*kap - a**2*a0*m**2 - a**2*a0 - 2*a*a0*b0*kap*m**2 + 2*a*a0*b0*kap + 2*a*a0*b0*m**2 - 2*a*a0*b0 - a0*b0**2*m**2 - a0*b0**2 - 2*av**2*b0*kap*m/(the*np.arctan(m)) + 2*av*b0*bv*kap*m*rho/(the*np.arctan(m))

    return np.roots(coeffs)

@nb.njit  
def roots_utility_b(m,the,kap,a0,b0,av,rho,bv,b):
    coeffs = np.empty(4,dtype=np.complex128)

    coeffs[0] = kap**2*m**2 + kap**2
    coeffs[1] = -2*a0*kap*m**2 + 2*a0*kap - 2*b*kap**2*m**2 - 2*b*kap**2 + 2*b*kap*m**2 + 2*b*kap - b0*kap**2*m**2 - b0*kap**2
    coeffs[2] = a0**2*m**2 + a0**2 + 2*a0*b*kap*m**2 - 2*a0*b*kap - 2*a0*b*m**2 + 2*a0*b + 2*a0*b0*kap*m**2 - 2*a0*b0*kap + b**2*kap**2*m**2 + b**2*kap**2 - 2*b**2*kap*m**2 - 2*b**2*kap + b**2*m**2 + b**2 + 2*b*b0*kap**2*m**2 + 2*b*b0*kap**2 - 2*b*b0*kap*m**2 - 2*b*b0*kap
    coeffs[3] = -a0**2*b0*m**2 - a0**2*b0 + 2*a0*av*bv*kap*m/(the*np.arctan(m)) - 2*a0*b*b0*kap*m**2 + 2*a0*b*b0*kap + 2*a0*b*b0*m**2 - 2*a0*b*b0 - 2*a0*bv**2*kap*m*rho/(the*np.arctan(m)) - b**2*b0*kap**2*m**2 - b**2*b0*kap**2 + 2*b**2*b0*kap*m**2 + 2*b**2*b0*kap - b**2*b0*m**2 - b**2*b0

    return np.roots(coeffs)

@nb.njit
def is_equilibrium_b(roots,m,the,kap,a0,b0,av,rho,bv,eps):
    tf_array = np.zeros(roots.shape[0],dtype=np.bool_)
    for i in range(0,len(roots)):
        roots_u = roots_utility_b(m,the,kap,a0,b0,av,rho,bv,roots[i])
        roots_u = np.real(roots_u[np.abs(np.imag(roots_u)) < eps]) # only real roots
        roots_u = roots_u[roots_u > 0] # only positive roots
        roots_u = roots_u[roots_u < b0+bv] # only roots in admissible range
        roots_u = np.append(roots_u, b0+bv) # append potential corner solution
        index_root = np.argmin(np.abs(roots_u - roots[i]))
        index_max = np.argmax(utility_b(m,the,kap,a0,b0,av,rho,bv,roots[i],roots_u))
        if index_max == index_root:
            tf_array[i] = True
    return tf_array


@nb.njit
def count_equilibria_b(m,the,kap,a0,b0,av,rho,bv,eps):
    if kap > 0:
        roots = roots_aux_b(m,the,kap,a0,b0,av,rho,bv)
        roots = np.real(roots[np.abs(np.imag(roots))< eps]) # only real roots
        roots = roots[roots > b0]
        if np.any(roots > b0 + bv):
            roots = roots[roots < b0 + bv]
            roots = np.append(roots, b0 + bv)
        eq_tf = is_equilibrium_b(roots,m,the,kap,a0,b0,av,rho,bv,eps)
        eqs = np.count_nonzero(eq_tf)
        eqs_winning = np.count_nonzero(np.logical_and(eq_tf,roots>a0))
        return (eqs,eqs_winning)
    else:
        if a0 >= b0:
            return (1,0)
        else:
            return (1,1)

@nb.njit
def get_highest_eq_turnout_b(m,the,kap,a0,b0,av,rho,bv,eps):
    if kap >0:
        roots = roots_aux_b(m,the,kap,a0,b0,av,rho,bv)
        roots = np.real(roots[np.abs(np.imag(roots))< eps]) # only real roots
        roots = roots[roots > b0]
        if np.any(roots > b0 + bv):
            roots = roots[roots < b0 + bv]
            roots = np.append(roots, b0 + bv)
        eq_tf = is_equilibrium_b(roots,m,the,kap,a0,b0,av,rho,bv,eps)
        highest_eq = np.max(roots[eq_tf])
        return highest_eq
    else:
        return b0

@nb.jit(nopython=True,parallel=True)
def calc_turnouts_b(param_arr,eps,arr1,arr2,pos1,pos2,N):
    param_arrs = np.zeros((N,8))
    for i in range(0,N):
        param_arrs[i,:] = param_arr
    turnouts = np.zeros((N,N))
    for i in nb.prange(0,N):
        for j in range(0,N):
            param_arrs[i,pos1] = arr1[i]
            param_arrs[i,pos2] = arr2[j]
            params = (param_arrs[i,0],param_arrs[i,1],param_arrs[i,2],param_arrs[i,3],param_arrs[i,4],param_arrs[i,5],param_arrs[i,6],param_arrs[i,7],eps)
            turnouts[i,j] = get_highest_eq_turnout_b(*params)
    return turnouts

@nb.jit(nopython=True,parallel=True)
def calc_utilities_b(turnouts,param_arr,arr1,arr2,pos1,pos2,N):
    param_arrs = np.zeros((N,8))
    for i in range(0,N):
        param_arrs[i,:] = param_arr
    utilities = np.zeros((N,N))
    for i in nb.prange(0,N):
        for j in range(0,N):
            param_arrs[i,pos1] = arr1[i]
            param_arrs[i,pos2] = arr2[j]
            params = (param_arrs[i,0],param_arrs[i,1],param_arrs[i,2],param_arrs[i,3],param_arrs[i,4],param_arrs[i,5],param_arrs[i,6],param_arrs[i,7],turnouts[i,j],turnouts[i,j])
            utilities[i,j] = utility_b(*params)
    return utilities

@nb.jit(nopython=True,parallel=True)
def counts_equilibria_b(param_arr,eps,arr1,arr2,pos1,pos2,N):
    param_arrs = np.zeros((N,8))
    for i in range(0,N):
        param_arrs[i,:] = param_arr
    amount_eq = np.zeros((N,N))
    amount_eq_winning = np.zeros((N,N))
    for i in nb.prange(0,N):
        for j in range(0,N):
            param_arrs[i,pos1] = arr1[i]
            param_arrs[i,pos2] = arr2[j]
            params = (param_arrs[i,0],param_arrs[i,1],param_arrs[i,2],param_arrs[i,3],param_arrs[i,4],param_arrs[i,5],param_arrs[i,6],param_arrs[i,7],eps)
            amount_eq[i,j], amount_eq_winning[i,j] = count_equilibria_b(*params)
    return amount_eq, amount_eq_winning

def make_param_string(param_arr,pos1,pos2):
    name_dict = {0:'$m$',1:r'$\theta$',2:r'$\kappa$',3:'$a_0$',4:'$b_0$',5:'$a_v$',6:r'$\rho$',7:'$b_v$'}
    poslist = [i for i in range(0,8)]
    poslist.remove(pos1)
    poslist.remove(pos2)

    str1 = ""
    for i in range(0,6):
        str1 += name_dict[poslist[i]] + "=" + str(param_arr[poslist[i]])
        if i < 5:
            str1 += ", "

    return str1

def plot_counts(arr_i,arr_j,amount_eq,amount_eq_winning,name1,name2,fn1,fn2,str1):
    cmap = plt.get_cmap('viridis',4)
    norm = plc.BoundaryNorm([-0.5,0.5,1.5,2.5,3.5],4)
    fig, ax = plt.subplots(figsize=(6.4,4.8))
    mappable = ax.pcolormesh(arr_j,arr_i,amount_eq,cmap=cmap,norm=norm)
    ax.set_xlabel(name1)
    ax.set_ylabel(name2)
    fig.colorbar(ax=ax,mappable=mappable,ticks=np.linspace(0,3,4))
    fig.text(0.5,-0.01,str1,ha="center")
    plt.savefig("2d_plots/vis_multiplicity_" + fn1 + "_"+ fn2 + "_high_res.png",format="png",bbox_inches='tight',dpi=1200)
    plt.savefig("2d_plots/vis_multiplicity_" + fn1 + "_"+ fn2 + "_low_res.png",format="png",bbox_inches='tight',dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(6.4,4.8))
    mappable = ax.pcolormesh(arr_j,arr_i,amount_eq_winning,cmap=cmap,norm=norm)
    ax.set_xlabel(name1)
    ax.set_ylabel(name2)
    fig.colorbar(ax=ax,mappable=mappable,ticks=np.linspace(0,3,4))
    fig.text(0.5,-0.01,str1,ha="center")
    plt.savefig("2d_plots/vis_multiplicity_winning_" + fn1 + "_"+ fn2 + "_high_res.png",format="png",bbox_inches='tight',dpi=1200)
    plt.savefig("2d_plots/vis_multiplicity_winning_" + fn1 + "_"+ fn2 + "_low_res.png",format="png",bbox_inches='tight',dpi=300)
    plt.close()

def plot_turnouts_utilities(arr_i,arr_j,turnouts,utilities,name1,name2,fn1,fn2,str1):
    fig, ax = plt.subplots(figsize=(6.4,4.8))
    mappable = ax.pcolormesh(arr_j,arr_i,turnouts,cmap='viridis')
    ax.set_xlabel(name1)
    ax.set_ylabel(name2)
    fig.colorbar(mappable=mappable,ax=ax)
    fig.text(0.5,-0.01,str1,ha="center")
    plt.savefig("2d_plots/vis_turnouts_" + fn1 + "_"+ fn2 + "_high_res.png",format="png",bbox_inches='tight',dpi=1200)
    plt.savefig("2d_plots/vis_turnouts_" + fn1 + "_"+ fn2 + "_low_res.png",format="png",bbox_inches='tight',dpi=300)
    plt.close()

    # X,Y = np.meshgrid(arr_j,arr_i)
    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # mappable = ax.plot_surface(X,Y,turnouts,cmap='viridis')
    # ax.set_xlabel(name1)
    # ax.set_ylabel(name2)
    # fig.text(0.5,-0.01,str1,ha="center")
    # plt.savefig("2d_plots/vis_turnouts_" + fn1 + "_"+ fn2 + "_high_res.png",format="png",bbox_inches='tight',dpi=1200)
    # plt.savefig("2d_plots/vis_turnouts_" + fn1 + "_"+ fn2 + "_low_res.png",format="png",bbox_inches='tight',dpi=300)
    # plt.close()


    fig, ax = plt.subplots(figsize=(6.4,4.8))
    mappable = ax.pcolormesh(arr_j,arr_i,utilities,cmap='viridis')
    ax.set_xlabel(name1)
    ax.set_ylabel(name2)
    fig.colorbar(ax=ax,mappable=mappable)
    fig.text(0.5,-0.01,str1,ha="center")
    plt.savefig("2d_plots/vis_utilities_" + fn1 + "_"+ fn2 + "_high_res.png",format="png",bbox_inches='tight',dpi=1200)
    plt.savefig("2d_plots/vis_utilities_" + fn1 + "_"+ fn2 + "_low_res.png",format="png",bbox_inches='tight',dpi=300)
    plt.close()

    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # mappable = ax.plot_surface(X,Y,utilities,cmap='viridis')
    # ax.set_xlabel(name1)
    # ax.set_ylabel(name2)
    # fig.text(0.5,-0.01,str1,ha="center")
    # plt.savefig("2d_plots/vis_utilities_" + fn1 + "_"+ fn2 + "_high_res.png",format="png",bbox_inches='tight',dpi=1200)
    # plt.savefig("2d_plots/vis_utilities_" + fn1 + "_"+ fn2 + "_low_res.png",format="png",bbox_inches='tight',dpi=300)
    # plt.close()
    