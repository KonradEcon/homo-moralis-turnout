import numba as nb
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.colors as plc

@nb.njit("f8[:](f8[:],f8)")
def h(x,m):
    return np.arctan(m*x)/np.arctan(m)

@nb.njit("f8[:](f8,f8,f8,f8,f8,f8,f8,f8[:])")
def utility_a(m,the,kap,a0,av,a,b,ai):
    akappa = (1-kap)*a + kap*ai
    return h((akappa-b)/(akappa+b),m) - the*(ai-a0)**2/(2*av**2)

@nb.njit("f8[:](f8,f8,f8,f8,f8,f8,f8,f8,f8[:])")
def utility_b(m,the,kap,rho,b0,bv,a,b,bi):
    bkappa = (1-kap)*b + kap*bi
    return rho * h((bkappa-a)/(bkappa+a),m) - the*(bi-b0)**2/(2*bv**2)

@nb.njit("f8[:](f8,f8,f8,f8,f8,f8[:],f8)")
def aux_a(m,the,kap,a0,av,a,b):
    return kap*h((a-b)/(a+b),m) - the*(a-a0)**2/(2*av**2)

@nb.njit("f8[:](f8,f8,f8,f8,f8,f8,f8,f8[:])")
def aux_b(m,the,kap,rho,b0,bv,a,b):
    return rho*kap*h((b-a)/(b+a),m) - the*(b-b0)**2/(2*bv**2)

@nb.njit("f8[:](f8,f8,f8,f8,f8,f8[:],f8)")
def aux_a_diff(m,the,kap,a0,av,a,b):
    quot = np.arctan(m)*((1+m**2)*(a**2 + b**2) + (2-2*m**2)*a*b)
    cost_ratio = the/av**2
    return 2*m*kap*b / quot - cost_ratio*(a-a0)

@nb.njit("f8[:](f8,f8,f8,f8,f8,f8,f8,f8[:])")
def aux_b_diff(m,the,kap,rho,b0,bv,a,b):
    quot = np.arctan(m)*((1+m**2)*(a**2 + b**2) + (2-2*m**2)*a*b)
    cost_ratio = the/bv**2
    return 2*m*kap*rho*a / quot - cost_ratio*(b-b0)

@nb.njit("c16[:](f8,f8,f8,f8,f8,f8)")
def roots_aux_a(m, thea, kap, a0, av,b):
    coeffs = np.empty(4,dtype=np.complex128)

    coeffs[0] = m**2 + 1
    coeffs[1] = -a0*m**2 - a0 - 2*b*m**2 + 2*b
    coeffs[2] = 2*a0*b*m**2 - 2*a0*b + b**2*m**2 + b**2
    coeffs[3] = -a0*b**2*m**2 - a0*b**2 - 2*av**2*b*kap*m/(thea*np.arctan(m))

    return np.roots(coeffs)

@nb.njit("c16[:](f8,f8,f8,f8,f8,f8,f8)")
def roots_aux_b(m, theb, kap,rho, b0, bv,a):
    coeffs = np.empty(4,dtype=np.complex128)

    coeffs[0] = m**2 + 1
    coeffs[1] = -2*a*m**2 + 2*a - b0*m**2 - b0
    coeffs[2] = a**2*m**2 + a**2 + 2*a*b0*m**2 - 2*a*b0
    coeffs[3] = -a**2*b0*m**2 - a**2*b0 - 2*a*bv**2*kap*m*rho/(theb*np.arctan(m))

    return np.roots(coeffs)

@nb.njit("c16[:](f8,f8,f8,f8,f8,f8,f8)")
def roots_utility_a(m, thea, kap, a0, av,a,b):
    coeffs = np.empty(4,dtype=np.complex128)

    coeffs[0] = kap**2*m**2 + kap**2
    coeffs[1] = -2*a*kap**2*m**2 - 2*a*kap**2 + 2*a*kap*m**2 + 2*a*kap - a0*kap**2*m**2 - a0*kap**2 - 2*b*kap*m**2 + 2*b*kap
    coeffs[2] = a**2*kap**2*m**2 + a**2*kap**2 - 2*a**2*kap*m**2 - 2*a**2*kap + a**2*m**2 + a**2 + 2*a*a0*kap**2*m**2 + 2*a*a0*kap**2 - 2*a*a0*kap*m**2 - 2*a*a0*kap + 2*a*b*kap*m**2 - 2*a*b*kap - 2*a*b*m**2 + 2*a*b + 2*a0*b*kap*m**2 - 2*a0*b*kap + b**2*m**2 + b**2
    coeffs[3] = -a**2*a0*kap**2*m**2 - a**2*a0*kap**2 + 2*a**2*a0*kap*m**2 + 2*a**2*a0*kap - a**2*a0*m**2 - a**2*a0 - 2*a*a0*b*kap*m**2 + 2*a*a0*b*kap + 2*a*a0*b*m**2 - 2*a*a0*b - a0*b**2*m**2 - a0*b**2 - 2*av**2*b*kap*m/(thea*np.arctan(m))

    return np.roots(coeffs)

@nb.njit("c16[:](f8,f8,f8,f8,f8,f8,f8,f8)")
def roots_utility_b(m, theb, kap,rho, b0, bv,a,b):
    coeffs = np.empty(4,dtype=np.complex128)

    coeffs[0] = kap**2*m**2 + kap**2
    coeffs[1] = -2*a*kap*m**2 + 2*a*kap - 2*b*kap**2*m**2 - 2*b*kap**2 + 2*b*kap*m**2 + 2*b*kap - b0*kap**2*m**2 - b0*kap**2
    coeffs[2] = a**2*m**2 + a**2 + 2*a*b*kap*m**2 - 2*a*b*kap - 2*a*b*m**2 + 2*a*b + 2*a*b0*kap*m**2 - 2*a*b0*kap + b**2*kap**2*m**2 + b**2*kap**2 - 2*b**2*kap*m**2 - 2*b**2*kap + b**2*m**2 + b**2 + 2*b*b0*kap**2*m**2 + 2*b*b0*kap**2 - 2*b*b0*kap*m**2 - 2*b*b0*kap
    coeffs[3] = -a**2*b0*m**2 - a**2*b0 - 2*a*b*b0*kap*m**2 + 2*a*b*b0*kap + 2*a*b*b0*m**2 - 2*a*b*b0 - 2*a*bv**2*kap*m*rho/(theb*np.arctan(m)) - b**2*b0*kap**2*m**2 - b**2*b0*kap**2 + 2*b**2*b0*kap*m**2 + 2*b**2*b0*kap - b**2*b0*m**2 - b**2*b0

    return np.roots(coeffs)

@nb.njit("b1[:](f8[:],f8,f8,f8,f8,f8,f8,f8)")
def is_group_br_a(roots,m,the,kap,a0,av,b,eps):
    tf_array = np.zeros(roots.shape[0],dtype=np.bool_)
    for i in range(0,len(roots)):
        roots_u = roots_utility_a(m, the, kap, a0, av,roots[i],b)
        roots_u = np.real(roots_u[np.abs(np.imag(roots_u)) < eps]) # only real roots
        roots_u = roots_u[roots_u > 0] # only positive roots
        roots_u = roots_u[roots_u < a0+av] # only roots in admissible range
        roots_u = np.append(roots_u, a0+av) # append potential corner solution
        index_root = np.argmin(np.abs(roots_u - roots[i]))
        index_max = np.argmax(utility_a(m,the,kap,a0,av,roots[i],b,roots_u))
        if index_max == index_root:
            tf_array[i] = True
    return tf_array

@nb.njit("b1[:](f8[:],f8,f8,f8,f8,f8,f8,f8,f8)")
def is_group_br_b(roots,m,the,kap,rho,b0,bv,a,eps):
    tf_array = np.zeros(roots.shape[0],dtype=np.bool_)
    for i in range(0,len(roots)):
        roots_u = roots_utility_b(m, the, kap,rho, b0, bv,a,roots[i])
        roots_u = np.real(roots_u[np.abs(np.imag(roots_u)) < eps]) # only real roots
        roots_u = roots_u[roots_u > 0] # only positive roots
        roots_u = roots_u[roots_u < b0+bv] # only roots in admissible range
        roots_u = np.append(roots_u, b0+bv) # append potential corner solution
        index_root = np.argmin(np.abs(roots_u - roots[i]))
        index_max = np.argmax(utility_b(m,the,kap,rho,b0,bv,a,roots[i],roots_u))
        if index_max == index_root:
            tf_array[i] = True
    return tf_array

@nb.njit("f8[:](f8,f8,f8,f8,f8,f8,f8)")
def find_group_br_a(m,the,kap,a0,av,b,eps):
    roots = roots_aux_a(m, the, kap,a0,av,b)
    roots = np.real(roots[np.abs(np.imag(roots))< eps]) # only real roots
    if np.any(roots> a0 + av):
        roots = roots[roots < a0 + av] # only admissible
        roots = np.append(roots, a0+av) # append potential corner solution

    tf_array = is_group_br_a(roots,m,the,kap,a0,av,b,eps)
    return roots[tf_array]

@nb.njit("f8[:](f8,f8,f8,f8,f8,f8,f8,f8)")
def find_group_br_b(m,the,kap,rho,b0,bv,a,eps):
    roots = roots_aux_b(m, the, kap,rho,b0,bv,a)
    roots = np.real(roots[np.abs(np.imag(roots))< eps]) # only real roots
    if np.any(roots> b0 + bv):
        roots = roots[roots < b0 + bv] # only admissible roots
        roots = np.append(roots, b0+bv) # append potential corner solution
    
    tf_array = is_group_br_b(roots,m,the,kap,rho,b0,bv,a,eps)
    return roots[tf_array]

def find_group_br_a_vecb(m,the,kap,a0,av,b_vec,eps):
    x = np.zeros(3*len(b_vec))
    y = np.zeros(3*len(b_vec))
    count = 0
    for i in range(0,len(b_vec)):
        br_list = find_group_br_a(m,the,kap,a0,av,b_vec[i],eps)
        for j in range(0,len(br_list)):
            x[count] = br_list[j]
            y[count] = b_vec[i]
            count += 1
    return (x,y)

def find_group_br_b_veca(m,the,kap,rho,b0,bv,a_vec,eps):
    x = np.zeros(3*len(a_vec))
    y = np.zeros(3*len(a_vec))
    count = 0
    for i in range(0,len(a_vec)):
        br_list = find_group_br_b(m,the,kap,rho,b0,bv,a_vec[i],eps)
        for j in range(0,len(br_list)):
            x[count] = a_vec[i]
            y[count] = br_list[j]
            count += 1
    return (x,y)

@nb.njit("c16[:](f8,f8,f8,f8,f8,f8,f8,f8,f8)")
def resultant_roots(m,thea,theb,kap,rho,b0,bv,a0,av):
    coeffs = np.empty(5,dtype=np.complex128)
    
    coeffs[0] = 4*av**4*kap**2*m**8/(thea**2*np.arctan(m)**2) + 12*av**4*kap**2*m**6/(thea**2*np.arctan(m)**2) + 12*av**4*kap**2*m**4/(thea**2*np.arctan(m)**2) + 4*av**4*kap**2*m**2/(thea**2*np.arctan(m)**2) - 8*av**2*bv**2*kap**2*m**8*rho/(thea*theb*np.arctan(m)**2) + 40*av**2*bv**2*kap**2*m**6*rho/(thea*theb*np.arctan(m)**2) + 40*av**2*bv**2*kap**2*m**4*rho/(thea*theb*np.arctan(m)**2) - 8*av**2*bv**2*kap**2*m**2*rho/(thea*theb*np.arctan(m)**2) + 4*bv**4*kap**2*m**8*rho**2/(theb**2*np.arctan(m)**2) + 12*bv**4*kap**2*m**6*rho**2/(theb**2*np.arctan(m)**2) + 12*bv**4*kap**2*m**4*rho**2/(theb**2*np.arctan(m)**2) + 4*bv**4*kap**2*m**2*rho**2/(theb**2*np.arctan(m)**2)
    coeffs[1] = 8*a0*av**2*bv**2*kap**2*m**8*rho/(thea*theb*np.arctan(m)**2) + 8*a0*av**2*bv**2*kap**2*m**6*rho/(thea*theb*np.arctan(m)**2) - 8*a0*av**2*bv**2*kap**2*m**4*rho/(thea*theb*np.arctan(m)**2) - 8*a0*av**2*bv**2*kap**2*m**2*rho/(thea*theb*np.arctan(m)**2) - 8*a0*bv**4*kap**2*m**8*rho**2/(theb**2*np.arctan(m)**2) - 8*a0*bv**4*kap**2*m**6*rho**2/(theb**2*np.arctan(m)**2) + 8*a0*bv**4*kap**2*m**4*rho**2/(theb**2*np.arctan(m)**2) + 8*a0*bv**4*kap**2*m**2*rho**2/(theb**2*np.arctan(m)**2) - 12*av**4*b0*kap**2*m**8/(thea**2*np.arctan(m)**2) - 36*av**4*b0*kap**2*m**6/(thea**2*np.arctan(m)**2) - 36*av**4*b0*kap**2*m**4/(thea**2*np.arctan(m)**2) - 12*av**4*b0*kap**2*m**2/(thea**2*np.arctan(m)**2) + 16*av**2*b0*bv**2*kap**2*m**8*rho/(thea*theb*np.arctan(m)**2) - 80*av**2*b0*bv**2*kap**2*m**6*rho/(thea*theb*np.arctan(m)**2) - 80*av**2*b0*bv**2*kap**2*m**4*rho/(thea*theb*np.arctan(m)**2) + 16*av**2*b0*bv**2*kap**2*m**2*rho/(thea*theb*np.arctan(m)**2) - 4*b0*bv**4*kap**2*m**8*rho**2/(theb**2*np.arctan(m)**2) - 12*b0*bv**4*kap**2*m**6*rho**2/(theb**2*np.arctan(m)**2) - 12*b0*bv**4*kap**2*m**4*rho**2/(theb**2*np.arctan(m)**2) - 4*b0*bv**4*kap**2*m**2*rho**2/(theb**2*np.arctan(m)**2)
    coeffs[2] = 4*a0**2*bv**4*kap**2*m**8*rho**2/(theb**2*np.arctan(m)**2) + 12*a0**2*bv**4*kap**2*m**6*rho**2/(theb**2*np.arctan(m)**2) + 12*a0**2*bv**4*kap**2*m**4*rho**2/(theb**2*np.arctan(m)**2) + 4*a0**2*bv**4*kap**2*m**2*rho**2/(theb**2*np.arctan(m)**2) - 16*a0*av**2*b0*bv**2*kap**2*m**8*rho/(thea*theb*np.arctan(m)**2) - 16*a0*av**2*b0*bv**2*kap**2*m**6*rho/(thea*theb*np.arctan(m)**2) + 16*a0*av**2*b0*bv**2*kap**2*m**4*rho/(thea*theb*np.arctan(m)**2) + 16*a0*av**2*b0*bv**2*kap**2*m**2*rho/(thea*theb*np.arctan(m)**2) + 8*a0*b0*bv**4*kap**2*m**8*rho**2/(theb**2*np.arctan(m)**2) + 8*a0*b0*bv**4*kap**2*m**6*rho**2/(theb**2*np.arctan(m)**2) - 8*a0*b0*bv**4*kap**2*m**4*rho**2/(theb**2*np.arctan(m)**2) - 8*a0*b0*bv**4*kap**2*m**2*rho**2/(theb**2*np.arctan(m)**2) + 12*av**4*b0**2*kap**2*m**8/(thea**2*np.arctan(m)**2) + 36*av**4*b0**2*kap**2*m**6/(thea**2*np.arctan(m)**2) + 36*av**4*b0**2*kap**2*m**4/(thea**2*np.arctan(m)**2) + 12*av**4*b0**2*kap**2*m**2/(thea**2*np.arctan(m)**2) - 8*av**2*b0**2*bv**2*kap**2*m**8*rho/(thea*theb*np.arctan(m)**2) + 40*av**2*b0**2*bv**2*kap**2*m**6*rho/(thea*theb*np.arctan(m)**2) + 40*av**2*b0**2*bv**2*kap**2*m**4*rho/(thea*theb*np.arctan(m)**2) - 8*av**2*b0**2*bv**2*kap**2*m**2*rho/(thea*theb*np.arctan(m)**2) - 32*av**2*bv**4*kap**3*m**7*rho**2/(thea*theb**2*np.arctan(m)**3) + 32*av**2*bv**4*kap**3*m**3*rho**2/(thea*theb**2*np.arctan(m)**3)
    coeffs[3] = -4*a0**2*b0*bv**4*kap**2*m**8*rho**2/(theb**2*np.arctan(m)**2) - 12*a0**2*b0*bv**4*kap**2*m**6*rho**2/(theb**2*np.arctan(m)**2) - 12*a0**2*b0*bv**4*kap**2*m**4*rho**2/(theb**2*np.arctan(m)**2) - 4*a0**2*b0*bv**4*kap**2*m**2*rho**2/(theb**2*np.arctan(m)**2) + 8*a0*av**2*b0**2*bv**2*kap**2*m**8*rho/(thea*theb*np.arctan(m)**2) + 8*a0*av**2*b0**2*bv**2*kap**2*m**6*rho/(thea*theb*np.arctan(m)**2) - 8*a0*av**2*b0**2*bv**2*kap**2*m**4*rho/(thea*theb*np.arctan(m)**2) - 8*a0*av**2*b0**2*bv**2*kap**2*m**2*rho/(thea*theb*np.arctan(m)**2) + 8*a0*av**2*bv**4*kap**3*m**7*rho**2/(thea*theb**2*np.arctan(m)**3) + 16*a0*av**2*bv**4*kap**3*m**5*rho**2/(thea*theb**2*np.arctan(m)**3) + 8*a0*av**2*bv**4*kap**3*m**3*rho**2/(thea*theb**2*np.arctan(m)**3) - 8*a0*bv**6*kap**3*m**7*rho**3/(theb**3*np.arctan(m)**3) - 16*a0*bv**6*kap**3*m**5*rho**3/(theb**3*np.arctan(m)**3) - 8*a0*bv**6*kap**3*m**3*rho**3/(theb**3*np.arctan(m)**3) - 4*av**4*b0**3*kap**2*m**8/(thea**2*np.arctan(m)**2) - 12*av**4*b0**3*kap**2*m**6/(thea**2*np.arctan(m)**2) - 12*av**4*b0**3*kap**2*m**4/(thea**2*np.arctan(m)**2) - 4*av**4*b0**3*kap**2*m**2/(thea**2*np.arctan(m)**2) + 32*av**2*b0*bv**4*kap**3*m**7*rho**2/(thea*theb**2*np.arctan(m)**3) - 32*av**2*b0*bv**4*kap**3*m**3*rho**2/(thea*theb**2*np.arctan(m)**3)
    coeffs[4] = -8*a0*av**2*b0*bv**4*kap**3*m**7*rho**2/(thea*theb**2*np.arctan(m)**3) - 16*a0*av**2*b0*bv**4*kap**3*m**5*rho**2/(thea*theb**2*np.arctan(m)**3) - 8*a0*av**2*b0*bv**4*kap**3*m**3*rho**2/(thea*theb**2*np.arctan(m)**3) - 16*av**2*bv**6*kap**4*m**6*rho**3/(thea*theb**3*np.arctan(m)**4) - 16*av**2*bv**6*kap**4*m**4*rho**3/(thea*theb**3*np.arctan(m)**4)

    return np.roots(coeffs)

@nb.njit("UniTuple(f8[:],2)(c16[:],f8,f8,f8,f8,f8,f8,f8,f8)")
def candidates_from_roots(roots,rho,thea,theb,a0,b0,av,bv,eps):
    b_array = np.real(roots[np.abs(np.imag(roots)) < eps]) # only real roots
    b_array = b_array[b_array > b0]
    b_array = b_array[b_array < b0+bv] # only roots in admissible range


    a_array = a0 /2 + np.sqrt(a0**2/ 4 + (theb*av**2/(rho*bv**2*thea))*(b_array**2 - b0*b_array))

    tf_array = np.logical_and(a_array > a0, a_array < a0+av)

    return (a_array[tf_array],b_array[tf_array])

@nb.njit("b1(f8,f8,f8,f8,f8,f8,f8,f8)")
def is_group_br_a_single(a,m,the,kap,a0,av,b,eps):
    roots = roots_utility_a(m, the, kap, a0, av,a,b)
    roots = np.real(roots[np.abs(np.imag(roots)) < eps]) # only real roots
    roots = roots[roots > 0] # only positive roots
    roots = roots[roots < a0+av] # only roots in admissible range
    roots = np.append(roots, a0+av) # append potential corner solution
    index_root = np.argmin(np.abs(roots - a))
    index_max = np.argmax(utility_a(m,the,kap,a0,av,a,b,roots))
    return index_max == index_root

@nb.njit("b1(f8,f8,f8,f8,f8,f8,f8,f8,f8)")
def is_group_br_b_single(b,m,the,kap,rho,b0,bv,a,eps):
    roots = roots_utility_b(m, the, kap,rho, b0, bv,a,b)
    roots = np.real(roots[np.abs(np.imag(roots)) < eps]) # only real roots
    roots = roots[roots > 0]
    roots = roots[roots < b0+bv]
    roots = np.append(roots, b0+bv)
    index_root = np.argmin(np.abs(roots - b))
    index_max = np.argmax(utility_b(m,the,kap,rho,b0,bv,a,b,roots))
    return index_max == index_root

@nb.njit("UniTuple(f8[:],2)(f8,f8,f8,f8,f8,f8,f8,f8,f8,f8)")
def find_all_equilibria(m,thea,theb,kap,rho,a0,b0,av,bv,eps):
    if kap > 0:
        roots = resultant_roots(m,thea,theb,kap,rho,b0,bv,a0,av)
        cand_tup = candidates_from_roots(roots,rho,thea,theb,a0,b0,av,bv,eps)
        len_int = len(cand_tup[0])
        corner_cand_a = find_group_br_a(m,thea,kap,a0,av,b0+bv,eps)
        len_a = len(corner_cand_a)
        a_array = np.append(cand_tup[0],corner_cand_a)
        b_array = np.append(cand_tup[1],b0+bv*np.ones(len_a))
        corner_cand_b = find_group_br_b(m,theb,kap,rho,b0,bv,a0+av,eps)
        len_b = len(corner_cand_b)
        if corner_cand_a[-1] == a0 + av:
            if len_b > 1:
                a_array = np.append(a_array,a0+av*np.ones(len_b-1))
                b_array = np.append(b_array,corner_cand_b[:-1])
            len_b -= 1
        else:
            a_array = np.append(a_array,a0+av*np.ones(len_b))
            b_array = np.append(b_array,corner_cand_b)

        tf_array = np.ones(len(a_array),dtype=np.bool_)
        for i in range(0,len_int):
            tf_array[i] = is_group_br_a_single(a_array[i],m,thea,kap,a0,av,b_array[i],eps) and is_group_br_b_single(b_array[i],m,theb,kap,rho,b0,bv,a_array[i],eps)
        for i in range(len_int,len_int+len_a):
            tf_array[i] = is_group_br_b_single(b_array[i],m,theb,kap,rho,b0,bv,a_array[i],eps)
        for i in range(len_int+len_a,len_int+len_a+len_b):
            tf_array[i] = is_group_br_a_single(a_array[i],m,thea,kap,a0,av,b_array[i],eps)
        return (a_array[tf_array],b_array[tf_array])
    else:
        return (np.array([a0]),np.array([b0]))

@nb.njit("UniTuple(f8[:],2)(f8,f8,f8,f8,f8,f8,f8,f8,f8,f8)")
def find_all_interior(m,thea,theb,kap,rho,a0,b0,av,bv,eps):
    roots = resultant_roots(m,thea,theb,kap,rho,b0,bv,a0,av)
    cand_tup = candidates_from_roots(roots,rho,thea,theb,a0,b0,av,bv,eps)
    len_int = len(cand_tup[0])
    a_array = cand_tup[0]
    b_array = cand_tup[1]
    tf_array = np.ones(len(a_array),dtype=np.bool_)
    for i in range(0,len_int):
        tf_array[i] = is_group_br_a_single(a_array[i],m,thea,kap,a0,av,b_array[i],eps) and is_group_br_b_single(b_array[i],m,theb,kap,rho,b0,bv,a_array[i],eps)
    return (a_array[tf_array],b_array[tf_array])

@nb.njit("i8(f8,f8,f8,f8,f8,f8,f8,f8,f8,f8)")
def count_all_equilibria(m,thea,theb,kap,rho,a0,b0,av,bv,eps):
    return find_all_equilibria(m,thea,theb,kap,rho,a0,b0,av,bv,eps)[0].shape[0]

@nb.njit(parallel=True)
def counts_turnouts_utilities(param_arr,eps,arr1,arr2,pos1,pos2,N):
    param_arrs = np.zeros((N,9))
    for i in range(0,N):
        param_arrs[i,:] = param_arr

    counts = np.zeros((N,N),dtype=np.int64)
    counts_a_winning = np.zeros((N,N),dtype=np.int64)
    counts_b_winning = np.zeros((N,N),dtype=np.int64)
    turnouts_a = 100 * np.ones((N,N),dtype=np.float64)
    turnouts_b = 100 * np.ones((N,N),dtype=np.float64)
    utilities_a = 100 * np.ones((N,N),dtype=np.float64)
    utilities_b = 100 * np.ones((N,N),dtype=np.float64)

    for i in nb.prange(0,N):
        for j in range(0,N):
            param_arrs[i,pos1] = arr1[i]
            param_arrs[i,pos2] = arr2[j]

            eq = find_all_equilibria(param_arrs[i,0],param_arrs[i,1],param_arrs[i,2],param_arrs[i,3],param_arrs[i,4],param_arrs[i,5],param_arrs[i,6],param_arrs[i,7],param_arrs[i,8],eps)

            counts[i,j] = eq[0].shape[0]
            counts_a_winning[i,j] = np.sum(eq[0] > eq[1])
            counts_b_winning[i,j] = np.sum(eq[0] < eq[1])

            if counts[i,j] > 0:
                turnouts_a[i,j] = np.max(eq[0])
                maxindex_a = np.argmax(eq[0])
                turnouts_b[i,j] = np.max(eq[1])
                maxindex_b = np.argmax(eq[1])

                utilities_a[i,j] = utility_a(param_arrs[i,0],param_arrs[i,1],param_arrs[i,3],param_arrs[i,5],param_arrs[i,7],turnouts_a[i,j],eq[1][maxindex_a],np.array([turnouts_a[i,j]]))[0]
                utilities_b[i,j] = utility_b(param_arrs[i,0],param_arrs[i,2],param_arrs[i,3],param_arrs[i,4],param_arrs[i,6],param_arrs[i,8],eq[0][maxindex_b],turnouts_b[i,j],np.array([turnouts_b[i,j]]))[0]

    return (counts,counts_a_winning,counts_b_winning,turnouts_a,turnouts_b,utilities_a,utilities_b)

def make_param_string(param_arr,pos1,pos2):
    name_dict = {0:'$m=$',1:r'$\theta_A=$',2: r'$\theta_B=$',3:r'$\kappa=$',5:'$a_0=$',6:'$b_0=$',7:'$a_v=$',4:r'$\rho=$',8:r'$b_v=\frac{b_0 a_v}{a_0}\approx$'}
    poslist = [i for i in range(0,9)]
    poslist.remove(pos1)
    poslist.remove(pos2)

    str1 = ""
    for i in range(0,7):
        if poslist[i] == 8:
            str1 += name_dict[poslist[i]] + str(np.round(param_arr[poslist[i]],2))
        else:
            str1 += name_dict[poslist[i]] + str(param_arr[poslist[i]])
        if i < 6:
            str1 += ", "

    return str1

def make_param_string_all(param_arr):
    name_dict = {0:'$m=$',1:r'$\theta_A=$',2: r'$\theta_B=$',3:r'$\kappa=$',5:'$a_0=$',6:'$b_0=$',7:'$a_v=$',4:r'$\rho=$',8:r'$b_v=\frac{b_0 a_v}{a_0}\approx$'}
    poslist = [i for i in range(0,9)]

    str1 = ""
    for i in range(0,9):
        if poslist[i] == 8:
            str1 += name_dict[poslist[i]] + str(np.round(param_arr[poslist[i]],2))
        else:
            str1 += name_dict[poslist[i]] + str(param_arr[poslist[i]])
        if i < 8:
            str1 += ", "

    return str1

def plot_counts(arr_i,arr_j,amount_eq,a_winning,b_winning,name1,name2,fn1,fn2,str1):
    cmap_len = np.max(amount_eq)+1
    cmap = plt.get_cmap('viridis',cmap_len)
    norm = plc.BoundaryNorm([-0.5 + i for i in range(0,cmap_len+1)],cmap_len)

    fig = plt.figure(figsize=(6,5),layout=None)
    ax = fig.add_axes([0.09,0.15,0.95,0.82],anchor="C")
    mappable = ax.pcolormesh(arr_j,arr_i,amount_eq,cmap=cmap,norm=norm)
    ax.set_xlabel(name1)
    ax.set_ylabel(name2)
    fig.colorbar(mappable=mappable,ax=ax,ticks=np.linspace(0,cmap_len-1,cmap_len))
    fig.text(0.5,0.02,str1,ha='center')
    plt.savefig("2d_plots/vis_multiplicity_ex_post_" + fn1 + "_"+ fn2 + "_high_res.png",format="png",dpi=1200)
    plt.savefig("2d_plots/vis_multiplicity_ex_post_" + fn1 + "_"+ fn2 + "_low_res.png",format="png",dpi=300)
    plt.close()

    fig = plt.figure(figsize=(6,5),layout=None)
    ax = fig.add_axes([0.09,0.15,0.95,0.82],anchor="C")
    mappable = ax.pcolormesh(arr_j,arr_i,a_winning,cmap=cmap,norm=norm)
    ax.set_xlabel(name1)
    ax.set_ylabel(name2)
    fig.colorbar(mappable=mappable,ax=ax,ticks=np.linspace(0,cmap_len-1,cmap_len))
    fig.text(0.5,0.02,str1,ha='center')
    plt.savefig("2d_plots/vis_multiplicity_ex_post_A_winning_" + fn1 + "_"+ fn2 + "_high_res.png",format="png",dpi=1200)
    plt.savefig("2d_plots/vis_multiplicity_ex_post_A_winning_" + fn1 + "_"+ fn2 + "_low_res.png",format="png",dpi=300)
    plt.close()

    fig = plt.figure(figsize=(6,5),layout=None)
    ax = fig.add_axes([0.09,0.15,0.95,0.82],anchor="C")
    mappable = ax.pcolormesh(arr_j,arr_i,b_winning,cmap=cmap,norm=norm)
    ax.set_xlabel(name1)
    ax.set_ylabel(name2)
    fig.colorbar(mappable=mappable,ax=ax,ticks=np.linspace(0,cmap_len-1,cmap_len))
    fig.text(0.5,0.02,str1,ha='center')
    plt.savefig("2d_plots/vis_multiplicity_ex_post_B_winning_" + fn1 + "_"+ fn2 + "_high_res.png",format="png",dpi=1200)
    plt.savefig("2d_plots/vis_multiplicity_ex_post_B_winning_" + fn1 + "_"+ fn2 + "_low_res.png",format="png",dpi=300)
    plt.close()

def plot_turnouts_utilities(arr_i,arr_j,turnouts_a,turnouts_b,utilities_a,utilities_b,name1,name2,fn1,fn2,str1):
    turnouts_a[turnouts_a>99] = np.NaN
    utilities_a[utilities_a>99] = np.NaN
    turnouts_b[turnouts_b>99] = np.NaN
    utilities_b[utilities_b>99] = np.NaN
    X,Y = np.meshgrid(arr_j,arr_i)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    mappable = ax.plot_surface(X,Y,turnouts_a,cmap='viridis')
    ax.set_xlabel(name1)
    ax.set_ylabel(name2)
    # fig.colorbar(mappable=mappable,ax=ax)
    fig.text(0.01,-0.01,str1,ha='left')
    plt.savefig("2d_plots/vis_turnouts_A_ex_post_" + fn1 + "_"+ fn2 + "_high_res.png",format="png",dpi=1200)
    plt.savefig("2d_plots/vis_turnouts_A_ex_post_" + fn1 + "_"+ fn2 + "_low_res.png",format="png",dpi=300)
    plt.close()

    X,Y = np.meshgrid(arr_j,arr_i)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    mappable = ax.plot_surface(X,Y,turnouts_b,cmap='viridis')
    ax.set_xlabel(name1)
    ax.set_ylabel(name2)
    # fig.colorbar(mappable=mappable,ax=ax)
    fig.text(0.01,-0.01,str1,ha='left')
    plt.savefig("2d_plots/vis_turnouts_B_ex_post_" + fn1 + "_"+ fn2 + "_high_res.png",format="png",dpi=1200)
    plt.savefig("2d_plots/vis_turnouts_B_ex_post_" + fn1 + "_"+ fn2 + "_low_res.png",format="png",dpi=300)
    plt.close()

    X,Y = np.meshgrid(arr_j,arr_i)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    mappable = ax.plot_surface(X,Y,utilities_a,cmap='viridis')
    ax.set_xlabel(name1)
    ax.set_ylabel(name2)
    # fig.colorbar(mappable=mappable,ax=ax)
    fig.text(0.01,-0.01,str1,ha='left')
    plt.savefig("2d_plots/vis_utilities_A_ex_post_" + fn1 + "_"+ fn2 + "_high_res.png",format="png",dpi=1200)
    plt.savefig("2d_plots/vis_utilities_A_ex_post_" + fn1 + "_"+ fn2 + "_low_res.png",format="png",dpi=300)
    plt.close()

    X,Y = np.meshgrid(arr_j,arr_i)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    mappable = ax.plot_surface(X,Y,utilities_b,cmap='viridis')
    ax.set_xlabel(name1)
    ax.set_ylabel(name2)
    # fig.colorbar(mappable=mappable,ax=ax)
    fig.text(0.01,-0.01,str1,ha='left')
    plt.savefig("2d_plots/vis_utilities_B_ex_post_" + fn1 + "_"+ fn2 + "_high_res.png",format="png",dpi=1200)
    plt.savefig("2d_plots/vis_utilities_B_ex_post_" + fn1 + "_"+ fn2 + "_low_res.png",format="png",dpi=300)
    plt.close()

def plot_turnouts_utilities_flat(arr_i,arr_j,turnouts_a,turnouts_b,utilities_a,utilities_b,name1,name2,fn1,fn2,str1):
    turnouts_a[turnouts_a>99] = np.NaN
    utilities_a[utilities_a>99] = np.NaN
    turnouts_b[turnouts_b>99] = np.NaN
    utilities_b[utilities_b>99] = np.NaN
    fig = plt.figure(figsize=(6,5),layout=None)
    ax = fig.add_axes([0.09,0.15,0.95,0.82],anchor="C")
    mappable = ax.pcolormesh(arr_j,arr_i,turnouts_a,cmap='viridis')
    ax.set_xlabel(name1)
    ax.set_ylabel(name2)
    fig.colorbar(mappable=mappable,ax=ax)
    fig.text(0.5,0.02,str1,ha='center')
    plt.savefig("2d_plots/vis_turnouts_A_ex_post_" + fn1 + "_"+ fn2 + "_high_res.png",format="png",dpi=1200)
    plt.savefig("2d_plots/vis_turnouts_A_ex_post_" + fn1 + "_"+ fn2 + "_low_res.png",format="png",dpi=300)
    plt.close()

    fig = plt.figure(figsize=(6,5),layout=None)
    ax = fig.add_axes([0.09,0.15,0.95,0.82],anchor="C")
    mappable = ax.pcolormesh(arr_j,arr_i,turnouts_b,cmap='viridis')
    ax.set_xlabel(name1)
    ax.set_ylabel(name2)
    fig.colorbar(mappable=mappable,ax=ax)
    fig.text(0.5,0.02,str1,ha='center')
    plt.savefig("2d_plots/vis_turnouts_B_ex_post_" + fn1 + "_"+ fn2 + "_high_res.png",format="png",dpi=1200)
    plt.savefig("2d_plots/vis_turnouts_B_ex_post_" + fn1 + "_"+ fn2 + "_low_res.png",format="png",dpi=300)
    plt.close()

    fig = plt.figure(figsize=(6,5),layout=None)
    ax = fig.add_axes([0.09,0.15,0.95,0.82],anchor="C")
    mappable = ax.pcolormesh(arr_j,arr_i,utilities_a,cmap='viridis')
    ax.set_xlabel(name1)
    ax.set_ylabel(name2)
    fig.colorbar(mappable=mappable,ax=ax)
    fig.text(0.5,0.02,str1,ha='center')
    plt.savefig("2d_plots/vis_utilities_A_ex_post_" + fn1 + "_"+ fn2 + "_high_res.png",format="png",dpi=1200)
    plt.savefig("2d_plots/vis_utilities_A_ex_post_" + fn1 + "_"+ fn2 + "_low_res.png",format="png",dpi=300)
    plt.close()

    fig = plt.figure(figsize=(6,5),layout=None)
    ax = fig.add_axes([0.09,0.15,0.95,0.82],anchor="C")
    mappable = ax.pcolormesh(arr_j,arr_i,utilities_b,cmap='viridis')
    ax.set_xlabel(name1)
    ax.set_ylabel(name2)
    fig.colorbar(mappable=mappable,ax=ax)
    fig.text(0.5,0.02,str1,ha='center')
    plt.savefig("2d_plots/vis_utilities_B_ex_post_" + fn1 + "_"+ fn2 + "_high_res.png",format="png",dpi=1200)
    plt.savefig("2d_plots/vis_utilities_B_ex_post_" + fn1 + "_"+ fn2 + "_low_res.png",format="png",dpi=300, pad_inches=0)
    plt.close()

@nb.njit("Tuple((i8[:,:,:,:,:,:,:,:,:],f8[:,:,:,:,:,:,:,:,:,:,:]))(i8,f8)",parallel=True)
def find_equilibria(step,eps):
    m_arr = np.linspace(0.1,40.0,step)
    rho_arr = np.linspace(1.0,5.0,step)
    thea_arr = np.linspace(0.1,5.0,step)
    theb_arr = np.linspace(0.1,5.0,step)
    a0_arr = np.linspace(0.1,1.0,step)
    b0_arr = np.linspace(0.1,1.0,step)
    av_arr = np.linspace(0.1,2.0,step)
    bv_arr = np.linspace(0.1,2.0,step)
    kap_arr = np.linspace(0.0,1.0,step)

    counts = np.zeros((step,step,step,step,step,step,step,step,step),dtype=np.int64)
    eqs = np.zeros((2,10,step,step,step,step,step,step,step,step,step),dtype=np.float64)
    nrange = range(0,step)
    Prange = nb.prange(0,step*step)
    for i in Prange:
        with nb.objmode():
            print("starting " + str(i) + " out of " + str(step*step-1),flush=True)
            print("at " + str(time.time()),flush=True)
        i1 = i % step
        i0 = i // step
        for i2 in nrange:
            for i3 in nrange:
                for i4 in nrange:
                    for i5 in nrange:
                        for i6 in nrange:
                            for i7 in nrange:
                                for i8 in nrange:
                                    eq = find_all_equilibria(m_arr[i0],thea_arr[i1],theb_arr[i2],kap_arr[i3],rho_arr[i4],a0_arr[i5],b0_arr[i6],av_arr[i7],bv_arr[i8],eps)
                                    len_a = eq[0].shape[0]
                                    counts[i0,i1,i2,i3,i4,i5,i6,i7,i8] = len_a
                                    for j in range(0,len_a):
                                        eqs[0,j,i0,i1,i2,i3,i4,i5,i6,i7,i8] = eq[0][j]
                                        eqs[1,j,i0,i1,i2,i3,i4,i5,i6,i7,i8] = eq[1][j]
        with nb.objmode():
            print("finished " + str(i) + " out of " + str(step*step-1),flush=True)
            print("at " + str(time.time()),flush=True)
    return (counts,eqs)

@nb.njit("Tuple((i8[:,:,:,:,:,:,:],f8[:,:,:,:,:,:,:,:,:]))(i8,f8)",parallel=True)
def find_equilibria_same_cost_structure(step,eps):
    m_arr = np.linspace(0.1,40.0,step)
    rho_arr = np.linspace(1.0,5.0,step)
    the_arr = np.linspace(0.1,5.0,step)
    a0_arr = np.linspace(0.1,1.0,step)
    b0_arr = np.linspace(0.1,1.0,step)
    av_arr = np.linspace(0.1,2.0,step)
    kap_arr = np.linspace(0.0,1.0,step)

    counts = np.zeros((step,step,step,step,step,step,step),dtype=np.int64)
    eqs = np.zeros((2,10,step,step,step,step,step,step,step),dtype=np.float64)
    nrange = range(0,step)
    Prange = nb.prange(0,step*step)
    for i in Prange:
        i1 = i % step
        i0 = i // step
        for i2 in nrange:
            for i3 in nrange:
                for i4 in nrange:
                    for i5 in nrange:
                        for i6 in nrange:
                            eq = find_all_equilibria(m_arr[i0],the_arr[i1],the_arr[i1],kap_arr[i2],rho_arr[i3],a0_arr[i4],b0_arr[i5],av_arr[i6],av_arr[i6]*b0_arr[i5]/a0_arr[i4],eps)
                            len_a = eq[0].shape[0]
                            counts[i0,i1,i2,i3,i4,i5,i6] = len_a
                            for j in range(0,len_a):
                                eqs[0,j,i0,i1,i2,i3,i4,i5,i6] = eq[0][j]
                                eqs[1,j,i0,i1,i2,i3,i4,i5,i6] = eq[1][j]
    return (counts,eqs)

@nb.njit("Tuple((i8[:,:,:,:,:,:,:,:,:],f8[:,:,:,:,:,:,:,:,:,:,:]))(i8,f8)",parallel=True)
def find_equilibria(step,eps):
    m_arr = np.linspace(0.1,40.0,step)
    rho_arr = np.linspace(1.0,5.0,step)
    thea_arr = np.linspace(0.1,5.0,step)
    theb_arr = np.linspace(0.1,5.0,step)
    a0_arr = np.linspace(0.1,1.0,step)
    b0_arr = np.linspace(0.1,1.0,step)
    av_arr = np.linspace(0.1,2.0,step)
    bv_arr = np.linspace(0.1,2.0,step)
    kap_arr = np.linspace(0.0,1.0,step)

    counts = np.zeros((step,step,step,step,step,step,step,step,step),dtype=np.int64)
    eqs = np.zeros((2,10,step,step,step,step,step,step,step,step,step),dtype=np.float64)
    nrange = range(0,step)
    Prange = nb.prange(0,step*step)
    for i in Prange:
        i1 = i % step
        i0 = i // step
        for i2 in nrange:
            for i3 in nrange:
                for i4 in nrange:
                    for i5 in nrange:
                        for i6 in nrange:
                            for i7 in nrange:
                                for i8 in nrange:
                                    eq = find_all_equilibria(m_arr[i0],thea_arr[i1],theb_arr[i2],kap_arr[i3],rho_arr[i4],a0_arr[i5],b0_arr[i6],av_arr[i7],bv_arr[i8],eps)
                                    len_a = eq[0].shape[0]
                                    counts[i0,i1,i2,i3,i4,i5,i6,i7,i8] = len_a
                                    for j in range(0,len_a):
                                        eqs[0,j,i0,i1,i2,i3,i4,i5,i6,i7,i8] = eq[0][j]
                                        eqs[1,j,i0,i1,i2,i3,i4,i5,i6,i7,i8] = eq[1][j]
    return (counts,eqs)

@nb.njit("Tuple((i8[:,:,:,:,:,:,:],f8[:,:,:,:,:,:,:,:,:]))(i8,f8)",parallel=True)
def find_equilibria_same_cost_structure_interior(step,eps):
    m_arr = np.linspace(0.1,40.0,step)
    rho_arr = np.linspace(1.0,5.0,step)
    the_arr = np.linspace(0.1,5.0,step)
    a0_arr = np.linspace(0.1,1.0,step)
    b0_arr = np.linspace(0.1,1.0,step)
    av_arr = np.linspace(0.1,2.0,step)
    kap_arr = np.linspace(0.0,1.0,step)

    counts = np.zeros((step,step,step,step,step,step,step),dtype=np.int64)
    eqs = np.zeros((2,4,step,step,step,step,step,step,step),dtype=np.float64)
    nrange = range(0,step)
    Prange = nb.prange(0,step*step)
    for i in Prange:
        i1 = i % step
        i0 = i // step
        for i2 in nrange:
            for i3 in nrange:
                for i4 in nrange:
                    for i5 in nrange:
                        for i6 in nrange:
                            eq = find_all_interior(m_arr[i0],the_arr[i1],the_arr[i1],kap_arr[i2],rho_arr[i3],a0_arr[i4],b0_arr[i5],av_arr[i6],av_arr[i6]*b0_arr[i5]/a0_arr[i4],eps)
                            len_a = eq[0].shape[0]
                            counts[i0,i1,i2,i3,i4,i5,i6] = len_a
                            for j in range(0,len_a):
                                eqs[0,j,i0,i1,i2,i3,i4,i5,i6] = eq[0][j]
                                eqs[1,j,i0,i1,i2,i3,i4,i5,i6] = eq[1][j]
    return (counts,eqs)
