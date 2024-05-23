from numba.pycc import CC
import numba as nb
import numpy as np

cc = CC('ex_post_funs')

@nb.njit("f8[:](f8[:],f8)")
@cc.export("h", "f8[:](f8[:],f8)")
def h(x,m):
    return np.arctan(m*x)/np.arctan(m)

@nb.njit("f8[:](f8,f8,f8,f8,f8,f8,f8,f8[:])")
@cc.export("utility_a", "f8[:](f8,f8,f8,f8,f8,f8,f8,f8[:])")
def utility_a(m,the,kap,a0,av,a,b,ai):
    akappa = (1-kap)*a + kap*ai
    return h((akappa-b)/(akappa+b),m) - the*(ai-a0)**2/(2*av**2)

@nb.njit("f8[:](f8,f8,f8,f8,f8,f8,f8,f8,f8[:])")
@cc.export("utility_b", "f8[:](f8,f8,f8,f8,f8,f8,f8,f8,f8[:])")
def utility_b(m,the,kap,rho,b0,bv,a,b,bi):
    bkappa = (1-kap)*b + kap*bi
    return rho * h((bkappa-a)/(bkappa+a),m) - the*(bi-b0)**2/(2*bv**2)

@nb.njit("f8[:](f8,f8,f8,f8,f8,f8[:],f8)")
@cc.export("aux_a", "f8[:](f8,f8,f8,f8,f8,f8[:],f8)")
def aux_a(m,the,kap,a0,av,a,b):
    return kap*h((a-b)/(a+b),m) - the*(a-a0)**2/(2*av**2)

@nb.njit("f8[:](f8,f8,f8,f8,f8,f8,f8,f8[:])")
@cc.export("aux_b", "f8[:](f8,f8,f8,f8,f8,f8,f8,f8[:])")
def aux_b(m,the,kap,rho,b0,bv,a,b):
    return rho*kap*h((b-a)/(b+a),m) - the*(b-b0)**2/(2*bv**2)

@nb.njit("f8[:](f8,f8,f8,f8,f8,f8[:],f8)")
@cc.export("aux_a_diff", "f8[:](f8,f8,f8,f8,f8,f8[:],f8)")
def aux_a_diff(m,the,kap,a0,av,a,b):
    quot = np.arctan(m)*((1+m**2)*(a**2 + b**2) + (2-2*m**2)*a*b)
    cost_ratio = the/av**2
    return 2*m*kap*b / quot - cost_ratio*(a-a0)

@nb.njit("f8[:](f8,f8,f8,f8,f8,f8,f8,f8[:])")
@cc.export("aux_b_diff", "f8[:](f8,f8,f8,f8,f8,f8,f8,f8[:])")
def aux_b_diff(m,the,kap,rho,b0,bv,a,b):
    quot = np.arctan(m)*((1+m**2)*(a**2 + b**2) + (2-2*m**2)*a*b)
    cost_ratio = the/bv**2
    return 2*m*kap*rho*a / quot - cost_ratio*(b-b0)

@nb.njit("c16[:](f8,f8,f8,f8,f8,f8)")
@cc.export("roots_aux_a", "c16[:](f8,f8,f8,f8,f8,f8)")
def roots_aux_a(m, thea, kap, a0, av,b):
    coeffs = np.empty(4,dtype=np.complex128)
    
    coeffs[0] = m**2 + 1
    coeffs[1] = -a0*m**2 - a0 - 2*b*m**2 + 2*b
    coeffs[2] = 2*a0*b*m**2 - 2*a0*b + b**2*m**2 + b**2
    coeffs[3] = -a0*b**2*m**2 - a0*b**2 - 2*av**2*b*kap*m/(thea*np.arctan(m))

    return np.roots(coeffs)

@nb.njit("c16[:](f8,f8,f8,f8,f8,f8,f8)")
@cc.export("roots_aux_b", "c16[:](f8,f8,f8,f8,f8,f8,f8)")
def roots_aux_b(m, theb, kap,rho, b0, bv,a):
    coeffs = np.empty(4,dtype=np.complex128)

    coeffs[0] = m**2 + 1
    coeffs[1] = -2*a*m**2 + 2*a - b0*m**2 - b0
    coeffs[2] = a**2*m**2 + a**2 + 2*a*b0*m**2 - 2*a*b0
    coeffs[3] = -a**2*b0*m**2 - a**2*b0 - 2*a*bv**2*kap*m*rho/(theb*np.arctan(m))

    return np.roots(coeffs)

@nb.njit("c16[:](f8,f8,f8,f8,f8,f8,f8)")
@cc.export("roots_utility_a", "c16[:](f8,f8,f8,f8,f8,f8,f8)")
def roots_utility_a(m, thea, kap, a0, av,a,b):
    coeffs = np.empty(4,dtype=np.complex128)

    coeffs[0] = kap**2*m**2 + kap**2
    coeffs[1] = -2*a*kap**2*m**2 - 2*a*kap**2 + 2*a*kap*m**2 + 2*a*kap - a0*kap**2*m**2 - a0*kap**2 - 2*b*kap*m**2 + 2*b*kap
    coeffs[2] = a**2*kap**2*m**2 + a**2*kap**2 - 2*a**2*kap*m**2 - 2*a**2*kap + a**2*m**2 + a**2 + 2*a*a0*kap**2*m**2 + 2*a*a0*kap**2 - 2*a*a0*kap*m**2 - 2*a*a0*kap + 2*a*b*kap*m**2 - 2*a*b*kap - 2*a*b*m**2 + 2*a*b + 2*a0*b*kap*m**2 - 2*a0*b*kap + b**2*m**2 + b**2
    coeffs[3] = -a**2*a0*kap**2*m**2 - a**2*a0*kap**2 + 2*a**2*a0*kap*m**2 + 2*a**2*a0*kap - a**2*a0*m**2 - a**2*a0 - 2*a*a0*b*kap*m**2 + 2*a*a0*b*kap + 2*a*a0*b*m**2 - 2*a*a0*b - a0*b**2*m**2 - a0*b**2 - 2*av**2*b*kap*m/(thea*np.arctan(m))

    return np.roots(coeffs)

@nb.njit("c16[:](f8,f8,f8,f8,f8,f8,f8,f8)")
@cc.export("roots_utility_b", "c16[:](f8,f8,f8,f8,f8,f8,f8,f8)")
def roots_utility_b(m, theb, kap,rho, b0, bv,a,b):
    coeffs = np.empty(4,dtype=np.complex128)

    coeffs[0] = kap**2*m**2 + kap**2
    coeffs[1] = -2*a*kap*m**2 + 2*a*kap - 2*b*kap**2*m**2 - 2*b*kap**2 + 2*b*kap*m**2 + 2*b*kap - b0*kap**2*m**2 - b0*kap**2
    coeffs[2] = a**2*m**2 + a**2 + 2*a*b*kap*m**2 - 2*a*b*kap - 2*a*b*m**2 + 2*a*b + 2*a*b0*kap*m**2 - 2*a*b0*kap + b**2*kap**2*m**2 + b**2*kap**2 - 2*b**2*kap*m**2 - 2*b**2*kap + b**2*m**2 + b**2 + 2*b*b0*kap**2*m**2 + 2*b*b0*kap**2 - 2*b*b0*kap*m**2 - 2*b*b0*kap
    coeffs[3] = -a**2*b0*m**2 - a**2*b0 - 2*a*b*b0*kap*m**2 + 2*a*b*b0*kap + 2*a*b*b0*m**2 - 2*a*b*b0 - 2*a*bv**2*kap*m*rho/(theb*np.arctan(m)) - b**2*b0*kap**2*m**2 - b**2*b0*kap**2 + 2*b**2*b0*kap*m**2 + 2*b**2*b0*kap - b**2*b0*m**2 - b**2*b0

    return np.roots(coeffs)

@nb.njit("b1[:](f8[:],f8,f8,f8,f8,f8,f8,f8)")
@cc.export("is_group_br_a", "b1[:](f8[:],f8,f8,f8,f8,f8,f8,f8)")
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
@cc.export("is_group_br_b", "b1[:](f8[:],f8,f8,f8,f8,f8,f8,f8,f8)")
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
@cc.export("find_group_br_a", "f8[:](f8,f8,f8,f8,f8,f8,f8)")
def find_group_br_a(m,the,kap,a0,av,b,eps):
    roots = roots_aux_a(m, the, kap,a0,av,b)
    roots = np.real(roots[np.abs(np.imag(roots))< eps]) # only real roots
    if np.any(roots> a0 + av):
        roots = roots[roots < a0 + av] # only admissible
        roots = np.append(roots, a0+av) # append potential corner solution

    tf_array = is_group_br_a(roots,m,the,kap,a0,av,b,eps)
    return roots[tf_array]

@nb.njit("f8[:](f8,f8,f8,f8,f8,f8,f8,f8)")
@cc.export("find_group_br_b", "f8[:](f8,f8,f8,f8,f8,f8,f8,f8)")
def find_group_br_b(m,the,kap,rho,b0,bv,a,eps):
    roots = roots_aux_b(m, the, kap,rho,b0,bv,a)
    roots = np.real(roots[np.abs(np.imag(roots))< eps]) # only real roots
    if np.any(roots> b0 + bv):
        roots = roots[roots < b0 + bv] # only admissible roots
        roots = np.append(roots, b0+bv) # append potential corner solution
    
    tf_array = is_group_br_b(roots,m,the,kap,rho,b0,bv,a,eps)
    return roots[tf_array]

@cc.export("find_group_br_a_vecb", "UniTuple(f8[:],2)(f8,f8,f8,f8,f8,f8[:],f8)")
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

@cc.export("find_group_br_b_veca", "UniTuple(f8[:],2)(f8,f8,f8,f8,f8,f8,f8[:],f8)")
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
cc.compile()