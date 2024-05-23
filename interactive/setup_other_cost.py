from numba.pycc import CC
import numba as nb
import numpy as np

cc = CC('ex_post_funs_other_cost')

@nb.njit("f8[:](f8[:],f8)")
@cc.export("h", "f8[:](f8[:],f8)")
def h(x,m):
    return np.arctan(m*x)/np.arctan(m)

@nb.njit("f8[:](f8,f8,f8,f8,f8,f8,f8,f8[:],f8)")
@cc.export("utility_a", "f8[:](f8,f8,f8,i4,f8,f8,f8,f8[:],f8)")
def utility_a(m,cbar,kap,k,av,a,b,ai,a0):
    akappa = (1-kap)*a + kap*ai
    return h((akappa-b)/(akappa+b),m) - cbar*(ai-a0)**(k+1)/((k+1)*av**(k+1))

@nb.njit("f8[:](f8,f8,f8,i4,f8,f8[:],f8,f8)")
@cc.export("aux_a", "f8[:](f8,f8,f8,i4,f8,f8[:],f8,f8)")
def aux_a(m,cbar,kap,k,av,a,b,a0):
    return kap*h((a-b)/(a+b),m) - cbar*(a-a0)**(k+1)/((k+1)*av**(k+1))

@nb.njit("c16[:](f8,f8,f8,i4,f8,f8,f8)")
@cc.export("roots_aux_a", "c16[:](f8,f8,f8,i4,f8,f8,f8)")
def roots_aux_a(m, cbar, kap, k, av,b,a0):
    coeffs = np.zeros(k+3,dtype=np.complex128)
    
    coeffs[0] = m**2 + 1
    coeffs[1] = 2*a0*m**2 + 2*a0 - 2*b*m**2 + 2*b
    coeffs[2] = a0**2*m**2 + a0**2 - 2*a0*b*m**2 + 2*a0*b + b**2*m**2 + b**2
    coeffs[k+2] -= 2*kap*m*b*av**(k+1)/(np.arctan(m)*cbar)

    return np.roots(coeffs) + a0

@nb.njit("c16[:](f8,f8,f8,i4,f8,f8,f8,f8)")
@cc.export("roots_utility_a", "c16[:](f8,f8,f8,i4,f8,f8,f8,f8)")
def roots_utility_a(m, cbar, kap, k, av,a,b,a0):
    coeffs = np.zeros(k+3,dtype=np.complex128)

    coeffs[0] = kap**2*m**2 + kap**2
    coeffs[1] = -2*a*kap**2*m**2 - 2*a*kap**2 + 2*a*kap*m**2 + 2*a*kap + 2*a0*kap**2*m**2 + 2*a0*kap**2 - 2*b*kap*m**2 + 2*b*kap
    coeffs[2] = a**2*kap**2*m**2 + a**2*kap**2 - 2*a**2*kap*m**2 - 2*a**2*kap + a**2*m**2 + a**2 - 2*a*a0*kap**2*m**2 - 2*a*a0*kap**2 + 2*a*a0*kap*m**2 + 2*a*a0*kap + 2*a*b*kap*m**2 - 2*a*b*kap - 2*a*b*m**2 + 2*a*b + a0**2*kap**2*m**2 + a0**2*kap**2 - 2*a0*b*kap*m**2 + 2*a0*b*kap + b**2*m**2 + b**2
    coeffs[k+2] -= 2*kap*m*b*av**(k+1)/(np.arctan(m)*cbar)

    return np.roots(coeffs) + a0

@nb.njit("b1[:](f8[:],f8,f8,f8,i4,f8,f8,f8,f8)")
@cc.export("is_group_br_a", "b1[:](f8[:],f8,f8,f8,i4,f8,f8,f8,f8)")
def is_group_br_a(roots,m,cbar,kap,k,av,b,a0,eps):
    tf_array = np.zeros(roots.shape[0],dtype=np.bool_)
    for i in range(0,len(roots)):
        roots_u = roots_utility_a(m, cbar, kap, k, av,roots[i],b,a0)
        roots_u = np.real(roots_u[np.abs(np.imag(roots_u)) < eps]) # only real roots
        roots_u = roots_u[roots_u > a0] # only positive roots
        roots_u = roots_u[roots_u < a0+av] # only roots in admissible range
        roots_u = np.append(roots_u, a0+av) # append potential corner solutions
        roots_u = np.append(roots_u, a0)
        index_root = np.argmin(np.abs(roots_u - roots[i]))
        index_max = np.argmax(utility_a(m,cbar,kap,k,av,roots[i],b,roots_u,a0))
        if index_max == index_root:
            tf_array[i] = True
    return tf_array

@nb.njit("f8[:](f8,f8,f8,i4,f8,f8,f8,f8)")
@cc.export("find_group_br_a", "f8[:](f8,f8,f8,i4,f8,f8,f8,f8)")
def find_group_br_a(m,cbar,kap,k,av,b,a0,eps):
    roots = roots_aux_a(m, cbar, kap,k,av,b,a0)
    roots = np.real(roots[np.abs(np.imag(roots))< eps]) # only real roots
    roots = roots[roots < a0+av] # only admissible
    roots = roots[roots > a0]
    roots = np.append(roots, a0+av) # append potential corner solutions
    roots = np.append(roots, a0)

    tf_array = is_group_br_a(roots,m,cbar,kap,k,av,b,a0,eps)
    return roots[tf_array]

@cc.export("find_group_br_a_vecb", "UniTuple(f8[:],2)(f8,f8,f8,i4,f8,f8[:],f8,f8)")
def find_group_br_a_vecb(m,cbar,kap,k,av,b_vec,a0,eps):
    x = np.zeros(10*len(b_vec))
    y = np.zeros(10*len(b_vec))
    count = 0
    for i in range(0,len(b_vec)):
        br_list = find_group_br_a(m,cbar,kap,k,av,b_vec[i],a0,eps)
        for j in range(0,len(br_list)):
            x[count] = br_list[j]
            y[count] = b_vec[i]
            count += 1
    return (x,y)

cc.compile()