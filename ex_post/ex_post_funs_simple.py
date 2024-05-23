import numpy as np

def h(x,m):
    return np.arctan(m*x)/np.arctan(m)

def utility_a(m,the,kap,a0,av,a,b,ai):
    akappa = (1-kap)*a + kap*ai
    return h((akappa-b)/(akappa+b),m) - the*(ai-a0)**2/(2*av**2)

def utility_b(m,the,kap,rho,b0,bv,a,b,bi):
    bkappa = (1-kap)*b + kap*bi
    return rho * h((bkappa-a)/(bkappa+a),m) - the*(bi-b0)**2/(2*bv**2)

def aux_a(m,the,kap,a0,av,a,b):
    return kap*h((a-b)/(a+b),m) - the*(a-a0)**2/(2*av**2)

def aux_b(m,the,kap,rho,b0,bv,a,b):
    return rho*kap*h((b-a)/(b+a),m) - the*(b-b0)**2/(2*bv**2)

def aux_a_diff(m,the,kap,a0,av,a,b):
    quot = np.arctan(m)*((1+m**2)*(a**2 + b**2) + (2-2*m**2)*a*b)
    cost_ratio = the/av**2
    return 2*m*kap*b / quot - cost_ratio*(a-a0)

def aux_b_diff(m,the,kap,rho,b0,bv,a,b):
    quot = np.arctan(m)*((1+m**2)*(a**2 + b**2) + (2-2*m**2)*a*b)
    cost_ratio = the/bv**2
    return 2*m*kap*rho*a / quot - cost_ratio*(b-b0)

def roots_aux_a(m, thea, kap, a0, av,b):
    coeffs = np.empty(4,dtype=np.complex128)

    coeffs[0] = m**2 + 1
    coeffs[1] = -a0*m**2 - a0 - 2*b*m**2 + 2*b
    coeffs[2] = 2*a0*b*m**2 - 2*a0*b + b**2*m**2 + b**2
    coeffs[3] = -a0*b**2*m**2 - a0*b**2 - 2*av**2*b*kap*m/(thea*np.arctan(m))

    return np.roots(coeffs)

def roots_aux_b(m, theb, kap,rho, b0, bv,a):
    coeffs = np.empty(4,dtype=np.complex128)

    coeffs[0] = m**2 + 1
    coeffs[1] = -2*a*m**2 + 2*a - b0*m**2 - b0
    coeffs[2] = a**2*m**2 + a**2 + 2*a*b0*m**2 - 2*a*b0
    coeffs[3] = -a**2*b0*m**2 - a**2*b0 - 2*a*bv**2*kap*m*rho/(theb*np.arctan(m))

    return np.roots(coeffs)

def roots_utility_a(m, thea, kap, a0, av,a,b):
    coeffs = np.empty(4,dtype=np.complex128)

    coeffs[0] = kap**2*m**2 + kap**2
    coeffs[1] = -2*a*kap**2*m**2 - 2*a*kap**2 + 2*a*kap*m**2 + 2*a*kap - a0*kap**2*m**2 - a0*kap**2 - 2*b*kap*m**2 + 2*b*kap
    coeffs[2] = a**2*kap**2*m**2 + a**2*kap**2 - 2*a**2*kap*m**2 - 2*a**2*kap + a**2*m**2 + a**2 + 2*a*a0*kap**2*m**2 + 2*a*a0*kap**2 - 2*a*a0*kap*m**2 - 2*a*a0*kap + 2*a*b*kap*m**2 - 2*a*b*kap - 2*a*b*m**2 + 2*a*b + 2*a0*b*kap*m**2 - 2*a0*b*kap + b**2*m**2 + b**2
    coeffs[3] = -a**2*a0*kap**2*m**2 - a**2*a0*kap**2 + 2*a**2*a0*kap*m**2 + 2*a**2*a0*kap - a**2*a0*m**2 - a**2*a0 - 2*a*a0*b*kap*m**2 + 2*a*a0*b*kap + 2*a*a0*b*m**2 - 2*a*a0*b - a0*b**2*m**2 - a0*b**2 - 2*av**2*b*kap*m/(thea*np.arctan(m))

    return np.roots(coeffs)

def roots_utility_b(m, theb, kap,rho, b0, bv,a,b):
    coeffs = np.empty(4,dtype=np.complex128)

    coeffs[0] = kap**2*m**2 + kap**2
    coeffs[1] = -2*a*kap*m**2 + 2*a*kap - 2*b*kap**2*m**2 - 2*b*kap**2 + 2*b*kap*m**2 + 2*b*kap - b0*kap**2*m**2 - b0*kap**2
    coeffs[2] = a**2*m**2 + a**2 + 2*a*b*kap*m**2 - 2*a*b*kap - 2*a*b*m**2 + 2*a*b + 2*a*b0*kap*m**2 - 2*a*b0*kap + b**2*kap**2*m**2 + b**2*kap**2 - 2*b**2*kap*m**2 - 2*b**2*kap + b**2*m**2 + b**2 + 2*b*b0*kap**2*m**2 + 2*b*b0*kap**2 - 2*b*b0*kap*m**2 - 2*b*b0*kap
    coeffs[3] = -a**2*b0*m**2 - a**2*b0 - 2*a*b*b0*kap*m**2 + 2*a*b*b0*kap + 2*a*b*b0*m**2 - 2*a*b*b0 - 2*a*bv**2*kap*m*rho/(theb*np.arctan(m)) - b**2*b0*kap**2*m**2 - b**2*b0*kap**2 + 2*b**2*b0*kap*m**2 + 2*b**2*b0*kap - b**2*b0*m**2 - b**2*b0

    return np.roots(coeffs)

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

def find_group_br_a(m,the,kap,a0,av,b,eps):
    roots = roots_aux_a(m, the, kap,a0,av,b)
    roots = np.real(roots[np.abs(np.imag(roots))< eps]) # only real roots
    if np.any(roots> a0 + av):
        roots = roots[roots < a0 + av] # only admissible
        roots = np.append(roots, a0+av) # append potential corner solution

    tf_array = is_group_br_a(roots,m,the,kap,a0,av,b,eps)
    return roots[tf_array]

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

def resultant_roots(m,thea,theb,kap,rho,b0,bv,a0,av):
    coeffs = np.empty(5,dtype=np.complex128)

    coeffs[0] = 4*av**4*kap**2*m**8/(thea**2*np.arctan(m)**2) + 12*av**4*kap**2*m**6/(thea**2*np.arctan(m)**2) + 12*av**4*kap**2*m**4/(thea**2*np.arctan(m)**2) + 4*av**4*kap**2*m**2/(thea**2*np.arctan(m)**2) - 8*av**2*bv**2*kap**2*m**8*rho/(thea*theb*np.arctan(m)**2) + 40*av**2*bv**2*kap**2*m**6*rho/(thea*theb*np.arctan(m)**2) + 40*av**2*bv**2*kap**2*m**4*rho/(thea*theb*np.arctan(m)**2) - 8*av**2*bv**2*kap**2*m**2*rho/(thea*theb*np.arctan(m)**2) + 4*bv**4*kap**2*m**8*rho**2/(theb**2*np.arctan(m)**2) + 12*bv**4*kap**2*m**6*rho**2/(theb**2*np.arctan(m)**2) + 12*bv**4*kap**2*m**4*rho**2/(theb**2*np.arctan(m)**2) + 4*bv**4*kap**2*m**2*rho**2/(theb**2*np.arctan(m)**2)
    coeffs[1] = 8*a0*av**2*bv**2*kap**2*m**8*rho/(thea*theb*np.arctan(m)**2) + 8*a0*av**2*bv**2*kap**2*m**6*rho/(thea*theb*np.arctan(m)**2) - 8*a0*av**2*bv**2*kap**2*m**4*rho/(thea*theb*np.arctan(m)**2) - 8*a0*av**2*bv**2*kap**2*m**2*rho/(thea*theb*np.arctan(m)**2) - 8*a0*bv**4*kap**2*m**8*rho**2/(theb**2*np.arctan(m)**2) - 8*a0*bv**4*kap**2*m**6*rho**2/(theb**2*np.arctan(m)**2) + 8*a0*bv**4*kap**2*m**4*rho**2/(theb**2*np.arctan(m)**2) + 8*a0*bv**4*kap**2*m**2*rho**2/(theb**2*np.arctan(m)**2) - 12*av**4*b0*kap**2*m**8/(thea**2*np.arctan(m)**2) - 36*av**4*b0*kap**2*m**6/(thea**2*np.arctan(m)**2) - 36*av**4*b0*kap**2*m**4/(thea**2*np.arctan(m)**2) - 12*av**4*b0*kap**2*m**2/(thea**2*np.arctan(m)**2) + 16*av**2*b0*bv**2*kap**2*m**8*rho/(thea*theb*np.arctan(m)**2) - 80*av**2*b0*bv**2*kap**2*m**6*rho/(thea*theb*np.arctan(m)**2) - 80*av**2*b0*bv**2*kap**2*m**4*rho/(thea*theb*np.arctan(m)**2) + 16*av**2*b0*bv**2*kap**2*m**2*rho/(thea*theb*np.arctan(m)**2) - 4*b0*bv**4*kap**2*m**8*rho**2/(theb**2*np.arctan(m)**2) - 12*b0*bv**4*kap**2*m**6*rho**2/(theb**2*np.arctan(m)**2) - 12*b0*bv**4*kap**2*m**4*rho**2/(theb**2*np.arctan(m)**2) - 4*b0*bv**4*kap**2*m**2*rho**2/(theb**2*np.arctan(m)**2)
    coeffs[2] = 4*a0**2*bv**4*kap**2*m**8*rho**2/(theb**2*np.arctan(m)**2) + 12*a0**2*bv**4*kap**2*m**6*rho**2/(theb**2*np.arctan(m)**2) + 12*a0**2*bv**4*kap**2*m**4*rho**2/(theb**2*np.arctan(m)**2) + 4*a0**2*bv**4*kap**2*m**2*rho**2/(theb**2*np.arctan(m)**2) - 16*a0*av**2*b0*bv**2*kap**2*m**8*rho/(thea*theb*np.arctan(m)**2) - 16*a0*av**2*b0*bv**2*kap**2*m**6*rho/(thea*theb*np.arctan(m)**2) + 16*a0*av**2*b0*bv**2*kap**2*m**4*rho/(thea*theb*np.arctan(m)**2) + 16*a0*av**2*b0*bv**2*kap**2*m**2*rho/(thea*theb*np.arctan(m)**2) + 8*a0*b0*bv**4*kap**2*m**8*rho**2/(theb**2*np.arctan(m)**2) + 8*a0*b0*bv**4*kap**2*m**6*rho**2/(theb**2*np.arctan(m)**2) - 8*a0*b0*bv**4*kap**2*m**4*rho**2/(theb**2*np.arctan(m)**2) - 8*a0*b0*bv**4*kap**2*m**2*rho**2/(theb**2*np.arctan(m)**2) + 12*av**4*b0**2*kap**2*m**8/(thea**2*np.arctan(m)**2) + 36*av**4*b0**2*kap**2*m**6/(thea**2*np.arctan(m)**2) + 36*av**4*b0**2*kap**2*m**4/(thea**2*np.arctan(m)**2) + 12*av**4*b0**2*kap**2*m**2/(thea**2*np.arctan(m)**2) - 8*av**2*b0**2*bv**2*kap**2*m**8*rho/(thea*theb*np.arctan(m)**2) + 40*av**2*b0**2*bv**2*kap**2*m**6*rho/(thea*theb*np.arctan(m)**2) + 40*av**2*b0**2*bv**2*kap**2*m**4*rho/(thea*theb*np.arctan(m)**2) - 8*av**2*b0**2*bv**2*kap**2*m**2*rho/(thea*theb*np.arctan(m)**2) - 32*av**2*bv**4*kap**3*m**7*rho**2/(thea*theb**2*np.arctan(m)**3) + 32*av**2*bv**4*kap**3*m**3*rho**2/(thea*theb**2*np.arctan(m)**3)
    coeffs[3] = -4*a0**2*b0*bv**4*kap**2*m**8*rho**2/(theb**2*np.arctan(m)**2) - 12*a0**2*b0*bv**4*kap**2*m**6*rho**2/(theb**2*np.arctan(m)**2) - 12*a0**2*b0*bv**4*kap**2*m**4*rho**2/(theb**2*np.arctan(m)**2) - 4*a0**2*b0*bv**4*kap**2*m**2*rho**2/(theb**2*np.arctan(m)**2) + 8*a0*av**2*b0**2*bv**2*kap**2*m**8*rho/(thea*theb*np.arctan(m)**2) + 8*a0*av**2*b0**2*bv**2*kap**2*m**6*rho/(thea*theb*np.arctan(m)**2) - 8*a0*av**2*b0**2*bv**2*kap**2*m**4*rho/(thea*theb*np.arctan(m)**2) - 8*a0*av**2*b0**2*bv**2*kap**2*m**2*rho/(thea*theb*np.arctan(m)**2) + 8*a0*av**2*bv**4*kap**3*m**7*rho**2/(thea*theb**2*np.arctan(m)**3) + 16*a0*av**2*bv**4*kap**3*m**5*rho**2/(thea*theb**2*np.arctan(m)**3) + 8*a0*av**2*bv**4*kap**3*m**3*rho**2/(thea*theb**2*np.arctan(m)**3) - 8*a0*bv**6*kap**3*m**7*rho**3/(theb**3*np.arctan(m)**3) - 16*a0*bv**6*kap**3*m**5*rho**3/(theb**3*np.arctan(m)**3) - 8*a0*bv**6*kap**3*m**3*rho**3/(theb**3*np.arctan(m)**3) - 4*av**4*b0**3*kap**2*m**8/(thea**2*np.arctan(m)**2) - 12*av**4*b0**3*kap**2*m**6/(thea**2*np.arctan(m)**2) - 12*av**4*b0**3*kap**2*m**4/(thea**2*np.arctan(m)**2) - 4*av**4*b0**3*kap**2*m**2/(thea**2*np.arctan(m)**2) + 32*av**2*b0*bv**4*kap**3*m**7*rho**2/(thea*theb**2*np.arctan(m)**3) - 32*av**2*b0*bv**4*kap**3*m**3*rho**2/(thea*theb**2*np.arctan(m)**3)
    coeffs[4] = -8*a0*av**2*b0*bv**4*kap**3*m**7*rho**2/(thea*theb**2*np.arctan(m)**3) - 16*a0*av**2*b0*bv**4*kap**3*m**5*rho**2/(thea*theb**2*np.arctan(m)**3) - 8*a0*av**2*b0*bv**4*kap**3*m**3*rho**2/(thea*theb**2*np.arctan(m)**3) - 16*av**2*bv**6*kap**4*m**6*rho**3/(thea*theb**3*np.arctan(m)**4) - 16*av**2*bv**6*kap**4*m**4*rho**3/(thea*theb**3*np.arctan(m)**4)

    return np.roots(coeffs)

def candidates_from_roots(roots,rho,thea,theb,a0,b0,av,bv,eps):
    b_array = np.real(roots[np.abs(np.imag(roots)) < eps]) # only real roots
    b_array = b_array[b_array > b0]
    b_array = b_array[b_array < b0+bv] # only roots in admissible range


    a_array = a0 /2 + np.sqrt(a0**2/ 4 + (theb*av**2/(rho*bv**2*thea))*(b_array**2 - b0*b_array))

    tf_array = np.logical_and(a_array > a0, a_array < a0+av)

    return (a_array[tf_array],b_array[tf_array])

def is_group_br_a_single(a,m,the,kap,a0,av,b,eps):
    roots = roots_utility_a(m, the, kap, a0, av,a,b)
    roots = np.real(roots[np.abs(np.imag(roots)) < eps]) # only real roots
    roots = roots[roots > 0] # only positive roots
    roots = roots[roots < a0+av] # only roots in admissible range
    roots = np.append(roots, a0+av) # append potential corner solution
    index_root = np.argmin(np.abs(roots - a))
    index_max = np.argmax(utility_a(m,the,kap,a0,av,a,b,roots))
    return index_max == index_root

def is_group_br_b_single(b,m,the,kap,rho,b0,bv,a,eps):
    roots = roots_utility_b(m, the, kap,rho, b0, bv,a,b)
    roots = np.real(roots[np.abs(np.imag(roots)) < eps]) # only real roots
    roots = roots[roots > 0]
    roots = roots[roots < b0+bv]
    roots = np.append(roots, b0+bv)
    index_root = np.argmin(np.abs(roots - b))
    index_max = np.argmax(utility_b(m,the,kap,rho,b0,bv,a,b,roots))
    return index_max == index_root

def find_all_equilibria(m,thea,theb,kap,rho,a0,b0,av,bv,eps):
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


def find_equilibria_interior(m,thea,theb,kap,rho,a0,b0,av,bv,eps):
    roots = resultant_roots(m,thea,theb,kap,rho,b0,bv,a0,av)
    cand_tup = candidates_from_roots(roots,rho,thea,theb,a0,b0,av,bv,eps)
    len_int = len(cand_tup[0])
    a_array = cand_tup[0]
    b_array = cand_tup[1]

    tf_array = np.ones(len(a_array),dtype=np.bool_)
    for i in range(0,len_int):
        tf_array[i] = is_group_br_a_single(a_array[i],m,thea,kap,a0,av,b_array[i],eps) and is_group_br_b_single(b_array[i],m,theb,kap,rho,b0,bv,a_array[i],eps)
    return (a_array[tf_array],b_array[tf_array])