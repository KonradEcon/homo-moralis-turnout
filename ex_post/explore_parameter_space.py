from ex_post_funs_jit import *
# modify grid size here
grid_size = 6

# run the brute force search and print the maximum amount of equilibria found
counts, eqs = find_equilibria(grid_size,1e-10)
print("Maximum amount of equilibria: ", np.max(counts))
higheq_idx = np.where(counts>=np.max(counts))

# redefine the grid and print the parameters and equilibria found for parameters yielding the maximum amount
m_arr = np.linspace(0.1,40.0,grid_size)
rho_arr = np.linspace(1.0,5.0,grid_size)
thea_arr = np.linspace(0.1,5.0,grid_size)
theb_arr = np.linspace(0.1,5.0,grid_size)
a0_arr = np.linspace(0.1,1.0,grid_size)
b0_arr = np.linspace(0.1,1.0,grid_size)
av_arr = np.linspace(0.1,2.0,grid_size)
bv_arr = np.linspace(0.1,2.0,grid_size)
kap_arr = np.linspace(0.0,1.0,grid_size)
for i in range(0,len(higheq_idx[0])):
    a_arr = eqs[0,:,higheq_idx[0][i],higheq_idx[1][i],higheq_idx[2][i],higheq_idx[3][i],higheq_idx[4][i],higheq_idx[5][i],higheq_idx[6][i],higheq_idx[7][i],higheq_idx[8][i]]
    
    b_arr = eqs[1,:,higheq_idx[0][i],higheq_idx[1][i],higheq_idx[2][i],higheq_idx[3][i],higheq_idx[4][i],higheq_idx[5][i],higheq_idx[6][i],higheq_idx[7][i],higheq_idx[8][i]]

    print("m      ", m_arr[higheq_idx[0][i]])
    print("thetaA ", thea_arr[higheq_idx[1][i]])
    print("thetaB ", theb_arr[higheq_idx[2][i]])
    print("kappa  ", kap_arr[higheq_idx[3][i]])
    print("rho    ", rho_arr[higheq_idx[4][i]])
    print("a0     ", a0_arr[higheq_idx[5][i]])
    print("b0     ", b0_arr[higheq_idx[6][i]])
    print("av     ", av_arr[higheq_idx[7][i]])
    print("bv     ", bv_arr[higheq_idx[8][i]])

    for j in range(0,len(a_arr)):
        if a_arr[j] > 0:
            print(a_arr[j],b_arr[j])

# save the counts and the equilibria
np.savez_compressed("eqs_"+ str(grid_size),counts=counts,eqs=eqs)