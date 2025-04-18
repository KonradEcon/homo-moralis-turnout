import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as plc
from pathlib import Path

cache_dir = Path("/home/konrad/.homo_moralis_cache")

# N = 4000 # set to 100 or less for testing, 4000 for high resolution plots

# m = 40.0
# kap = 0.14
# rho = 5.0
# thea = 2.9
# theb = thea
# av = 0.9
# a0 = 0.85
# b0 = 0.6
# bv = b0*av/a0
# eps = 1e-10
# param_arr = np.array([m,thea,theb,kap,rho,a0,b0,av,bv])



vars1 = [
    ["$m$", r"$\kappa$", "m", "kap"],
    [r"$b_0$", r"$\kappa$", "b0", "kap"],
    [r"$\rho$", r"$\kappa$", "rho", "kap"]
]
vars2 = [
    [r"$\theta_B$", r"$\theta_A$", "theb", "thea"],
    [r"$b_0$", r"$a_0$", "b0", "a0"],
    [r"$b_0$", r"$b_v$", "b0", "bv"],
]

with open(cache_dir / "all_params.txt", "r") as f:
    str1 = f.read()

cmap_len = 6
cmap = plt.get_cmap('viridis',cmap_len)
norm = plc.BoundaryNorm([-0.5 + i for i in range(0,cmap_len+1)],cmap_len)

for vars in [vars1, vars2]:
    fig, axes = plt.subplots(3,2, figsize=(13,15))
    i = 0
    for var in vars:
        print(var)
        x_arr, y_arr, counts,counts_a_winning,counts_b_winning,turnouts_a,turnouts_b,utilities_a,utilities_b = np.load(cache_dir / f'output_{var[2]}_{var[3]}.npz').values()

        
        
        pcm =axes[i,0].pcolormesh(y_arr, x_arr, counts, cmap=cmap,norm=norm, shading='auto')

        axes[i,1].pcolormesh(y_arr, x_arr, counts_b_winning, cmap=cmap,norm=norm, shading='auto')
        
        axes[i,0].set_xlabel(var[1])
        axes[i,0].set_ylabel(var[0])
        axes[i,1].set_xlabel(var[1])
        axes[i,1].set_ylabel(var[0])
        fig.colorbar(pcm,ax=axes[i,:],location="right",pad=0.01,ticks=range(cmap_len))

        i += 1

    namestring = "2d_plots/counts_" + "_".join([var[2] for var in vars]) + "__" + "_".join([var[3] for var in vars])
    fig.text(0.45,0.072,str1,ha='center')
    plt.savefig(namestring + "_low_res.png", dpi=300,bbox_inches='tight')
    plt.savefig(namestring + "_high_res.png", dpi=1200,bbox_inches='tight')