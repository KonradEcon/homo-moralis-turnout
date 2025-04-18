import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as plc
from pathlib import Path

cache_dir = Path("/home/konrad/.homo_moralis_cache_ex_ante")

vars = [
    ["$m$", r"$\kappa$", "m", "kap"],
    [r"$b_0$", r"$\kappa$", "b0", "kap"],
    [r"$\rho$", r"$\kappa$", "rho", "kap"]
]

with open(cache_dir / "all_params.txt", "r") as f:
    str1 = f.read()

cmap_len = 4
cmap = plt.get_cmap('viridis',cmap_len)
norm = plc.BoundaryNorm([-0.5 + i for i in range(0,cmap_len+1)],cmap_len)

fig, axes = plt.subplots(3,2, figsize=(13,15))
i = 0
for var in vars:
    print(var)
    x_arr, y_arr, counts,counts_b_winning = np.load(cache_dir / f'eq_{var[2]}_{var[3]}.npz').values()

    
    
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