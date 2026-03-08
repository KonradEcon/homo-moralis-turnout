# Code repository
The code associated to our paper ''Homo moralis goes to the voting booth: a new theory of voter turnout'' 

## Setup
The code is confirmed to run on Python 3.14. Consider installing that version of python and creating a virtual environment. On Linux or MacOS, creating the virtual environment can be done using
```
python3.14 -m venv .venv
```
This creates a virtual environment with the name .venv in the current folder.

Activate it using
```
source .venv/bin/activate
```
Check the python documentation for Windows instructions.

Make sure to have all packages found in `requirements.txt` installed. You can install the required packages within the virtual environment using
```
pip install -r requirements.txt
```
after activating the virtual environment and changing directory into the cloned repository.

In order to use the jupyter notebooks (`.ipynb`), with the VSCode Jupyter extension, it is enough to have `ipykernel` installed in the environment. Otherwise, consider installing a proper jupyter package (see `jupyter.org`).


## How to use
The code to generate graphics for the ex-ante (nonpartisan) and ex-post (partisan) cases of our paper can be found in the `ex_ante` and `ex_post` folders, respectively, and run using `python [filename].py` from within the directory the file is located in. 

The files ending in `1d.py` will generate all the line plots. The 2D graphics (that show the number of equilibria or turnouts or utilities while varying two parameters) are created in two steps: as the computation takes a significant amount of time, the underlying numpy arrays are computed by the scripts ending in `2d_calculations.py` and saved in a cache directory (make sure to modify). Then, the numpy arrays are plotted by the scripts ending in `2d_plots.py`. This allows adjusting the plots without having to recompute the underlying data.

`symbolic_calculations.ipynb` is a Jupyter notebook that performs the symbolic calculations, using `SymPy`, that are needed in order to obtain the polynomial coefficients used for computing equilibrium, consistent strategy and deviation candidates in the ex-ante (nonpartisan) and ex-post (partisan) code.

`explore_parameter_space.py` in the `ex_post` folder determines the maximum number of equilibria doing a bruteforce search in the parameter space in a 6x6x...x6 (9 repetitions) grid. Then, it prints the maximum amount of equilibria found, together with all the parameter tuples that yield this amount of equilibria and the associated equilibria. 
You may increase the grid size in the script, but check to have enough RAM, disk space and time before trying with a size greater than $N=6$. For the 6x6x...x6 grid, plan on at least 2.6 GB of RAM and 100 MB of disk space, these are proportional to $N^9$, where $N$ is the grid size. Also, be aware that numba's just-in-time compilation takes a couple of minutes.

`explore_parameter_space_same_cost_structure.py` does the same for what we call the groups having "same cost structure" in the paper. This reduces the size of the parameter space to search, so one can increase $N$ a bit more here.

All other files in the `ex_ante` and `ex_post` folders contain helper functions for the scripts described above.
