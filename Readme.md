# Code repository
The code associated to our paper ''Homo moralis goes to the voting booth: a new theory of voter turnout'' 

## Setup for reproducing the graphics
The code is confirmed to run on Python 3.11. Consider installing that version of python and creating a virtual environment. On Linux or MacOS, creating the virtual environment can be done using
```
python3.11 -m venv 311env
```
This creates a virtual environment with the name 311env in the current folder.

Activate it using
```
source 311env/bin/activate
```
Check the python documentation for Windows instructions.

 At the time of writing, `numba` was unavailable for Python 3.12. Make sure to have all packages found in `requirements.txt` installed. In case of issues, install the versions specified in `requirements.txt`. You can install the required packages within the virtual environment using
```
pip install -r requirements.txt
```
after activating the virtual environment and changing directory into the cloned repository.


## How to use
The code used to generate graphics for the ex-ante and ex-post cases of our paper can be found in the `ex_ante` and `ex_post` folders, respectively. More precisely, the files ending in `1d` will generate all the line plots, and the files ending in `2d` will generate the 2D graphics that show the number of equilibria or turnouts or utilities while varying two parameters.

`symbolic_calculations.ipynb` is a Jupyter notebook that performs the symbolic calculations, using `SymPy`, that are needed in order to obtain the polynomial coefficients used for computing equilibrium and deviation candidates in the ex-ante and ex-post code.

`symbolic_existence_kappa.ipynb` is a Jupyter notebook in which the symbolic calculations for existence and uniqueness proofs are executed.

`explore_parameter_space.py` in the `ex_post` folder determines the maximum number of equilibria doing a bruteforce search in the parameter space in a 6x6x...x6 (9 repetitions) grid. Then, it prints the maximum amount of equilibria found, together with all the parameter tuples that yield this amount of equilibria and the associated equilibria. 
You may increase the grid size in the script, but check to have enough RAM, disk space and time before trying with a size greater than $N=6$. For the 6x6x...x6 grid, plan on at least 2.6 GB of RAM and 100 MB of disk space, these are proportional to $N^9$, where $N$ is the grid size. Also, be aware that numba's just-in-time compilation takes a couple of minutes.

`explore_parameter_space_same_cost_structure.py` does the same for what we call the groups having "same cost structure" in the paper. This reduces the size of the parameter space to search, so one can increase $N$ a bit more here.

All other files in the `ex_ante` and `ex_post` folders contain helper functions for the scripts described above.

The source code of our interactive plot can be found in the `interactive` folder. It contains all the relevant files for setting up a Heroku app, except that one needs to compile the functions using `python setup.py` first. Note that in order to do this, you must have `python dev` installed for Python 3.11. On fedora, you can install this using
```
sudo dnf install python3.11-devel
```
This should work similarly for other Linux distributions.

Running `setup.py` creates a python module. One can then run the interactive plot locally by running, from that folder,
```
bokeh serve ex_post_vis_bokeh.py
```
and similarly for the other interactive plots.

The `Procfile`, `runtime.txt` and the additional `requirements.txt` are only for setting up a Heroku server.
