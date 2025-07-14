# MetaMat3D
This repository contains program written on Python 3 to calculate the properties of finite metamaterial samples, which takes into account the discrete structure of metamaterials based on capacitively loaded ring resonators.

## Root directory

Firstly, it's highly recommended to create virtual environment and then install all packages from `requirements.txt`. Usually it could be done in shell by
```sh
python3 -m venv METAvenv
python3 -m pip install -r requirements.txt
```
Some iterative solvers and theie usage (in `scipy` for example) are hugely depended from version to version, so configuring might be necessary.

## `Code` Folder 

There are functions and parameter files used in modeling. In Parameters files structure of resulted dictionaty are clearly described.

### `Ring_Class.py`
First steps in OOP, but really trivial usage. Definitely, just an array of parameters for ring (coordinates x, y, z, size and self-inductivity, capacitance and resistance) but provides a beat more readable code. Also function to calculate self-impedance Z and effective inductance M equal Z/jw.

### `Geometry.py`
Contains packing structures to create lists of rings in correct order for fast algorithms. There is necessary to divide different orientation to blocks because of each mutual impedance block are tree time Toeplitz and could be multiply by $N\log N$ instead of $N^2$ with also provided here correct order for rings.

### `Ring_visualize.ipynb`
Jupiter Notebook to create visualization of structure for repots and checking if packing was correct. Because of using `plotly` it's possible to interact smoothly with 3D objects.

### `Impedance_matrix.py`
Calculating mutual part of impedance, that based only at geomerty
of the problem. 

#### `Mnm`

Function to calculate mutual inductance with to parallel or orthogonal mutual orientation.

#### `Matrix`

Apply `Mnm.py` to each ring in list, used only for straight solution. Usually, rectangle structures might be represented  as Toeplitz vector with length $N$ instead of $N^2$ memory cost.

### `Straight_Method.py`

#### `solvesystem`
 Solving matrix equation with currents, voltage and Impedance using Gauss method. 
Do not recommend to use for number of ring greater than 3000 because of time and memory costs, but usually used to compare with evolving iterative methods.

#### `effective_mu`
Calculates effective magnetic permeabiliry for used parameters of infinite metamaterial, 
described in "PHYSICAL REVIEW B 93, 235156 (2016)
Slow convergence to effective medium in finite discrete metamaterials
M. Lapine, R. C. McPhedran, and C. G. Poulton"

#### `spherical_chi`, `needle_chi`, `disk_chi`
Calculates $\chi_{zz}$-component of polarization tensor for different ellipse cases (equal, 1-zero and 2-zero semi-axis).


### `Fast_Method.py`
Fast Method (`solvesystem`) for solving matrix equation which uses symmetry of Impedance Matrix, so
in certain numeration it turns to 3D Toeplitz matrix and solved numerically on Krylov subspaces (`scipy.gmres`).

### `Calc.py`
Used to simplify saving and opening data, where you need to import `save` and `open_model` function on the described way.

Also used to exact calculation in `py`-scripts avoiding ipynb memory problems, so you could run `Calc.py` directly from shell in different python processes, that provides vulgar paralleling (Necessary for supernova calculation and so on).

### `Time_estimation.ipynb`, `FormComparing.ipynb`, `Grad.ipynb`

Generally, all `ipynb` files are sandboxes for calculation and data analyzing.

- `Form_Comparing.ipynb` used to compare polarization spectra dependency with increasing size and compare with continues media theory. There is a lot of useful function for data plotting (such as `plot_slice` for current distribution) and support all new methods for calculations.
- `Time_estimation.ipynb` used to analyze numerical asymptotics for algorithms. Also compared number of iterations depended on condition of matrix that is similar to quality factor.
- `Grad.ipynb` used to analyze gradient in metamaterials, but until `Fast_method.py` is finished, so **needed to be completly rewritten**. To rewrite is useful to inspire by `Form_comparing.ipynb`

## `Verifying MRI` Folder
Modeling based on MRI-Lenz from "Exact modelling method for discrete finite
metamaterial lens
M. Lapine, L. Jelinek, R. Marque´s,
 M.J. Freire" experiment to compare with.

## `Plots` Folder
Contains al plots and animation used to analyze results, such as current distributions and polarization along $z$-axes
## `Data` and `DataLowTol` Folders
Folder will be created after calculations to save data with currents (large files, be aware because possible weighs more than 5 Gb).

Following structure for calculation:
#### Folder `PackingType_IsGrad_shape_orientations_solverType`
Contains:
- `numpy.ndarray` of each current on each frequence in file `Currents.npz`
- 'numpy.ndarray' with precalculated polarization for each directions that allows to not import whole currents file for data-plotting and analyze.
- 'json' file, which structure is same as dictionary in parameters files.
