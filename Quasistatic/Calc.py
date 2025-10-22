from Metastructure import Metastructure
from Solvers import MetaSolver, MetaSolution
# import default parameters for rings from file 
from Parameters import SRR, Omega

import numpy as np
import matplotlib.pyplot as plt

# Creating example metastructure 
# Size of cube in number of metacell
nz, ny, nx = 10, 10, 10
MetaCube = Metastructure(metamaterial = SRR,	# Metaatom parameters
						 shape = (nz, ny, nx), 	# Size for uniform grid
						 geometry='rectangle',  	# form of the structure
						 Type = 'smooth' 		# type of finite metastructure
)
MetaCube.fill_grid()
MetaCube.visualize()

# Create solver for chosen metastructure
solver = MetaSolver(Omega = Omega,
					metastructure = MetaCube)
print(solver.number, "rings in the solver")
# Create uniform excitation
U_z = np.ones(MetaCube.numbers['z'])
U_y = np.zeros(MetaCube.numbers['y'])
U_x = np.zeros(MetaCube.numbers['x'])
U = np.concatenate([U_z, U_y, U_x])
# Calculate solution on given frequency range
solution = solver.direct_solve(U)
# Save solution to folder './Data'
solution.save_sol('./Data')