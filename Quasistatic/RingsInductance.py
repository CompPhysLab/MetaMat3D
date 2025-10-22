# Calculating geometry matrix M
import json

import numpy as np
from numpy import sqrt, cos, sin, pi
from scipy import integrate
from scipy import special
from tqdm import tqdm
from Metamaterial import Ring

K = special.ellipk  # Сomplete elliptic integral of the first kind
E = special.ellipe  # Сomplete elliptic integral of the second kind

mu_0 = 4 * pi * 10 ** -7
'''
Here we have a function that calculates mutual inductance between two rings
The special way to optimize the calculation is represent strip as two
thin rings with different radiuses, because of experimental data
proves that current distribution concentrates near the edge of the strip
'''
# Computing for parallel-oriented rings
def L_parallel(dx:float, dy:float, dz:float, r_1:float, r_2:float, w_1:float = 0, w_2:float = 0) -> float:
	"""Calculates mutual inductance between two parallel-oriented rings

	Parameters
	----------
	dx : float
		distance between the centers of the rings along the x-axis
	dy : float
		distance between the centers of the rings along the y-axis
	dz : float
		distance between the centers of the rings along the z-axis
	r1 : float
		radius of the first ring
	r2 : float
		radius of the second ring
	width : float, optional
		width of the strip, by default 0

	Returns
	-------
	float
		mutual inductance between two parallel-oriented rings
	"""    
	# Define function to integrate over first defined parameter

	def dl(alpha, dx, dy, dz, r_1, r_2):
		db = sqrt(dx ** 2 + dy ** 2)
		dp = sqrt(r_2 ** 2 + db ** 2 + 2 * r_2 * db * cos(alpha))
		kappa_sq = 4 * r_1 * dp / ((dp + r_1) ** 2 + dz ** 2)
		kappa = sqrt(kappa_sq)
		A = 1/(2*pi)*sqrt(r_1/dp) * ((2/kappa - kappa) * K(kappa_sq) - 2 * E(kappa_sq)/kappa)
		return A * r_2 * (r_2 + db * cos(alpha))/dp

	#Considering stripe width
	if r_1 != r_2 or (w_1 and w_2 and w_1 != w_2):
		R1, R2 = r_1 + w_1/2, r_2 + w_2/2
		r1, r2 = r_1 - w_1/2, r_2 - w_2/2

		L_mm = integrate.quad(dl, 0, 2 * pi, args=(dx, dy, dz, r1, r2))[0]
		L_mp = integrate.quad(dl, 0, 2 * pi, args=(dx, dy, dz, r1, R2))[0]
		L_pm = integrate.quad(dl, 0, 2 * pi, args=(dx, dy, dz, R1, r2))[0]
		L_pp = integrate.quad(dl, 0, 2 * pi, args=(dx, dy, dz, R1, R2))[0]
		L = (L_pp + L_pm + L_mp + L_mm)/4
	
	elif r_1 == r_2 and w_1 and w_2 and w_1 == w_2:
		R, r = r_1 + w_1/2, r_2 - w_1/2

		L_m  = integrate.quad(dl, 0, 2 * pi, args=(dx, dy, dz, r, r))[0]
		L_mp = integrate.quad(dl, 0, 2 * pi, args=(dx, dy, dz, r, R))[0]
		L_p  = integrate.quad(dl, 0, 2 * pi, args=(dx, dy, dz, R, R))[0]
		L = (L_p + 2*L_mp + L_m)/4
	elif w_1 == 0:
		L_1 = integrate.quad(dl, 0, 2 * pi, args=(dx, dy, dz, r_1, r_2 + w_2/2))[0]
		L_2 = integrate.quad(dl, 0, 2 * pi, args=(dx, dy, dz, r_1, r_2 - w_2/2))[0]
		return (L_1 + L_2)/2
	elif w_2 == 0:
		L_1 = integrate.quad(dl, 0, 2 * pi, args=(dx, dy, dz, r_1 + w_1/2, r_2))[0]
		L_2 = integrate.quad(dl, 0, 2 * pi, args=(dx, dy, dz, r_1 - w_1/2, r_2))[0]
		return (L_1 + L_2)/2
	else:
		L, err = integrate.quad(dl, 0, 2 * pi, args= (dx, dy, dz, r_1, r_2))
	return L

# Computing for orthogonal-oriented rings

def L_orthogonal(dx:float, dy:float ,dz:float, r_1:float, r_2:float, w_1:float = 0, w_2:float = 0) -> float:
	"""Calculates mutual inductance between two orthogonal-oriented rings

	Parameters
	----------
	dx : float
		distance between the centers of the rings along the x-axis
	dy : float
		distance between the centers of the rings along the y-axis
	dz : float
		distance between the centers of the rings along the z-axis
	r1 : float
		radius of the first ring
	r2 : float
		radius of the second ring
	width : float, optional
		width of the strip, by default 0

	Returns
	-------
	float
		mutual inductance between two orthogonal-oriented rings
	"""    
	def dl(alpha, dx, dy, dz, r_1, r_2):
		dp = sqrt((dx - r_2 * sin(alpha)) ** 2 + dy ** 2)
		kappa_sq = 4 * r_1 * dp / ((dp + r_1) ** 2 + (dz - r_2 * cos(alpha)) ** 2)
		kappa = sqrt(kappa_sq)

		A = 1 / (2 * pi) * sqrt(r_1 / (dp)) * ((2 / kappa - kappa) * K(kappa_sq) - 2 * E(kappa_sq) / kappa)
		return A * r_2 * dy * cos(alpha) / (dp)

	# Considering stripe width
	if r_1 != r_2 or (w_1 and w_2 and w_1 != w_2):
		R1, R2 = r_1 + w_1/2, r_2 + w_2/2
		r1, r2 = r_1 - w_1/2, r_2 - w_2/2

		L_mm = integrate.quad(dl, 0, 2 * pi, args=(dx, dy, dz, r1, r2))[0]
		L_mp = integrate.quad(dl, 0, 2 * pi, args=(dx, dy, dz, r1, R2))[0]
		L_pm = integrate.quad(dl, 0, 2 * pi, args=(dx, dy, dz, R1, r2))[0]
		L_pp = integrate.quad(dl, 0, 2 * pi, args=(dx, dy, dz, R1, R2))[0]
		L = (L_pp + L_pm + L_mp + L_mm)/4
	
	elif r_1 == r_2 and w_1 and w_2 and w_1 == w_2:
		R, r = r_1 + w_1/2, r_2 - w_1/2

		L_m  = integrate.quad(dl, 0, 2 * pi, args=(dx, dy, dz, r, r))[0]
		L_mp = integrate.quad(dl, 0, 2 * pi, args=(dx, dy, dz, r, R))[0]
		L_p  = integrate.quad(dl, 0, 2 * pi, args=(dx, dy, dz, R, R))[0]
		L = (L_p + 2*L_mp + L_m)/4
	elif w_1 == 0:
		L_1 = integrate.quad(dl, 0, 2 * pi, args=(dx, dy, dz, r_1, r_2 + w_2/2))[0]
		L_2 = integrate.quad(dl, 0, 2 * pi, args=(dx, dy, dz, r_1, r_2 - w_2/2))[0]
		return (L_1 + L_2)/2
	elif w_2 == 0:
		L_1 = integrate.quad(dl, 0, 2 * pi, args=(dx, dy, dz, r_1 + w_1/2, r_2))[0]
		L_2 = integrate.quad(dl, 0, 2 * pi, args=(dx, dy, dz, r_1 - w_1/2, r_2))[0]
		return (L_1 + L_2)/2
	else:
		L, err = integrate.quad(dl, 0, 2 * pi, args= (dx, dy, dz, r_1, r_2))
	return L

# Computing for any pair and any rings
def Mnm(First_ring:Ring, Second_ring:Ring, Data:dict = {}) -> float:
	"""Calculates mutual inductance between two rings with both parallel 
	and orthogonal orientation. Data is a dictionary with all parameters
	and values to avoid calculating integrals with same params each time.

	Parameters
	----------
	First_ring : Ring
		first ring
	Second_ring : Ring
		second ring
	Data : dict, optional
		Dictionary with mutual inductance, by default {}

	Returns
	-------
	float
		mutual inductance between two rings
	"""    
	dx = Second_ring.x - First_ring.x
	dy = Second_ring.y - First_ring.y
	dz = Second_ring.z - First_ring.z
	if dx == 0 and dy == 0 and dz == 0:
		return 0
	id_x = tuple([dx, dy, dz, First_ring.pos, Second_ring.pos,
				 First_ring.radius, Second_ring.radius,
				 First_ring.width, Second_ring.width])
	if id_x in Data:
		return Data[id_x] * mu_0
	r_1, r_2 = First_ring.radius, Second_ring.radius
	w_1, w_2 = First_ring.width, Second_ring.width
	# Consider all types of parallel orientation and symmetry for x-z axes

	if First_ring.pos == Second_ring.pos:
		if First_ring.pos == "z":                           # Z-oriented rings
			l = L_parallel(dx, dy, dz, r_1, r_2, w_1, w_2)
		elif First_ring.pos == "y":                         # Y-oriented rings
			l = L_parallel(dx, -dz, dy, r_1, r_2, w_1, w_2)
		else:                                               # X-oriented rings
			l = L_parallel(-dz, dy, dx, r_1, r_2, w_1, w_2)

	# Consider all types of orthogonal orientation
	else:
		if First_ring.pos == "z":
			if Second_ring.pos == "y":                      # Z-Y oriented pair
				l = L_orthogonal(dx, dy, dz, r_1, r_2, w_1, w_2)
			else:                                           # Z-X oriented pair
				l = L_orthogonal((dy), (dx), dz, r_1, r_2, w_1, w_2)
		elif First_ring.pos == "y":
			if Second_ring.pos == "z":                      # Y-Z oriented pair
				l = L_orthogonal(dx, (dz), (dy), r_1, r_2, w_1, w_2)
			else:                                           # Y-X oriented pair
				l = L_orthogonal(-dz, (dx), (dy), r_1, r_2,  w_1, w_2)
		elif First_ring.pos == "x":
			if Second_ring.pos == "z":                      # X-Z oriented pair
				l = L_orthogonal(dy, (dz), (dx), r_1, r_2, w_1, w_2)
			else:                                           # X-Y oriented pair
				l = L_orthogonal((dz), dy, (dx), r_1, r_2, w_1, w_2)
	Data[id_x] = l 
	return l * mu_0

# Function for periodic structures
def MutualInductance(dr, radius, width, dir_1, dir_2):
	dx, dy, dz = dr
	if dx == 0 and dy == 0 and dz == 0:
		return 0
	# Consider all types of parallel orientation and symmetry for x-z axes

	if dir_1 == dir_2:
		if dir_1 == "z":                           # Z-oriented rings
			l = L_parallel(dx, dy, dz, radius, radius, width, width)
		elif dir_1 == "y":                         # Y-oriented rings
			l = L_parallel(dx, -dz, dy, radius, radius, width, width)
		else:                                               # X-oriented rings
			l = L_parallel(-dz, dy, dx, radius, radius, width, width)

	# Consider all types of orthogonal orientation
	else:
		if dir_1 == "z":
			if dir_2 == "y":                      # Z-Y oriented pair
				l = L_orthogonal(dx, dy, dz, radius, radius, width, width)
			else:                                           # Z-X oriented pair
				l = L_orthogonal((dy), (dx), dz, radius, radius, width, width)
		elif dir_1 == "y":
			if dir_2 == "z":                      # Y-Z oriented pair
				l = L_orthogonal(dx, (dz), (dy), radius, radius, width, width)
			else:                                           # Y-X oriented pair
				l = L_orthogonal(-dz, (dx), (dy), radius, radius, width, width)
		elif dir_1 == "x":
			if dir_2 == "z":                      # X-Z oriented pair
				l = L_orthogonal(dy, (dz), (dx), radius, radius, width, width)
			else:                                           # X-Y oriented pair
				l = L_orthogonal((dz), dy, (dx), radius, radius, width, width)

	return l * mu_0