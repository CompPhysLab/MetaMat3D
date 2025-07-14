# Calculating geometry matrix M
import json

import numpy as np
from numpy import sqrt, cos, sin, pi
from scipy import integrate
from scipy import special
from tqdm import tqdm
from typing import List
from Ring_Class import Ring

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
def L_parallel(dx:float, dy:float, dz:float, r1:float, r2:float, width:float = 0) -> float:
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

    if r1 == r2 and width:
        R = r1 + width / 2
        r = r1 - width / 2
        L_1, err_1 = integrate.quad(dl, 0, 2 * pi, args=(dx, dy, dz, r, r))
        L_2, err_1 = integrate.quad(dl, 0, 2 * pi, args=(dx, dy, dz, r, R))
        L_3, err_3 = integrate.quad(dl, 0, 2 * pi, args=(dx, dy, dz, R, R))
        L = (L_1 + 2*L_2 + L_3)/4
    elif width:
        id_r1 = r1 == min(r1, r2)
        id_r2 = r2 == min(r1, r2)
        L_1, err_1 = integrate.quad(dl, 0, 2 * pi, args=(dx, dy, dz, r1 + id_r1 * width/2, r2 + id_r2*width/2))
        L_2, err_2 = integrate.quad(dl, 0, 2 * pi, args=(dx, dy, dz, r1 - id_r1 * width/2, r2 - id_r2*width/2))
        return (L_1 + L_2)/2
    else:
        L, err = integrate.quad(dl, 0, 2 * pi, args= (dx, dy, dz, r1, r2))
    return L

# Computing for orthogonal-oriented rings

def L_orthogonal(dx:float, dy:float ,dz:float, r1:float, r2:float, width:float = 0) -> float:
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

    if r1 == r2 and width:
        R = r1 + width / 2
        r = r1 - width / 2
        L_1, err_1 = integrate.quad(dl, 0, 2 * pi, args=(dx, dy, dz, r, r))
        L_2, err_2 = integrate.quad(dl, 0, 2 * pi, args=(dx, dy, dz, r, R))
        L_3, err_3 = integrate.quad(dl, 0, 2 * pi, args=(dx, dy, dz, R, R))
        L = (L_1 + 2 * L_2 + L_3) / 4
    elif width:
        id_r1 = r1 == min(r1, r2)
        id_r2 = r2 == min(r1, r2)
        L_1, err_1 = integrate.quad(dl, 0, 2 * pi, args=(dx, dy, dz, r1 + id_r1 * width/2, r2 + id_r2*width/2))
        L_2, err_2 = integrate.quad(dl, 0, 2 * pi, args=(dx, dy, dz, r1 - id_r1 * width/2, r2 - id_r2*width/2))
        return (L_1 + L_2)/2
    else:
        L, err = integrate.quad(dl, 0, 2 * pi, args=(dx, dy, dz, r1, r2))
    return L

# Computing for any pair

def Mnm(omega, First_ring:Ring, Second_ring:Ring, Data:dict = {}) -> float:
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
    r1 = First_ring.r
    r2 = Second_ring.r
    w = First_ring.w * 0.7

    # To avoid calculating integrals with same params each time
    # there is a dictionary with all parameters and values

    id_1 = f"{dx} {dy} {dz} {r1} {r2} {First_ring.pos}{Second_ring.pos}"
    id_2 = f"{-dx} {-dy} {-dz} {r2} {r1} {Second_ring.pos}{First_ring.pos}"

    if id_1 in Data:
        return Data[id_1]
    elif id_2 in Data:
        return Data[id_2]
    
    elif dx == 0 and dy == 0 and dz == 0:
        Data[id_1] = 0
        return 0

    # Consider all types of parallel orientation and symmetry for x-z axes

    if First_ring.pos == Second_ring.pos:
        if First_ring.pos == "z":                           # Z-oriented rings
            l = L_parallel(dx, dy, dz, r1, r2, w)
        elif First_ring.pos == "y":                         # Y-oriented rings
            l = L_parallel(dx, -dz, dy, r1, r2, w)
        else:                                               # X-oriented rings
            l = L_parallel(-dz, dy, dx, r1, r2, w)

    # Consider all types of orthogonal orientation

    else:
        if First_ring.pos == "z":
            if Second_ring.pos == "y":                      # Z-Y oriented pair
                l = L_orthogonal(dx, dy, dz, r1, r2, w)
            else:                                           # Z-X oriented pair
                l = L_orthogonal((dy), (dx), dz, r1, r2, w)
        elif First_ring.pos == "y":
            if Second_ring.pos == "z":                      # Y-Z oriented pair
                l = L_orthogonal(dx, (dz), (dy), r1, r2, w)
            else:                                           # Y-X oriented pair
                l = L_orthogonal(-dz, (dx), (dy), r1, r2,  w)
        elif First_ring.pos == "x":
            if Second_ring.pos == "z":                      # X-Z oriented pair
                l = L_orthogonal(dy, (dz), (dx), r1, r2, w)
            else:                                           # X-Y oriented pair
                l = L_orthogonal((dz), dy, (dx), r1, r2, w)

    Data[id_1], Data[id_2] = [l * mu_0] * 2
    return l * mu_0

# Calculating mutual inductance for each pair

def Matrix(rings:List[Ring], Data:dict = {}) -> np.ndarray:
    """Calculates mutual inductance matrix for all rings
    and represents it as a square matrix

    Parameters
    ----------
    rings : List[Ring]
        list of rings
    Data : dict, optional
        Dictionary with mutual inductance, by default {}

    Returns
    -------
    np.ndarray
        mutual inductance matrix with zeros on the diagonal
    """    
    M = np.zeros((len(rings), len(rings)))
    for n in tqdm(range(len(rings))):
        for m in range(n, len(rings)):
            R1 = rings[n]
            R2 = rings[m]
            M[n][m] = Mnm(R1, R2, Data)
            M[m][n] = M[n][m]
    return M