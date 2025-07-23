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
def L_parallel(omega, dx:float, dy:float, dz:float, r1:float, r2:float, width:float = 0) -> float:
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

    c=3e8
    k=omega/c
    dist = sqrt(dx ** 2 + dy ** 2+dz**2)

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
        L = (L_1 + L_2)/2
    else:
        L, err = integrate.quad(dl, 0, 2 * pi, args= (dx, dy, dz, r1, r2))
    return L * mu_0 * \
    np.exp(1j * k * dist) * (1-1j*k*dist) + \
    np.exp(1j * k * dist) * np.pi*k**2*r1**2*r2**2*(dx ** 2 + dy ** 2)/4/dist**3

# Computing for orthogonal-oriented rings

def L_orthogonal(omega, dx:float, dy:float ,dz:float, r1:float, r2:float, width:float = 0) -> float:
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
    c=3e8
    k=omega/c
    dist = sqrt(dx ** 2 + dy ** 2+dz**2)

    def dl(alpha, dx, dy, dz, r_1, r_2):
        dp = sqrt((dx - r_2 * sin(alpha)) ** 2 + dy ** 2)
        kappa_sq = 4 * r_1 * dp / ((dp + r_1) ** 2 + (dz - r_2 * cos(alpha)) ** 2)
        kappa = sqrt(kappa_sq) + 10 ** -7
        A = 1 / (2 * pi) * sqrt(r_1 / (dp + 10 ** -7)) * ((2 / kappa - kappa) * K(kappa_sq) - 2 * E(kappa_sq) / kappa)
        return A * r_2 * dy * cos(alpha) / (dp + 10 ** -7)

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
        L = (L_1 + L_2)/2
    else:
        L, err = integrate.quad(dl, 0, 2 * pi, args=(dx, dy, dz, r1, r2))
    return L * mu_0 * \
    np.exp(1j * k * dist) * (1-1j*k*dist) - \
    np.exp(1j * k * dist) * np.pi*k**2*r1**2*r2**2*dy*dz/4/dist**3