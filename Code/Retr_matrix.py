import json

import numpy as np
from numpy import sqrt, cos, sin, pi
from scipy import integrate
from scipy import special
from scipy.optimize import root_scalar
from tqdm import tqdm
from typing import List
from Ring_Class import Ring
from Asymp_matrix import L_parallel as L_par_asymp
from Asymp_matrix import L_orthogonal as L_ort_asymp
from Parameters import *

K = special.ellipk  # Сomplete elliptic integral of the first kind
E = special.ellipe  # Сomplete elliptic integral of the second kind

c=3e8
mu_0 = 4 * np.pi * 1e-7
accuracy = Params['Accuracy']

def L_parallel(omega, dx:float, dy:float, dz:float, r1:float, r2:float, width:float = 0):
    k = omega/c
    def A_real(phi_2, dx, dy, dz, r_1, r_2, k):
    #r_1 = R_m, r_2 = R_n

        def dA(phi_1):
            dR = np.sqrt((dx + r_2 * np.cos(phi_2) - r_1 * np.cos(phi_1)) ** 2 + \
            (dy + r_2 * np.sin(phi_2) - r_1 * np.sin(phi_1)) ** 2 + dz ** 2)
            return mu_0 / 4 /np.pi * r_1 * np.cos(k*dR) / dR

        def dA_x(phi_1):
            return dA(phi_1) * np.cos(phi_1)

        def dA_y(phi_1):
            return dA(phi_1) * np.sin(phi_1)

        A_x = integrate.quad(dA_x, 0, 2 * np.pi)[0]
        A_y = integrate.quad(dA_y, 0, 2 * np.pi)[0]

        return np.array([A_x, A_y])

    def A_imag(phi_2, dx, dy, dz, r_1, r_2, k):
    #r_1 = R_m, r_2 = R_n

        def dA(phi_1):
            dR = np.sqrt((dx + r_2 * np.cos(phi_2) - r_1 * np.cos(phi_1)) ** 2 + \
            (dy + r_2 * np.sin(phi_2) - r_1 * np.sin(phi_1)) ** 2 + dz ** 2)
            return mu_0 / 4 /np.pi * r_1 * np.sin(k*dR) / dR

        def dA_x(phi_1):
            return dA(phi_1) * np.cos(phi_1)
        
        def dA_y(phi_1):
            return dA(phi_1) * np.sin(phi_1)

        A_x = integrate.quad(dA_x, 0, 2 * np.pi)[0]
        A_y = integrate.quad(dA_y, 0, 2 * np.pi)[0]

        return np.array([A_x, A_y])

    def L_my(dx:float, dy:float, dz:float, r1:float, r2:float, k, width:float = 0) -> float:

        def f1(alpha):
            A = A_real(alpha, dx, dy, dz, r1, r2, k)
            return r2 * A[0] * np.cos(alpha) + r2 * A[1] * np.sin(alpha)

        def f2(alpha):
            A = A_imag(alpha, dx, dy, dz, r1, r2, k)
            return r2 * A[0] * np.cos(alpha) + r2 * A[1] * np.sin(alpha)

        imag = integrate.quad(f2, 0.0, 2*pi)[0]
        real = integrate.quad(f1, 0.0, 2*pi)[0]

        return real + 1j * imag
    
    return L_my(dx, dy, dz, r1, r2, k, width = 0)


def L_orthogonal(omega, dx:float, dy:float, dz:float, r1:float, r2:float, width:float = 0):
    k = omega/c
    def A_real(phi_2, dx, dy, dz, r_1, r_2, k):
    #r_1 = R_m, r_2 = R_n

#        def dA_x(phi_1):
#            dR = np.sqrt((dx + r_2 * np.cos(phi_2)) ** 2 + \
#            (dy + r_2 * np.sin(phi_2) + r_1 * np.sin(phi_1)) ** 2 + (dz+r_2 * np.cos(phi_2)) ** 2)
#            return mu_0 / 4 /np.pi * r_1  * np.cos(phi_1) * np.cos(k*dR) / dR

        def dA_y(phi_1):
            dR = np.sqrt((dx + r_2 * np.cos(phi_2)) ** 2 + \
            (dy + r_2 * np.sin(phi_2) + r_1 * np.sin(phi_1)) ** 2 + (dz+r_2 * np.cos(phi_2)) ** 2)
            return  mu_0 / 4 /np.pi * r_1  * np.sin(phi_1) * np.cos(k*dR) / dR

#        A_x = integrate.quad(dA_x, 0, 2 * np.pi)[0]
        A_y = integrate.quad(dA_y, 0, 2 * np.pi)[0]

        return np.array([0, A_y])

    def A_imag(phi_2, dx, dy, dz, r_1, r_2, k):
    #r_1 = R_m, r_2 = R_n

#        def dA_x(phi_1):
#            dR = np.sqrt((dx + r_2 * np.cos(phi_2)) ** 2 + \
#            (dy + r_2 * np.sin(phi_2) + r_1 * np.sin(phi_1)) ** 2 + (dz+r_2 * np.cos(phi_2)) ** 2)
#            return mu_0 / 4 /np.pi * r_1  * np.cos(phi_1) * np.sin(k*dR) / dR

        def dA_y(phi_1):
            dR = np.sqrt((dx + r_2 * np.cos(phi_2)) ** 2 + \
            (dy + r_2 * np.sin(phi_2) + r_1 * np.sin(phi_1)) ** 2 + (dz+r_2 * np.cos(phi_2)) ** 2)
            return  mu_0 / 4 /np.pi * r_1  * np.sin(phi_1) * np.sin(k*dR) / dR

#        A_x = integrate.quad(dA_x, 0, 2 * np.pi)[0]
        A_y = integrate.quad(dA_y, 0, 2 * np.pi)[0]

        return np.array([0, A_y])

    def L_my(dx:float, dy:float, dz:float, r1:float, r2:float, k, width:float = 0) -> float:

        def f1(alpha):
            A = A_real(alpha, dx, dy, dz, r1, r2, k)
            return (-1)*r2 * A[1] * np.cos(alpha)

        def f2(alpha):
            A = A_imag(alpha, dx, dy, dz, r1, r2, k)
            return (-1)*r2 * A[1] * np.cos(alpha)

        imag = integrate.quad(f2, 0.0, 2*pi)[0]
        real = integrate.quad(f1, 0.0, 2*pi)[0]

        return real + 1j * imag
    
    return L_my(dx, dy, dz, r1, r2, k, width = 0)

def asymp_par(omega:float, dist:float, r1:float, r2:float, accuracy, width:float = 0):
    Lambda = 2*np.pi*c/omega
    def S(x):
        return ((100.2-x)/10.7)**(-5/3)
    
    if accuracy > 0.9989529:
        return False
    elif accuracy <= 0.93136936:
        return True
    else:
        return S(accuracy*100)*r1<dist

def asymp_ort(omega:float, dist:float, r1:float, r2:float, accuracy, width:float = 0):
    Lambda = 2*np.pi*c/omega
    def S_0(x):
        return 89.831*(x-100)**3 + 0.0922*(x-100)**2 - 0.0006*(x-100) + 89.831 - accuracy*100
    
    def S_1(x):
        return 93.204*(x-150)**3 + 0.0466*(x-150)**2 - 0.0003*(x-150) + 93.204 - accuracy*100
    
    def S_2(x):
        return 94.8889*(x-200)**3 + 0.0247*(x-200)**2 - 0.0001*(x-200) + 94.8889 - accuracy*100
    
    def S_3(x):
        return 95.8974*(x-250)**3 + 0.0162*(x-250)**2 - 0.0001*(x-250) + 95.8974 - accuracy*100

    x_min_0 = 100
    x_max_0 = 150
    x_min_1 = x_max_0
    x_max_1 = 200
    x_min_2 = x_max_1
    x_max_2 = 250
    x_min_3 = x_max_2
    x_max_3 = 300
    step = 0.1

    if Lambda>30*r1:
        return False
    
    else:
        if dist<3*Lambda:
            return False
        else:

            if accuracy > 0.96567471:
                return False
            elif accuracy <= 0.89830974:
                return True
            elif 0.89830974< accuracy <= 0.93203955:
                x_values = np.arange(x_min_0, x_max_0 + step, step)
                roots = []

                for i in range(len(x_values) - 1):
                    a, b = x_values[i], x_values[i + 1]
                    if S_0(a) * S_0(b) < 0:
                        sol = root_scalar(S_0, bracket=[a, b], method='bisect', xtol=1e-6)
                        if sol.converged:
                            roots.append(sol.root)
                return max(roots)*r1<dist
            elif 0.93203955< accuracy <= 0.94888871:
                x_values = np.arange(x_min_1, x_max_1 + step, step)
                roots = []

                for i in range(len(x_values) - 1):
                    a, b = x_values[i], x_values[i + 1]
                    if S_1(a) * S_1(b) < 0:
                        sol = root_scalar(S_1, bracket=[a, b], method='bisect', xtol=1e-6)
                        if sol.converged:
                            roots.append(sol.root)
                return max(roots)*r1<dist
            elif 0.94888871< accuracy <= 0.95897383:
                x_values = np.arange(x_min_2, x_max_2 + step, step)
                roots = []

                for i in range(len(x_values) - 1):
                    a, b = x_values[i], x_values[i + 1]
                    if S_2(a) * S_2(b) < 0:
                        sol = root_scalar(S_2, bracket=[a, b], method='bisect', xtol=1e-6)
                        if sol.converged:
                            roots.append(sol.root)
                return max(roots)*r1<dist            
            elif 0.95897383< accuracy <= 0.96567471:
                x_values = np.arange(x_min_3, x_max_3 + step, step)
                roots = []

                for i in range(len(x_values) - 1):
                    a, b = x_values[i], x_values[i + 1]
                    if S_3(a) * S_3(b) < 0:
                        sol = root_scalar(S_3, bracket=[a, b], method='bisect', xtol=1e-6)
                        if sol.converged:
                            roots.append(sol.root)
                return max(roots)*r1<dist   
                     
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

    # if id_1 in Data:
    #     return Data[id_1]
    # elif id_2 in Data:
    #     return Data[id_2]
    
    if dx == 0 and dy == 0 and dz == 0:
        Data[id_1] = 0
        return 0

    # Consider all types of parallel orientation and symmetry for x-z axes
    R = np.sqrt(dx**2+dy**2+dz**2)
    
    if First_ring.pos == Second_ring.pos and not asymp_par(omega, R, r1, r2, accuracy):
        if First_ring.pos == "z":                           # Z-oriented rings
            l = L_parallel(omega, dy, dx, dz, r1, r2, w)
        elif First_ring.pos == "y":                         # Y-oriented rings
            l = L_parallel(omega, dy, -dz, dx, r1, r2, w)
        else:                                               # X-oriented rings
            l = L_parallel(omega, -dz, dx, dy, r1, r2, w)
        print('Double integral')

    elif First_ring.pos == Second_ring.pos and asymp_par(omega, R, r1, r2, accuracy):
        if First_ring.pos == "z":                           # Z-oriented rings
            l = L_par_asymp(omega, dx, dy, dz, r1, r2, w)
        elif First_ring.pos == "y":                         # Y-oriented rings
            l = L_par_asymp(omega, dx, -dz, dy, r1, r2, w)
        else:                                               # X-oriented rings
            l = L_par_asymp(omega, -dz, dy, dx, r1, r2, w)
        print('Single intergal')

        # Consider all types of orthogonal orientation

    elif First_ring.pos != Second_ring.pos and not asymp_ort(omega, R, r1, r2, accuracy):
        if First_ring.pos == "z":
            if Second_ring.pos == "y":                      # Z-Y oriented pair
                l = L_orthogonal(omega, dy, dx, dz, r1, r2, w)
            else:                                           # Z-X oriented pair
                l = L_orthogonal(omega, dx, dy, dz, r1, r2, w)
        elif First_ring.pos == "y":
            if Second_ring.pos == "z":                      # Y-Z oriented pair
                l = L_orthogonal(omega, dy, dz, dx, r1, r2, w)
            else:                                           # Y-X oriented pair
                l = L_orthogonal(omega, -dz, dy, dx, r1, r2,  w)
        elif First_ring.pos == "x":
            if Second_ring.pos == "z":                      # X-Z oriented pair
                l = L_orthogonal(omega, dx, dz, dy, r1, r2, w)
            else:                                           # X-Y oriented pair
                l = L_orthogonal(omega, dz, dx, dy, r1, r2, w)

    elif First_ring.pos != Second_ring.pos and asymp_ort(omega, R, r1, r2, accuracy):
        if First_ring.pos == "z":
            if Second_ring.pos == "y":                      # Z-Y oriented pair
                l = L_ort_asymp(omega, dx, dy, dz, r1, r2, w)
            else:                                           # Z-X oriented pair
                l = L_ort_asymp(omega, (dy), (dx), dz, r1, r2, w)
        elif First_ring.pos == "y":
            if Second_ring.pos == "z":                      # Y-Z oriented pair
                l = L_ort_asymp(omega, dx, (dz), (dy), r1, r2, w)
            else:                                           # Y-X oriented pair
                l = L_ort_asymp(omega, -dz, (dx), (dy), r1, r2,  w)
        elif First_ring.pos == "x":
            if Second_ring.pos == "z":                      # X-Z oriented pair
                l = L_ort_asymp(omega, dy, (dz), (dx), r1, r2, w)
            else:                                           # X-Y oriented pair
                l = L_ort_asymp(omega, (dz), dy, (dx), r1, r2, w)
    Data[id_1], Data[id_2] = [l * mu_0] * 2
    return l

# Calculating mutual inductance for each pair

def Matrix(omega, rings:List[Ring], Data:dict = {}) -> np.ndarray:
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
            M[n][m] = Mnm(omega, R1, R2, Data)
            M[m][n] = M[n][m]
    return M