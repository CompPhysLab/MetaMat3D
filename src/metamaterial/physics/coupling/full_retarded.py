''' Trapezoidal rule: int f(phi)dphi = 2 pi/n * sum f(phi_i), phi_i = 2 pi*i/n, i=0,...,n-1'''

import numpy as np
from scipy import integrate

c=3e8

def L_parallel(omega:float, dx:float, dy:float, dz:float, r1:float, r2:float, n:int=128) -> complex:
    """Calculates mutual inductance between two parallel-oriented rings

    Parameters
    ----------
    omega : float
        frequency of the electromagnetic wave
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
    n : int, optional
        number of quadrature points, by default 128

    Returns
    -------
    complex
        mutual inductance between two parallel-oriented rings
    """
    k = omega/c
    phi = 2 * np.pi * np.arange(n)/n
    phi1, phi2 = np.meshgrid(phi, phi, indexing='ij')
    A = dz**2 + dx**2 + dy**2 + 2*r2*(dx*np.cos(phi2) + dy*np.sin(phi2)) + r1**2 + r2**2
    B = -2*r1*(dx + r2*np.cos(phi2))
    C = -2*r1*(dy + r2*np.sin(phi2))
    arg = A+B*np.cos(phi1)+C*np.sin(phi1)
    R = np.sqrt(np.max(arg, 0.0))
    spr = np.cos(phi2-phi1)
    kernel = spr * np.exp(1j*k*R)/R
    L = kernel.sum() * np.pi * r1 * r2 / n**2
    return L

def L_orthogonal(omega:float, dx:float, dy:float, dz:float, r1:float, r2:float, n: int=128) -> complex:
    """Calculates mutual inductance between two orthogonal-oriented rings

    Parameters
    ----------
    omega : float
        frequency of the electromagnetic wave
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
    n : int, optional
        number of quadrature points, by default 128

    Returns
    -------
    complex
        mutual inductance between two orthogonal-oriented rings
    """
    k = omega/c
    phi = 2 * np.pi * np.arange(n)/n
    phi1, phi2 = np.meshgrid(phi, phi, indexing='ij')
    A = dz**2 + dx**2 + dy**2 + 2*r2*(dy*np.sin(phi2) - dz*np.cos(phi2)) + r1**2 + r2**2
    B = 2*dx*r1
    C = 2*r1*(dy + r2*np.sin(phi2))
    arg = A+B*np.cos(phi1)+C*np.sin(phi1)
    R = np.sqrt(np.max(arg, 0.0))
    spr = np.cos(phi1)*np.cos(phi2)
    kernel = spr * np.exp(1j*k*R)/R
    L = - kernel.sum() * np.pi * r1 * r2 / n**2
    return L