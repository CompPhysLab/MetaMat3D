import numpy as np
from scipy import integrate

c=3e8

def L_parallel(omega:float, dx:float, dy:float, dz:float, r1:float, r2:float) -> complex:
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

    Returns
    -------
    complex
        mutual inductance between two parallel-oriented rings
    """
    k = omega/c
    def internal(phi1, phi2):
        A = dz**2+dx**2+dy**2 + 2*r2*(dx*np.cos(phi2)+dy*np.sin(phi2)) + r1**2+r2**2
        B = -2*dx*r1-2*r1*r2*np.cos(phi2)
        C = -2*dy*r1-2*r1*r2*np.sin(phi2)
        arg = A+B*np.cos(phi1)+C*np.sin(phi1)
        if arg<0:
            return 0
        R = np.sqrt(arg)
        spr = np.cos(phi2-phi1)
        return spr*np.cos(k*R)/R, spr*np.sin(k*R)/R
    def internal_re(phi1,phi2):
        return internal(phi1, phi2)[0]
    def internal_im(phi1,phi2):
        return internal(phi1, phi2)[1]
    def outer_re(phi2):
        return integrate.quad(internal_re, 0, 2*np.pi, args=(phi2,))[0]
    def outer_im(phi2):
        return integrate.quad(internal_im, 0, 2*np.pi, args=(phi2,))[0]
    re = integrate.quad(outer_re, 0, 2*np.pi)[0]
    im = integrate.quad(outer_im, 0, 2*np.pi)[0]
    L = (re+1j*im)*r1*r2/4/np.pi
    return L

def L_orthogonal(omega:float, dx:float, dy:float, dz:float, r1:float, r2:float) -> complex:
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

    Returns
    -------
    complex
        mutual inductance between two orthogonal-oriented rings
    """
    k = omega/c
    def internal(phi1, phi2):
        A = dz**2+dx**2+dy**2 + 2*r2*(dy*np.sin(phi2)-dz*np.cos(phi2)) + r1**2+r2**2
        B = 2*dx*r1
        C = 2*dy*r1+2*r1*r2*np.sin(phi2)
        arg = A+B*np.cos(phi1)+C*np.sin(phi1)
        if arg<0:
            return 0
        R = np.sqrt(arg)
        spr = np.cos(phi1)*np.cos(phi2)
        return spr*np.cos(k*R)/R, spr*np.sin(k*R)/R
    def internal_re(phi1,phi2):
        return internal(phi1, phi2)[0]
    def internal_im(phi1,phi2):
        return internal(phi1, phi2)[1]
    def outer_re(phi2):
        return integrate.quad(internal_re, 0, 2*np.pi, args=(phi2,))[0]
    def outer_im(phi2):
        return integrate.quad(internal_im, 0, 2*np.pi, args=(phi2,))[0]
    re = integrate.quad(outer_re, 0, 2*np.pi)[0]
    im = integrate.quad(outer_im, 0, 2*np.pi)[0]
    L = -(re+1j*im)*r1*r2/4/np.pi
    return L