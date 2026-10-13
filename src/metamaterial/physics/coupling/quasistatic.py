import numpy as np
from scipy import integrate
from scipy import special

K = special.ellipk  # Сomplete elliptic integral of the first kind
E = special.ellipe  # Сomplete elliptic integral of the second kind

def L_parallel(dx:float, dy:float, dz:float, r1:float, r2:float) -> float:
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

    Returns
    -------
    float
        mutual inductance between two parallel-oriented rings
    """    
    # Define function to integrate over first defined parameter
    def dl(alpha, dx, dy, dz, r_1, r_2):
        db = np.sqrt(dx ** 2 + dy ** 2)
        dp = np.sqrt(r_2 ** 2 + db ** 2 + 2 * r_2 * db * np.cos(alpha))
        kappa_sq = 4 * r_1 * dp / ((dp + r_1) ** 2 + dz ** 2)
        kappa = np.sqrt(kappa_sq)
        A = 1/(2*np.pi)*np.sqrt(r_1/dp) * ((2/kappa - kappa) * K(kappa_sq) - 2 * E(kappa_sq)/kappa)
        return A * r_2 * (r_2 + db * np.cos(alpha))/dp
    L, err = integrate.quad(dl, 0, 2 * np.pi, args= (dx, dy, dz, r1, r2))
    return L

def L_orthogonal(dx:float, dy:float ,dz:float, r1:float, r2:float) -> float:
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

    Returns
    -------
    float
        mutual inductance between two orthogonal-oriented rings
    """    
    def dl(alpha, dx, dy, dz, r_1, r_2):
        dp = np.sqrt((dx - r_2 * np.sin(alpha)) ** 2 + dy ** 2)
        kappa_sq = 4 * r_1 * dp / ((dp + r_1) ** 2 + (dz - r_2 * np.cos(alpha)) ** 2)
        kappa = np.sqrt(kappa_sq)
        A = 1 / (2 * np.pi) * np.sqrt(r_1 / (dp)) * ((2 / kappa - kappa) * K(kappa_sq) - 2 * E(kappa_sq) / kappa)
        return A * r_2 * dy * np.cos(alpha) / (dp)
    L, err = integrate.quad(dl, 0, 2 * np.pi, args=(dx, dy, dz, r1, r2))
    return L