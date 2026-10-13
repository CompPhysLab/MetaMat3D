from functools import lru_cache
import numpy as np
from scipy.interpolate import BarycentricInterpolator
from scipy import special

c=3e8

#functions for numerical computation of recurrent inductance for parallel and orthogonal oriented rings
@lru_cache(maxsize=None)
def diff_theta(N:int) -> tuple[np.ndarray, np.ndarray]:
    '''Computes the differentiation matrix and the Chebyshev nodes
    
    Parameters
    ----------
    N : int
        Number of Chebyshev nodes minus one (N+1 nodes in total)

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        tuple containing the Chebyshev nodes and the differentiation matrix
    '''
    theta = np.pi * np.arange(N + 1) / N
    u = np.cos(theta)
    const = np.ones(N + 1)
    const[0] = 2.0
    const[N] = 2.0
    const *= (-1.0) ** np.arange(N + 1)
    X = np.tile(u, (N + 1, 1))
    dX = X.T - X
    D = np.outer(const, 1.0 / const) / (dX + np.eye(N + 1))
    D -= np.diag(D.sum(axis=1))
    return u, D

def compute_ln_par(r_1: float, r_2: float, k: float, N_max: int, N_theta=200) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    '''Computes the Ln modes for parallel-oriented rings

    Parameters
    ----------
    r_1 : float
        radius of the first ring
    r_2 : float
        radius of the second ring
    k : float
        wavenumber
    N_max : int
        maximum number of modes to compute
    N_theta : int, optional
        number of Chebyshev nodes, by default 200

    Returns
    -------
    tuple[np.ndarray, dict[int, np.ndarray]]
        tuple containing the Chebyshev nodes and a dictionary of the Ln modes
    '''
    u, D = diff_theta(N_theta)
    D2 = D @ D
    B = np.diag(1.0 - u**2) @ D2 - 2.0 * np.diag(u) @ D
    sinth = np.sqrt(np.maximum(1.0 - u**2, 0.0))
    L1 = np.pi * r_1 * r_2 * special.jv(1, k * r_1 * sinth) * special.jv(1, k * r_2 * sinth)
    Ln_data = {1: L1.astype(complex)}
    for n in range(1, N_max):
        Ln = Ln_data[n]
        BLn = B @ Ln
        Ln_data[n + 1] = ((n) * Ln + (1.0 / (n+1)) * BLn) / (2j * k)
    return u, Ln_data

def ln_interpolators_par(u_grid: np.ndarray, Ln_data: dict[int, np.ndarray]) -> dict[int, callable]:
    '''Creates interpolators for the Ln modes of parallel-oriented rings
    
    Parameters
    ----------
    u_grid : np.ndarray
        Chebyshev nodes
    Ln_data : dict[int, np.ndarray]
        dictionary of the Ln modes

    Returns
    -------
    dict[int, callable]
        dictionary of the interpolators for each Ln mode
    '''
    interpolators = {}
    for n, vals in Ln_data.items():
        interp_re = BarycentricInterpolator(u_grid, vals.real)
        interp_im = BarycentricInterpolator(u_grid, vals.imag)
        interpolators[n] = (
            lambda th, _re=interp_re, _im=interp_im:
                _re(np.cos(th)) + 1j * _im(np.cos(th))
                )
    return interpolators

def L_cached_par(N: int, r: float, k: float, theta: float, ln_interps: dict[int, callable]) -> complex:
    '''Computes the total inductance for a parallel-oriented ring configuration

    Parameters
    ----------
    N : int
        number of modes to include in the computation
    r : float
        distance between the centers of the rings
    k : float
        wavenumber
    theta : float
        angle at which to evaluate the inductance
    ln_interps : dict[int, callable]
        dictionary of interpolators for each Ln mode

    Returns
    -------
    complex
        total inductance at the specified angle
    '''
    total = 0.0 + 0j
    for i in range(1, N + 1):
        total += ln_interps[i](theta) / (r ** i)*np.exp(1j * k * r)
    return total

def compute_Fnm_modes(r_1: float, r_2: float, k: float, N_max: int, N_theta=200, M_modes=5) -> tuple[np.ndarray, dict[int, dict[int, np.ndarray]]]:
    '''Computes the Fnm modes for orthogonal-oriented rings
    
    Parameters
    ----------
    r_1 : float
        radius of the first ring
    r_2 : float
        radius of the second ring
    k : float
        wavenumber
    N_max : int
        maximum number of modes to include in the computation
    N_theta : int, optional
        number of theta points, by default 200
    M_modes : int, optional
        number of phi modes to include in the computation, by default 5

    Returns
    -------
    tuple[np.ndarray, dict[int, dict[int, np.ndarray]]]
        Chebyshev nodes and the Fnm modes
    '''
    u, D = diff_theta(N_theta)
    N = N_theta
    D2 = D @ D
    sinth = np.sqrt(np.maximum(1.0 - u**2, 0.0))
    cos_th = u
    N_phi = 2 * M_modes + 3
    phi_grid = np.linspace(0, 2*np.pi, N_phi, endpoint=False)
    L1_grid = np.zeros((N+1, N_phi), dtype=complex)
    for j, phi in enumerate(phi_grid):
        sin_phi = np.sin(phi)
        sin2_th = sinth**2
        under = np.sqrt(np.maximum(1.0 - sin2_th * sin_phi**2, 0.0))
        L1_grid[:, j] = -np.pi * r_1 * r_2 * (sin_phi * cos_th / under) \
                        * special.jv(1, k * r_1 * sinth) * special.jv(1, k * r_2 * under)
    L1_modes = np.fft.fft(L1_grid, axis=1) / N_phi
    F_data = {}
    for m in range(-M_modes, M_modes + 1):
        abs_m = abs(m)
        B_m = (np.diag(1.0 - u**2) @ D2 
               - 2.0 * np.diag(u * (abs_m + 1)) @ D
               - abs_m * (abs_m + 1) * np.eye(N + 1))
        if m >= 0:
            idx = m
        else:
            idx = N_phi + m
        L1_m = L1_modes[:, idx]
        w = (1.0 - u**2) ** (abs_m / 2.0)
        F1_m = np.zeros(N + 1, dtype=complex)
        F1_m[1:-1] = L1_m[1:-1] / w[1:-1]
        if abs_m > 0:
            interp = BarycentricInterpolator(u[1:-1], F1_m[1:-1])
            F1_m[0]  = interp(u[0])
            F1_m[-1] = interp(u[-1])
        else:
            F1_m = L1_m.copy()
        F_data[m] = {1: F1_m}
        for n in range(1, N_max):
            Fn = F_data[m][n]
            BFn = B_m @ Fn
            F_data[m][n + 1] = ((n - 1) * Fn + (1.0 / n) * BFn) / (2j * k)
    return u, F_data

def make_F_interpolators(u_grid: np.ndarray, F_data: dict[int, dict[int, np.ndarray]]) -> dict[int, dict[int, BarycentricInterpolator]]:
    '''Creates interpolators for the Fnm modes of orthogonal-oriented rings
    
    Parameters
    ----------
    u_grid : np.ndarray
        Chebyshev nodes
    F_data : dict[int, dict[int, np.ndarray]]
        Fnm modes

    Returns
    -------
    dict[int, dict[int, BarycentricInterpolator]]
        interpolators for the Fnm modes
    '''
    interpolators = {}
    for m, n_data in F_data.items():
        interpolators[m] = {}
        for n, Fn in n_data.items():
            interpolators[m][n] = BarycentricInterpolator(u_grid, Fn)
    return interpolators

def Ln_at_point(interpolators: dict[int, dict[int, BarycentricInterpolator]], m_modes: int, theta: float, phi: float, n: int) -> complex:
    '''Evaluates the Ln mode at a specific point (theta, phi) using the interpolators

    Parameters
    ----------
    interpolators : dict[int, dict[int, BarycentricInterpolator]]
        interpolators for the Fnm modes
    m_modes : int
        number of modes
    theta : float
        polar angle
    phi : float
        azimuthal angle
    n : int
        mode number

    Returns
    -------
    complex
        value of the Ln mode at the specified point
    '''

    u = np.cos(theta)
    result = 0.0 + 0.0j
    for m in range(-m_modes, m_modes + 1):
        abs_m = abs(m)
        F_val = interpolators[m][n](u)
        w = (1.0 - u**2) ** (abs_m / 2.0)
        result += (w* F_val* np.exp(1j * m * phi))
    return result

def L_cached_ort(N: int, r: float, k: float, theta: float, phi: float, interpolators: dict[int, dict[int, BarycentricInterpolator]], m_modes: int):
    '''Computes the total inductance for an orthogonal-oriented ring configuration

    Parameters
    ----------
    N : int
        number of modes
    r : float
        distance between the centers of the rings
    k : float
        wavenumber
    theta : float
        polar angle
    phi : float
        azimuthal angle
    interpolators : dict[int, dict[int, BarycentricInterpolator]]
        interpolators for the Fnm modes
    m_modes : int
        number of m modes

    Returns
    -------
    complex
        total inductance for the orthogonal-oriented ring configuration
    '''
    total = 0.0 + 0.0j
    for n in range(1, N + 1):
        Ln = Ln_at_point(interpolators, m_modes, theta, phi, n)
        total += Ln / r**n *np.exp(1j * k * r)
    return total

#cache for recurrent interpolators, keyed by (r1, r2, k, N)
_recurrent_cache_par = {}
_recurrent_cache_ort = {}

def _get_recurrent_interps_par(r1: float, r2: float, k: float, N: int) -> dict[int, callable]:
    '''Saves cached interpolators for the Ln modes of parallel-oriented rings

    Parameters
    ----------
    r1 : float
        radius of the first ring
    r2 : float
        radius of the second ring
    k : float
        wavenumber
    N : int
        number of modes

    Returns
    -------
    dict[int, callable]
        interpolators for the Ln modes  
    '''
    key = (r1, r2, k, N)
    if key not in _recurrent_cache_par:
        u_grid, ln_data = compute_ln_par(r1, r2, k, N)
        _recurrent_cache_par[key] = ln_interpolators_par(u_grid, ln_data)
    return _recurrent_cache_par[key]

def _get_recurrent_interps_ort(r1: float, r2: float, k: float, N: int, M: int = 5) -> dict[int, callable]:
    '''Saves cached interpolators for the Fnm modes of orthogonal-oriented rings

    Parameters
    ----------
    r1 : float
        radius of the first ring
    r2 : float
        radius of the second ring
    k : float
        wavenumber
    N : int
        number of modes
    M : int, optional
        number of phi modes, by default 5

    Returns
    -------
    dict[int, callable]
        interpolators for the Fnm modes
    '''
    key = (r1, r2, k, N)
    if key not in _recurrent_cache_ort:
        u_grid, F_data = compute_Fnm_modes(r1, r2, k, N, M_modes=M)
        _recurrent_cache_ort[key] = make_F_interpolators(u_grid, F_data)
    return _recurrent_cache_ort[key]

def L_parallel(omega: float, dx:float, dy:float, dz:float, r1:float, r2:float) -> complex:
    '''computes the recurrent inductance for a parallel-oriented ring configuration

    Parameters
    ----------
    omega : float
        angular frequency
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
    '''
    k=omega/c
    dist = np.sqrt(dx ** 2 + dy ** 2+dz**2)
    db = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(db, dz)
    N = 2 #fixing 2nd order approximation
    ln_interps = _get_recurrent_interps_par(r1, r2, k, N)
    return L_cached_par(N, float(dist), k, theta, ln_interps)

def L_orthogonal(omega: float, dx:float, dy:float, dz:float, r1:float, r2:float) -> complex:
    '''computes the recurrent inductance for a orthogonal-oriented ring configuration

    Parameters
    ----------
    omega : float
        angular frequency
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
    '''
    k=omega/c
    dist = np.sqrt(dx ** 2 + dy ** 2+dz**2)
    db = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(db, dz)
    phi = np.arctan2(dy, dx)
    N = 2 #fixing 2nd order approximation
    F_interps = _get_recurrent_interps_ort(r1, r2, k, N, M=5)
    return L_cached_ort(N, float(dist), k, theta, phi, F_interps, 5)


def clear_static_cache():
    """Clear all cached data (recurrent interpolators)
    Call when geometry changes between simulations
    """
    diff_theta.cache_clear()
    _recurrent_cache_par.clear()
    _recurrent_cache_ort.clear()