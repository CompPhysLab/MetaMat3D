import numpy as np
import numpy.typing as npt
from typing import Literal
FloatArray = npt.NDArray[np.float64]
ComplexArray = npt.NDArray[np.complex128]
StrArray = npt.NDArray[np.str_]

c = 3e8
mu_0 = 4.0 * np.pi * 1e-7

def mag_moment(currents: ComplexArray, radius: float, normals: FloatArray) -> ComplexArray:
    '''Calculates the magnetic moment of the sample
    
    Parameters
    ----------
    currents: ComplexArray
        Currents in each ring, shape (N,)
    radius: float
        Common mean ring radius
    normals: FloatArray
        Ring normals, shape (N, 3)
    '''
    sum = 0
    for i in range(len(currents)):
        sum += currents[i] * (np.pi * radius**2) * normals[i]
    return sum

def polarizability(currents: ComplexArray, radius: float, normals: FloatArray, H_0: float, H_direction: FloatArray) -> complex:
    '''Calculates the polarizability of the sample
    
    Parameters
    ----------
    currents: ComplexArray
        Currents in each ring, shape (N,)
    radius: float
        Common mean ring radius
    normals: FloatArray
        Ring normals, shape (N, 3)
    H_0: float
        Applied magnetic field amplitude
    H_direction: FloatArray
        Direction of the applied magnetic field, shape (3,)
    '''
    m = mag_moment(currents, radius, normals)
    return m @ H_direction / H_0

def extinction(currents: ComplexArray, radius: float, normals: FloatArray, centers: FloatArray, H_0: float, H_direction: FloatArray, k_direction: FloatArray, omega: float) -> float:
    '''Calculates the extinction cross-section of the sample
    
    Parameters
    ----------
    currents: ComplexArray
        Currents in each ring, shape (N,)
    radius: float
        Common mean ring radius
    normals: FloatArray
        Ring normals, shape (N, 3)
    centers: FloatArray
        Ring centers, shape (N, 3)
    H_0: float
        Applied magnetic field amplitude
    H_direction: FloatArray
        Direction of the applied magnetic field, shape (3,)
    k_direction: FloatArray
        Direction of the incident wavevector, shape (3,)
    omega: float
        Angular frequency
    '''
    sum = 0
    for i in range(len(currents)):
        prod1 = normals[i] @ H_direction
        prod2 = k_direction @ centers[i]
        sum += currents[i] * (np.pi * radius**2) * prod1 * np.exp(-1j * omega / c * prod2)
    return omega / (c * H_0) * np.imag(sum)