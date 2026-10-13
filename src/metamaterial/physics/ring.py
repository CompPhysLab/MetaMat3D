"""Electrical response of split-ring resonators."""

from __future__ import annotations
import numpy as np

def self_inductance(omega, R, L, C):
    """Return the effective self-inductance M(omega)

    Parameters
    ----------
    omega : float or array-like
        Angular frequency in rad/s. Must be nonzero
    R : float
        Resistance in ohms
    L : float
        Self-inductance in henries
    C : float
        Capacitance in farads

    Returns
    -------
    complex or array-like
        Effective self-inductance according to the legacy model
    """
    if np.any(np.asarray(omega) == 0):
        raise ValueError("omega must be nonzero")

    return R / (1j * omega) - L + 1 / (omega**2 * C)

def impedance(omega, R, L, C):
    """Return the complex self-impedance Z(omega)

    Parameters
    ----------
    omega : float or array-like
        Angular frequency in rad/s. Must be nonzero
    R : float
        Resistance in ohms
    L : float
        Self-inductance in henries
    C : float
        Capacitance in farads

    Returns
    -------
    complex or array-like
        Complex impedance in ohms
    """
    if np.any(np.asarray(omega) == 0):
        raise ValueError("omega must be nonzero")

    return R - 1j * omega * L + 1j / (omega * C)

def conductivity(omega, R, L, C):
    """Return the complex conductivity sigma(omega) = 1 / Z(omega)"""
    Z = impedance(omega, R, L, C)
    return 1 / Z