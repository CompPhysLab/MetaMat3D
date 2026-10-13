"""Physics package"""

from .excitation import build_rhs
from .ring import self_inductance, impedance, conductivity

__all__ = ["build_rhs", "self_inductance", "impedance", "conductivity"]