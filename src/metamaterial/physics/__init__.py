"""Physics package"""

from .excitation import build_rhs
from .ring import self_inductance, impedance, conductivity
from .coupling.operator import mutual_inductance, mutual_inductance_matrix

__all__ = ["build_rhs", "self_inductance", "impedance", "conductivity", "mutual_inductance", "mutual_inductance_matrix"]