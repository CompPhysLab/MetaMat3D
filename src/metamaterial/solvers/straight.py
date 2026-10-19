import numpy as np
import numpy.typing as npt
from typing import Literal
from tqdm import tqdm
from scipy.linalg import solve
from src.metamaterial.config import SimulationConfig
from src.metamaterial.geometry import Geometry, orientation_normals
from src.metamaterial.physics import mutual_inductance_matrix
from src.metamaterial.physics import self_inductance
from src.metamaterial.physics import build_rhs
ComplexArray = npt.NDArray[np.complex128]

def solve_system(config: SimulationConfig, geometry: Geometry) -> ComplexArray:
    '''Solve the linear system for the given configuration and geometry

    Parameters
    ----------
    config : SimulationConfig
        simulation configuration
    geometry : Geometry
        geometry of the system

    Returns
    -------
    currents : np.ndarray
        Complex currents, shape (N, n_omegas)
    '''
    omegas = np.asarray(config.omegas, dtype=np.float64)
    normals = orientation_normals(geometry.orientations)

    currents = np.zeros((len(geometry.centers), len(omegas)),dtype=np.complex128)

    if config.interaction_model=='quasistatic':
        M = mutual_inductance_matrix(geometry.centers, geometry.orientations, radius=config.radius, width=config.width, model=config.interaction_model, omega=None)

    for i, omega in tqdm(enumerate(omegas)):
        M_0 = self_inductance(omega, config.R, config.L, config.C)*np.ones((len(geometry.orientations)))
        if config.interaction_model != "quasistatic":
            M = mutual_inductance_matrix(geometry.centers, geometry.orientations, radius=config.radius, width=config.width, model=config.interaction_model, omega=omega)
        rhs = build_rhs(geometry.centers, normals, config.radius, config.H_0, config.H_direction, model=config.interaction_model, k_direction=config.k_direction, omega=omega)
        currents[:, i] = solve((np.diag(M_0)-M), rhs, assume_a='sym')
    return currents