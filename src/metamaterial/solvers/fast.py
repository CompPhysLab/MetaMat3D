import numpy as np
import numpy.typing as npt
from tqdm import tqdm
from scipy.sparse.linalg import LinearOperator, gmres
from src.metamaterial.config import SimulationConfig
from src.metamaterial.geometry import Geometry, orientation_normals
from src.metamaterial.physics import self_inductance, build_rhs
from src.metamaterial.physics.coupling.operator import build_fft_operator
from src.metamaterial.solvers.preconditioners import build_orientation_block_preconditioner
ComplexArray = npt.NDArray[np.complex128]

def solve_system(config: SimulationConfig, geometry: Geometry) -> ComplexArray:
    """Solves the linear system using the FFT-based mutual-inductance operator

    Parameters
    ----------
    config : SimulationConfig
        Simulation configuration
    geometry : Geometry
        Geometry of the system

    Returns
    -------
    currents : np.ndarray
        Complex currents, shape (N, n_omegas)
    """
    omegas = np.asarray(config.omegas, dtype=np.float64)
    normals = orientation_normals(geometry.orientations)

    n_rings = len(geometry.centers)
    currents = np.zeros((n_rings, len(omegas)), dtype=np.complex128)

    x0 = None
    M = None
    if config.interaction_model == "quasistatic":
        M = build_fft_operator(geometry.grid_indices, geometry.orientations, counts=(config.Nx, config.Ny, config.Nz), spacing=(config.Dx, config.Dy, config.Dz), type=config.type, radius=config.radius, width=config.width, model=config.interaction_model, omega=None)

    for i, omega in tqdm(enumerate(omegas)):
        M_0 = self_inductance(omega, config.R, config.L, config.C)*np.ones((len(geometry.orientations)))
        if config.interaction_model != "quasistatic":
            M = build_fft_operator(geometry.grid_indices, geometry.orientations, counts=(config.Nx, config.Ny, config.Nz), spacing=(config.Dx, config.Dy, config.Dz), type=config.type, radius=config.radius, width=config.width, model=config.interaction_model, omega=float(omega))
        rhs = build_rhs(geometry.centers, normals, config.radius, config.H_0, config.H_direction, model=config.interaction_model, k_direction=config.k_direction, omega=omega)
        A = -M

        def matvec(vector: np.ndarray) -> np.ndarray:
            return M_0 * vector + A @ vector
        
        system = LinearOperator(shape=(n_rings, n_rings), matvec=matvec, dtype=np.complex128)
        solution, info = gmres(system, rhs, x0=x0, rtol=1e-10, atol=0.0)
        if info != 0:
            if info > 0:
                raise RuntimeError(
                    f"GMRES did not converge for omega={omega}: "
                    f"reached iteration limit ({info})."
                )
            raise RuntimeError(
                f"GMRES failed for omega={omega}, info={info}."
            )
        x0 = solution
        currents[:, i] = solution
    return currents