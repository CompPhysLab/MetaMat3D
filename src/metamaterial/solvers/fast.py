import numpy as np
import numpy.typing as npt
from tqdm import tqdm
from scipy.sparse.linalg import LinearOperator, gmres
from src.metamaterial.config import SimulationConfig
from src.metamaterial.geometry import Geometry, orientation_normals
from src.metamaterial.physics import self_inductance, build_rhs
from src.metamaterial.physics.coupling.operator import build_fft_operator
from src.metamaterial.solvers.preconditioners import build_preconditioner
ComplexArray = npt.NDArray[np.complex128]

def solve_system(config: SimulationConfig, geometry: Geometry, *, use_preconditioner: bool = True) -> ComplexArray:
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
    spacings = np.asarray([config.Dx, config.Dy, config.Dz], dtype=np.float64)
    preconditioner_cutoff = 2
    currents = np.zeros((n_rings, len(omegas)), dtype=np.complex128)

    x0 = None
    M = None
    if config.interaction_model == "quasistatic":
        M = build_fft_operator(geometry.centers, geometry.orientations,
                               counts=(config.Nx, config.Ny, config.Nz), spacings=spacings,
                               type=config.type, radius=config.radius, width=config.width,
                               model=config.interaction_model, omega=None)
        rhs = build_rhs(geometry.centers, spacings, normals,
                        config.radius, config.H_0, config.H_direction,
                        model=config.interaction_model, k_direction=None, omega=None)

    # if use_preconditioner:
    #     P = build_preconditioner(geometry.centers, spacings, geometry.orientations, self_inductance(omegas[0], config.R, config.L, config.C),
    #                              radius=config.radius, width=config.width,
    #                              cutoff_distance=preconditioner_cutoff,
    #                              model='quasistatic', omega=None)

    for i, omega in enumerate(tqdm(omegas)):
        M_0 = self_inductance(omega, config.R, config.L, config.C)
        if config.interaction_model != "quasistatic":
            M = build_fft_operator(geometry.centers, geometry.orientations,
                                   counts=(config.Nx, config.Ny, config.Nz), spacings=spacings,
                                   type=config.type, radius=config.radius, width=config.width,
                                   model=config.interaction_model, omega=float(omega))
            rhs = build_rhs(geometry.centers, spacings, normals,
                            config.radius, config.H_0, config.H_direction,
                            model=config.interaction_model, k_direction=config.k_direction, omega=omega)

        def matvec(vector: np.ndarray) -> np.ndarray:
            return M_0 * vector - M @ vector
        
        system = LinearOperator(shape=(n_rings, n_rings), matvec=matvec, dtype=np.complex128)
        if use_preconditioner:
            P = build_preconditioner(geometry.centers, spacings, geometry.orientations, M_0,
                                    radius=config.radius, width=config.width,
                                    cutoff_distance=preconditioner_cutoff,
                                    model=config.interaction_model, omega=float(omega))
            #P = np.ones((n_rings))/M_0
            preconditioned_system = LinearOperator(shape=(n_rings, n_rings),
                                                   matvec=lambda x: system @(P @ x), dtype=np.complex128)
        else:
            preconditioned_system = system
        solution, info = gmres(preconditioned_system, rhs, x0=x0, rtol=1e-10, atol=0.0)
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
        currents[:, i] = P * solution if use_preconditioner else solution
    return currents