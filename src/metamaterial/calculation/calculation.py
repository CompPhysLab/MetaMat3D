import numpy as np
import numpy.typing as npt
import os
import json
from src.metamaterial.config import SimulationConfig
from src.metamaterial.geometry import build_geometry, orientation_normals
from src.metamaterial.solvers import straight, fast
from src.metamaterial.results import results
_SOLVERS = {
    "straight": straight,
    "fast": fast,
}
FloatArray = npt.NDArray[np.float64]
ComplexArray = npt.NDArray[np.complex128]

def calculate(config: SimulationConfig) -> None:
    '''Calculate the metamaterial response for a given configuration and save the results to disk'''
    print(f"Calculating system {config.name}")
    geometry = build_geometry(config)
    #make directory for results in DATA
    os.makedirs(f"DATA/{config.name}", exist_ok=True)
    #save the simulation configuration
    with open(f"DATA/{config.name}/config.json", "w") as f:
        json.dump(config.to_dict(), f)
    #save the simulation results
    solver = _SOLVERS[config.solver]
    I = solver.solve_system(config, geometry) # (N, len(omegas))
    if config.save_currents:
        np.save(f"DATA/{config.name}/currents.npy", I)
        print(f"Currents saved to DATA/{config.name}/currents.npy")
    normals = orientation_normals(geometry.orientations)
    if config.save_polarizability:
        print(f"Saving polarizability")
        polarizability = np.zeros(len(config.omegas), dtype=np.complex128)
        for i, omega in enumerate(config.omegas):
            polarizability[i] = results.polarizability(I[:, i], config.radius, normals, config.H_0, config.H_direction)
        np.save(f"DATA/{config.name}/polarizability.npy", polarizability)
        print(f"Polarizability saved to DATA/{config.name}/polarizability.npy")
    if config.save_extinction_cross_section:
        print(f"Saving extinction cross section")
        sigma_ext = np.zeros_like(config.omegas, dtype=np.float64)
        for i, omega in enumerate(config.omegas):
            sigma_ext[i] = results.extinction(I[:, i], config.radius, normals, geometry.centers, config.H_0, config.H_direction, config.k_direction, omega)
        np.save(f"DATA/{config.name}/extinction_cross_section.npy", sigma_ext)
        print(f"extinction cross section saved to DATA/{config.name}/extinction_cross_section.npy")
    print(f"")
    