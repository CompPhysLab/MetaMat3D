from pathlib import Path

import numpy as np
import pytest
import torch

from src.pytorch_version.calculation import calculate
from src.pytorch_version.config import SimulationConfig
from src.pytorch_version.geometry import build_geometry
from src.pytorch_version.physics import (
    build_rhs,
    mutual_inductance,
    mutual_inductance_matrix,
    self_inductance,
)
from src.pytorch_version.physics.coupling.operator import build_fft_operator
from src.pytorch_version.solvers import fast, straight
from src.metamaterial.physics.coupling.full_retarded import L_parallel as legacy_parallel


def make_config(*, model="quasistatic", solver="straight", output=False):
    return SimulationConfig(
        name="torch_test",
        R=1.0,
        L=1e-6,
        C=1e-9,
        radius=1e-3,
        width=1e-4,
        Dx=3e-3,
        Dy=3e-3,
        Dz=3e-3,
        orientations="zy",
        type="center",
        Nx=2,
        Ny=1,
        Nz=1,
        packing="rectangle",
        interaction_model=model,
        H_0=1.0,
        H_direction=(0.0, 0.0, 1.0),
        k_direction=(1.0, 0.0, 0.0),
        omegas=np.array([1e8, 2e8]),
        solver=solver,
        save_currents=output,
        save_polarizability=output,
        save_extinction_cross_section=output,
    )


def available_devices():
    return ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", available_devices())
@pytest.mark.parametrize("model", ["quasistatic", "full_retarded"])
def test_torch_direct_and_gmres_solvers_agree(device, model):
    config = make_config(model=model)
    geometry = build_geometry(config, device=device)

    direct = straight.solve_system(config, geometry, device=device)
    iterative = fast.solve_system(
        config, geometry, device=device, use_preconditioner=False
    )

    assert direct.device.type == device
    assert direct.dtype == torch.complex128
    assert torch.isfinite(direct).all()
    torch.testing.assert_close(iterative, direct, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("device", available_devices())
def test_torch_physics_outputs_stay_on_device(device):
    geometry = build_geometry(make_config(), device=device)
    normals = torch.tensor(
        [[0, 0, 1], [0, 1, 0], [0, 0, 1], [0, 1, 0]],
        dtype=torch.int64,
        device=device,
    )
    rhs = build_rhs(
        geometry.centers,
        (3e-3, 3e-3, 3e-3),
        normals,
        1e-3,
        1.0,
        (0.0, 0.0, 1.0),
    )
    self_response = self_inductance(1e8, 1.0, 1e-6, 1e-9, device=device)
    coupling = mutual_inductance(
        displacement=(3e-3, 0.0, 0.0),
        r1=1e-3,
        r2=1e-3,
        width=1e-4,
        orientation1="z",
        orientation2="z",
        model="full_retarded",
        omega=1e8,
        device=device,
    )

    assert rhs.device.type == device
    assert self_response.device.type == device
    assert coupling.device.type == device
    assert rhs.dtype == torch.complex128
    assert self_response.dtype == torch.complex128
    assert coupling.dtype == torch.complex128
    assert torch.isfinite(rhs).all()
    assert torch.isfinite(self_response)
    assert torch.isfinite(coupling)


def test_torch_approx_retarded_reports_unported_backend():
    with pytest.raises(NotImplementedError, match="has not yet been ported"):
        mutual_inductance(
            displacement=(3e-3, 0.0, 0.0),
            r1=1e-3,
            r2=1e-3,
            width=0.0,
            orientation1="z",
            orientation2="z",
            model="approx_retarded",
            omega=1e8,
        )


def test_torch_full_retarded_coupling_matches_legacy_quadrature():
    displacement = (0.0, 0.0, 3e-3)
    omega = 1e8
    expected = (
        4.0
        * np.pi
        * 1e-7
        * legacy_parallel(omega, *displacement, 1e-3, 1e-3)
    )
    actual = mutual_inductance(
        displacement=displacement,
        r1=1e-3,
        r2=1e-3,
        width=0.0,
        orientation1="z",
        orientation2="z",
        model="full_retarded",
        omega=omega,
    )

    assert actual.item() == pytest.approx(expected, rel=1e-12, abs=1e-20)


@pytest.mark.parametrize("device", available_devices())
@pytest.mark.parametrize("meta_atom_type", ["center", "border", "open"])
@pytest.mark.parametrize("model", ["quasistatic", "full_retarded"])
def test_torch_fft_operator_matches_dense_matrix(device, meta_atom_type, model):
    config = make_config(model=model)
    config = SimulationConfig(
        **{
            **config.to_dict(),
            "type": meta_atom_type,
            "Nx": 2,
            "Ny": 2,
            "Nz": 2,
            "orientations": "zyx",
            "omegas": np.array([1e8, 2e8]),
        }
    )
    geometry = build_geometry(config, device=device)
    omega = None if model == "quasistatic" else 1e8
    fft_operator = build_fft_operator(
        geometry.centers,
        geometry.orientations,
        counts=(config.Nx, config.Ny, config.Nz),
        spacings=(config.Dx, config.Dy, config.Dz),
        type=config.type,
        radius=config.radius,
        width=config.width,
        model=model,
        omega=omega,
    )
    dense = mutual_inductance_matrix(
        geometry.centers,
        (config.Dx, config.Dy, config.Dz),
        geometry.orientations,
        radius=config.radius,
        width=config.width,
        model=model,
        omega=omega,
    )
    vector = torch.arange(
        len(geometry.orientations), dtype=torch.float64, device=device
    ).to(torch.complex128)

    torch.testing.assert_close(
        fft_operator @ vector,
        dense @ vector,
        rtol=1e-11,
        atol=1e-22,
    )


def test_calculate_runs_and_saves_torch_results(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = make_config(output=True)

    calculate(config, device="cpu")

    output = tmp_path / "DATA" / config.name
    assert (output / "config.json").is_file()
    currents = np.load(output / "currents.npy")
    polarizability = np.load(output / "polarizability.npy")
    extinction = np.load(output / "extinction_cross_section.npy")
    assert currents.shape == (4, 2)
    assert polarizability.shape == (2,)
    assert extinction.shape == (2,)
    assert np.isfinite(currents).all()
    assert np.isfinite(polarizability).all()
    assert np.isfinite(extinction).all()
