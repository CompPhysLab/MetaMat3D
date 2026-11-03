"""Torch preconditioners for iterative linear solvers."""

import torch

Tensor = torch.Tensor


def build_jacobi_preconditioner(diagonal: Tensor) -> Tensor:
    """Return inverse diagonal entries for Jacobi left preconditioning."""
    diagonal = diagonal.to(torch.complex128)
    if torch.any(diagonal == 0):
        raise ValueError("Jacobi preconditioner has a zero diagonal entry")
    return torch.reciprocal(diagonal)
