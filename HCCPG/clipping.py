"""Receiver-centered critic-parameter clipping from SCCPG."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters


def flatten_parameters(module: nn.Module) -> Tensor:
    """Return a detached copy of all module parameters as one vector."""

    return parameters_to_vector(module.parameters()).detach().clone()


def load_parameters_from_vector(module: nn.Module, vector: Tensor) -> None:
    """Load a flat parameter vector into ``module`` after strict validation."""

    expected = sum(parameter.numel() for parameter in module.parameters())
    if vector.ndim != 1:
        raise ValueError(f"parameter vector must be one-dimensional, got {vector.shape}")
    if vector.numel() != expected:
        raise ValueError(f"parameter vector has {vector.numel()} entries; expected {expected}")
    if not torch.isfinite(vector).all():
        raise ValueError("parameter vector contains non-finite values")
    first_parameter = next(module.parameters(), None)
    if first_parameter is None:
        if vector.numel() != 0:
            raise ValueError("cannot load parameters into a parameter-free module")
        return
    source = vector.to(device=first_parameter.device, dtype=first_parameter.dtype)
    with torch.no_grad():
        vector_to_parameters(source, module.parameters())


def radial_clip(displacement: Tensor, radius: float) -> Tensor:
    """Apply Euclidean radial clipping, corresponding to Eq. (5)."""

    if radius < 0.0:
        raise ValueError("radius must be nonnegative")
    if not torch.isfinite(displacement).all():
        raise ValueError("displacement contains non-finite values")
    norm = torch.linalg.vector_norm(displacement)
    if norm.item() == 0.0 or norm.item() <= radius:
        return displacement.clone()
    return displacement * (radius / norm)


def reconstruct_critic(receiver: Tensor, incoming: Tensor, radius: float) -> Tensor:
    """Project an incoming critic vector onto the receiver-centered ball."""

    if receiver.ndim != 1 or incoming.ndim != 1:
        raise ValueError("receiver and incoming critic vectors must be one-dimensional")
    if receiver.shape != incoming.shape:
        raise ValueError(
            f"receiver and incoming critic vectors must have equal shape; "
            f"got {receiver.shape} and {incoming.shape}"
        )
    if not torch.isfinite(receiver).all() or not torch.isfinite(incoming).all():
        raise ValueError("critic vectors must contain only finite values")
    return receiver + radial_clip(incoming - receiver, radius)
