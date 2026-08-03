"""Continuous-control actor and scalar critic networks."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.distributions import Normal


def build_mlp(
    input_dim: int,
    hidden_sizes: Sequence[int],
    output_dim: int,
    *,
    output_activation: nn.Module | None = None,
) -> nn.Sequential:
    """Build a compact ReLU multilayer perceptron."""

    if input_dim <= 0 or output_dim <= 0:
        raise ValueError("input_dim and output_dim must be positive")
    layers: list[nn.Module] = []
    previous = input_dim
    for size in hidden_sizes:
        if size <= 0:
            raise ValueError("hidden layer sizes must be positive")
        layers.extend((nn.Linear(previous, size), nn.ReLU()))
        previous = size
    layers.append(nn.Linear(previous, output_dim))
    if output_activation is not None:
        layers.append(output_activation)
    return nn.Sequential(*layers)


class GaussianActor(nn.Module):
    """Tanh-squashed diagonal-Gaussian policy with bounded actions."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int],
        action_low: Tensor,
        action_high: Tensor,
    ) -> None:
        super().__init__()
        if action_dim <= 0:
            raise ValueError("action_dim must be positive")
        low = torch.as_tensor(action_low, dtype=torch.float32).reshape(-1)
        high = torch.as_tensor(action_high, dtype=torch.float32).reshape(-1)
        if low.numel() != action_dim or high.numel() != action_dim:
            raise ValueError("action bounds must contain one entry per action dimension")
        if not torch.all(high > low):
            raise ValueError("every action upper bound must exceed its lower bound")
        self.mean_network = build_mlp(observation_dim, hidden_sizes, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))
        self.register_buffer("action_scale", (high - low) / 2.0)
        self.register_buffer("action_bias", (high + low) / 2.0)

    def distribution(self, observations: Tensor) -> Normal:
        mean = self.mean_network(observations)
        std = self.log_std.clamp(-5.0, 2.0).exp().expand_as(mean)
        return Normal(mean, std)

    def sample(
        self,
        observations: Tensor,
        *,
        deterministic: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        distribution = self.distribution(observations)
        pre_tanh = distribution.mean if deterministic else distribution.rsample()
        squashed = torch.tanh(pre_tanh)
        action = squashed * self.action_scale + self.action_bias
        log_probability = self._log_probability(distribution, pre_tanh, squashed)
        entropy = distribution.entropy().sum(dim=-1)
        return action, log_probability, entropy

    def log_prob(self, observations: Tensor, actions: Tensor) -> tuple[Tensor, Tensor]:
        normalized = (actions - self.action_bias) / self.action_scale
        normalized = normalized.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        pre_tanh = torch.atanh(normalized)
        distribution = self.distribution(observations)
        log_probability = self._log_probability(distribution, pre_tanh, normalized)
        entropy = distribution.entropy().sum(dim=-1)
        return log_probability, entropy

    def _log_probability(self, distribution: Normal, pre_tanh: Tensor, squashed: Tensor) -> Tensor:
        base_log_prob = distribution.log_prob(pre_tanh).sum(dim=-1)
        squash_correction = torch.log(1.0 - squashed.square() + 1e-6).sum(dim=-1)
        scale_correction = torch.log(self.action_scale).sum()
        return base_log_prob - squash_correction - scale_correction


class ScalarCritic(nn.Module):
    """Ordinary scalar action-value approximator Q_w(z, a)."""

    def __init__(self, input_dim: int, hidden_sizes: Sequence[int]) -> None:
        super().__init__()
        self.network = build_mlp(input_dim, hidden_sizes, 1)

    def forward(self, critic_inputs: Tensor) -> Tensor:
        return self.network(critic_inputs).squeeze(-1)
