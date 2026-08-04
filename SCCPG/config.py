"""Configuration objects for SCCPG and its training environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(slots=True)
class SCCPGConfig:
    """Algorithm defaults, matching the values reported in the paper where available."""

    gamma: float = 0.99
    actor_lr: float = 5e-4
    critic_lr: float = 2.5e-4
    hidden_sizes: Tuple[int, ...] = (64, 64)
    entropy_coef: float = 0.1
    max_grad_norm: float = 3.0
    clipping_radius: float = 1.0
    actor_batch_size: int = 60
    critic_batch_size: int = 60
    critic_epochs: int = 4
    parameter_bound: float = 10.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.gamma < 1.0:
            raise ValueError("gamma must satisfy 0 <= gamma < 1")
        if self.actor_lr <= 0.0 or self.critic_lr <= 0.0:
            raise ValueError("learning rates must be positive")
        if not self.hidden_sizes or any(size <= 0 for size in self.hidden_sizes):
            raise ValueError("hidden_sizes must contain positive integers")
        if self.entropy_coef < 0.0:
            raise ValueError("entropy_coef must be nonnegative")
        if self.max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be positive")
        if self.clipping_radius < 0.0:
            raise ValueError("clipping_radius must be nonnegative")
        if self.actor_batch_size <= 0 or self.critic_batch_size <= 0:
            raise ValueError("batch sizes must be positive")
        if self.critic_epochs <= 0:
            raise ValueError("critic_epochs must be positive")
        if self.parameter_bound <= 0.0:
            raise ValueError("parameter_bound must be positive")


@dataclass(slots=True)
class CACCConfig:
    """Defaults for the built-in vectorized CACC interface."""

    num_agents: int = 5
    num_envs: int = 1
    horizon: int = 100
    dt: float = 0.1
    target_gap: float = 20.0
    vehicle_length: float = 5.0
    max_speed: float = 40.0
    min_accel: float = -5.0
    max_accel: float = 3.0
    catchup_lead_speed: float = 22.0
    catchup_follower_speed: float = 15.0
    slowdown_initial_speed: float = 25.0
    slowdown_target_speed: float = 12.0
    slowdown_start_step: int = 25
    slowdown_accel: float = -3.0

    def __post_init__(self) -> None:
        if self.num_agents < 2:
            raise ValueError("CACC requires at least two controlled vehicles")
        if self.num_envs <= 0 or self.horizon <= 0:
            raise ValueError("num_envs and horizon must be positive")
        if self.dt <= 0.0 or self.target_gap <= 0.0 or self.vehicle_length <= 0.0:
            raise ValueError("dt, target_gap, and vehicle_length must be positive")
        if self.min_accel >= self.max_accel:
            raise ValueError("min_accel must be smaller than max_accel")


@dataclass(slots=True)
class VMASConfig:
    """Defaults for VMAS task construction."""

    num_agents: int = 5
    num_envs: int = 32
    horizon: int = 100

    def __post_init__(self) -> None:
        if self.num_agents < 2:
            raise ValueError("VMAS requires at least two agents")
        if self.num_envs <= 0 or self.horizon <= 0:
            raise ValueError("num_envs and horizon must be positive")


@dataclass(slots=True)
class TrainConfig:
    """Top-level training defaults."""

    env_name: str = "cacc-catchup"
    total_steps: int = 1_000_000
    seed: int = 0
    device: str = "cpu"
    neighborhood_size: int = 3
    log_interval: int = 1
    checkpoint: str | None = None

    def __post_init__(self) -> None:
        if self.total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if self.neighborhood_size <= 0:
            raise ValueError("neighborhood_size must be positive")
        if self.log_interval <= 0:
            raise ValueError("log_interval must be positive")
