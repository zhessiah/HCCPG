"""Training environment interfaces for CACC and VMAS."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import Tensor

from .config import CACCConfig, VMASConfig


class MultiAgentEnv(ABC):
    """Small normalized interface consumed by ``train.py``."""

    num_agents: int
    num_envs: int
    horizon: int
    observation_dim: int
    action_dim: int
    action_low: Tensor
    action_high: Tensor
    device: torch.device

    @abstractmethod
    def reset(self) -> Tensor:
        """Reset all vector environments and return [env, agent, observation]."""

    @abstractmethod
    def step(self, actions: Tensor) -> tuple[Tensor, Tensor, Tensor, dict[str, Any]]:
        """Step all environments and return observations, rewards, done flags, and info."""


class CACCEnv(MultiAgentEnv):
    """Vectorized longitudinal cooperative adaptive cruise-control interface."""

    observation_dim = 5
    action_dim = 1

    def __init__(
        self,
        scenario: str,
        config: CACCConfig,
        *,
        device: str | torch.device = "cpu",
        seed: int = 0,
    ) -> None:
        if scenario not in {"catchup", "slowdown"}:
            raise ValueError("CACC scenario must be 'catchup' or 'slowdown'")
        self.scenario = scenario
        self.config = config
        self.num_agents = config.num_agents
        self.num_envs = config.num_envs
        self.horizon = config.horizon
        self.device = torch.device(device)
        self.action_low = torch.tensor([config.min_accel], dtype=torch.float32, device=self.device)
        self.action_high = torch.tensor([config.max_accel], dtype=torch.float32, device=self.device)
        self._generator = torch.Generator(device="cpu").manual_seed(seed)
        self._step_count = 0
        self.positions = torch.empty(0, device=self.device)
        self.velocities = torch.empty(0, device=self.device)
        self.accelerations = torch.empty(0, device=self.device)
        self.reset()

    def reset(self) -> Tensor:
        cfg = self.config
        spacing = cfg.target_gap + cfg.vehicle_length
        vehicle_indices = torch.arange(
            self.num_agents + 1,
            dtype=torch.float32,
            device=self.device,
        )
        base_positions = -vehicle_indices * spacing
        if self.scenario == "catchup":
            base_positions[1:] -= 0.5 * cfg.target_gap
            lead_speed = cfg.catchup_lead_speed
            follower_speed = cfg.catchup_follower_speed
        else:
            lead_speed = cfg.slowdown_initial_speed
            follower_speed = cfg.slowdown_initial_speed
        position_noise = self._noise((self.num_envs, self.num_agents + 1), scale=0.5)
        velocity_noise = self._noise((self.num_envs, self.num_agents + 1), scale=0.2)
        self.positions = base_positions.unsqueeze(0).repeat(self.num_envs, 1) + position_noise
        self.velocities = torch.full(
            (self.num_envs, self.num_agents + 1),
            follower_speed,
            dtype=torch.float32,
            device=self.device,
        )
        self.velocities[:, 0] = lead_speed
        self.velocities += velocity_noise
        self.accelerations = torch.zeros_like(self.velocities)
        self._step_count = 0
        return self._observations()

    def step(self, actions: Tensor) -> tuple[Tensor, Tensor, Tensor, dict[str, Any]]:
        if actions.shape != (self.num_envs, self.num_agents, self.action_dim):
            raise ValueError(
                f"expected actions with shape {(self.num_envs, self.num_agents, self.action_dim)}, "
                f"got {tuple(actions.shape)}"
            )
        cfg = self.config
        controlled_acceleration = actions.squeeze(-1).clamp(cfg.min_accel, cfg.max_accel)
        previous_controlled_acceleration = self.accelerations[:, 1:].clone()
        lead_acceleration = torch.zeros(self.num_envs, device=self.device)
        if self.scenario == "slowdown" and self._step_count >= cfg.slowdown_start_step:
            slowing = self.velocities[:, 0] > cfg.slowdown_target_speed
            lead_acceleration = torch.where(
                slowing,
                torch.full_like(lead_acceleration, cfg.slowdown_accel),
                lead_acceleration,
            )
        self.accelerations[:, 0] = lead_acceleration
        self.accelerations[:, 1:] = controlled_acceleration

        self.velocities = (
            self.velocities + self.accelerations * cfg.dt
        ).clamp(0.0, cfg.max_speed)
        if self.scenario == "slowdown":
            self.velocities[:, 0].clamp_(min=cfg.slowdown_target_speed)
        self.positions = self.positions + self.velocities * cfg.dt
        self._step_count += 1

        gaps = self.positions[:, :-1] - self.positions[:, 1:] - cfg.vehicle_length
        spacing_error = (gaps - cfg.target_gap) / cfg.target_gap
        relative_velocity = (self.velocities[:, :-1] - self.velocities[:, 1:]) / 10.0
        normalized_acceleration = controlled_acceleration / max(abs(cfg.min_accel), abs(cfg.max_accel))
        jerk = (controlled_acceleration - previous_controlled_acceleration) / max(
            abs(cfg.min_accel), abs(cfg.max_accel)
        )
        collision = gaps < 0.5
        cooperative_cost = (
            0.60 * spacing_error.square().mean(dim=1)
            + 0.25 * relative_velocity.square().mean(dim=1)
            + 0.03 * normalized_acceleration.square().mean(dim=1)
            + 0.02 * jerk.square().mean(dim=1)
            + 5.0 * collision.to(torch.float32).mean(dim=1)
        )
        shared_reward = -cooperative_cost
        rewards = shared_reward.unsqueeze(-1).expand(-1, self.num_agents).clone()
        dones = torch.full(
            (self.num_envs,),
            self._step_count >= self.horizon,
            dtype=torch.bool,
            device=self.device,
        )
        info = {
            "mean_gap": float(gaps.mean()),
            "mean_spacing_error": float(spacing_error.abs().mean()),
            "collision_rate": float(collision.to(torch.float32).mean()),
        }
        return self._observations(), rewards, dones, info

    def _observations(self) -> Tensor:
        cfg = self.config
        gaps = self.positions[:, :-1] - self.positions[:, 1:] - cfg.vehicle_length
        spacing_error = (gaps - cfg.target_gap) / cfg.target_gap
        relative_velocity = (self.velocities[:, :-1] - self.velocities[:, 1:]) / 10.0
        own_velocity = self.velocities[:, 1:] / cfg.max_speed
        own_acceleration = self.accelerations[:, 1:] / max(abs(cfg.min_accel), abs(cfg.max_accel))
        predecessor_acceleration = self.accelerations[:, :-1] / max(
            abs(cfg.min_accel), abs(cfg.max_accel)
        )
        observations = torch.stack(
            (
                spacing_error,
                relative_velocity,
                own_velocity,
                own_acceleration,
                predecessor_acceleration,
            ),
            dim=-1,
        )
        return observations.clamp(-5.0, 5.0)

    def _noise(self, shape: tuple[int, ...], *, scale: float) -> Tensor:
        return torch.randn(shape, generator=self._generator, dtype=torch.float32).to(self.device) * scale


class VMASEnv(MultiAgentEnv):
    """Adapter for the official unwrapped VMAS tensor API."""

    def __init__(
        self,
        scenario: str,
        config: VMASConfig,
        *,
        device: str | torch.device = "cpu",
        seed: int = 0,
    ) -> None:
        if scenario not in {"dropout", "dispersion"}:
            raise ValueError("VMAS scenario must be 'dropout' or 'dispersion'")
        try:
            import vmas
        except ImportError as error:
            raise RuntimeError(
                "VMAS is required for VMAS tasks. Install it with: pip install -e '.[vmas]'"
            ) from error
        self.scenario = scenario
        self.num_agents = config.num_agents
        self.num_envs = config.num_envs
        self.horizon = config.horizon
        self.device = torch.device(device)
        self._environment = vmas.make_env(
            scenario=scenario,
            num_envs=config.num_envs,
            device=str(self.device),
            continuous_actions=True,
            max_steps=config.horizon,
            seed=seed,
            dict_spaces=False,
            clamp_actions=True,
            grad_enabled=False,
            terminated_truncated=True,
            n_agents=config.num_agents,
        )
        initial_observations = self._normalize_observations(self._environment.reset())
        self.observation_dim = initial_observations.shape[-1]
        random_actions = self._environment.get_random_actions()
        if len(random_actions) != self.num_agents:
            raise RuntimeError("VMAS returned an unexpected number of agent actions")
        action_dims = {int(action.shape[-1]) for action in random_actions}
        if len(action_dims) != 1:
            raise RuntimeError("SCCPG requires equal VMAS action dimensions across agents")
        self.action_dim = action_dims.pop()
        self.action_low = torch.full((self.action_dim,), -1.0, device=self.device)
        self.action_high = torch.full((self.action_dim,), 1.0, device=self.device)

    def reset(self) -> Tensor:
        return self._normalize_observations(self._environment.reset())

    def step(self, actions: Tensor) -> tuple[Tensor, Tensor, Tensor, dict[str, Any]]:
        if actions.shape != (self.num_envs, self.num_agents, self.action_dim):
            raise ValueError(
                f"expected actions with shape {(self.num_envs, self.num_agents, self.action_dim)}, "
                f"got {tuple(actions.shape)}"
            )
        result = self._environment.step([actions[:, index] for index in range(self.num_agents)])
        if len(result) == 5:
            observations, rewards, terminated, truncated, infos = result
            dones = self._normalize_done(terminated) | self._normalize_done(truncated)
        elif len(result) == 4:
            observations, rewards, done, infos = result
            dones = self._normalize_done(done)
        else:
            raise RuntimeError(f"VMAS step returned {len(result)} values; expected four or five")
        normalized_observations = self._normalize_observations(observations)
        normalized_rewards = self._normalize_rewards(rewards)
        return normalized_observations, normalized_rewards, dones, {"raw_info": infos}

    def _normalize_observations(self, observations: Any) -> Tensor:
        if not isinstance(observations, (list, tuple)) or len(observations) != self.num_agents:
            raise RuntimeError("VMAS observations must be a list with one tensor per agent")
        tensors = []
        dimensions = set()
        for observation in observations:
            if not isinstance(observation, Tensor) or observation.ndim != 2:
                raise RuntimeError("VMAS task must return flat tensor observations")
            tensors.append(observation.to(self.device, dtype=torch.float32))
            dimensions.add(int(observation.shape[-1]))
        if len(dimensions) != 1:
            raise RuntimeError("SCCPG requires equal VMAS observation dimensions across agents")
        return torch.stack(tensors, dim=1)

    def _normalize_rewards(self, rewards: Any) -> Tensor:
        if not isinstance(rewards, (list, tuple)) or len(rewards) != self.num_agents:
            raise RuntimeError("VMAS rewards must be a list with one tensor per agent")
        return torch.stack(
            [torch.as_tensor(reward, device=self.device, dtype=torch.float32) for reward in rewards],
            dim=1,
        )

    def _normalize_done(self, done: Any) -> Tensor:
        if isinstance(done, Tensor):
            tensor = done.to(device=self.device, dtype=torch.bool)
            if tensor.ndim == 0:
                tensor = tensor.expand(self.num_envs)
            if tensor.ndim == 2:
                tensor = tensor.any(dim=1)
            return tensor.reshape(self.num_envs)
        if isinstance(done, (list, tuple)):
            tensors = [torch.as_tensor(item, device=self.device, dtype=torch.bool) for item in done]
            stacked = torch.stack(tensors, dim=1)
            return stacked.any(dim=1)
        raise RuntimeError("VMAS done signal has an unsupported type")


def make_environment(
    name: str,
    *,
    num_agents: int,
    num_envs: int,
    horizon: int,
    device: str | torch.device,
    seed: int,
) -> MultiAgentEnv:
    """Construct one of the four supported training environments."""

    normalized_name = name.lower().replace("_", "-")
    if normalized_name in {"cacc-catchup", "cacc-slowdown"}:
        scenario = normalized_name.removeprefix("cacc-")
        return CACCEnv(
            scenario,
            CACCConfig(num_agents=num_agents, num_envs=num_envs, horizon=horizon),
            device=device,
            seed=seed,
        )
    if normalized_name in {"vmas-dropout", "vmas-dispersion"}:
        scenario = normalized_name.removeprefix("vmas-")
        return VMASEnv(
            scenario,
            VMASConfig(num_agents=num_agents, num_envs=num_envs, horizon=horizon),
            device=device,
            seed=seed,
        )
    raise ValueError(
        "unsupported environment. Choose one of: cacc-catchup, cacc-slowdown, "
        "vmas-dropout, vmas-dispersion"
    )
