"""Minimal on-policy rollout storage for multi-agent continuous control."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(slots=True)
class RolloutBatch:
    """A time-major batch with one observation/action/reward block per agent."""

    observations: Tensor
    actions: Tensor
    rewards: Tensor
    dones: Tensor
    returns: Tensor

    def __post_init__(self) -> None:
        if self.observations.ndim != 4:
            raise ValueError("observations must have shape [time, env, agent, feature]")
        if self.actions.ndim != 4:
            raise ValueError("actions must have shape [time, env, agent, feature]")
        if self.rewards.ndim != 3 or self.returns.ndim != 3:
            raise ValueError("rewards and returns must have shape [time, env, agent]")
        if self.dones.ndim != 2:
            raise ValueError("dones must have shape [time, env]")
        prefix = self.observations.shape[:3]
        if self.actions.shape[:3] != prefix:
            raise ValueError("observation and action prefixes must match")
        if self.rewards.shape != prefix or self.returns.shape != prefix:
            raise ValueError("reward and return shapes must match observation prefix")
        if self.dones.shape != prefix[:2]:
            raise ValueError("done shape must match time and environment dimensions")

    @property
    def num_steps(self) -> int:
        return self.observations.shape[0]

    @property
    def num_envs(self) -> int:
        return self.observations.shape[1]

    @property
    def num_agents(self) -> int:
        return self.observations.shape[2]

    @property
    def sample_count(self) -> int:
        return self.num_steps * self.num_envs

    def agent_observations(self, agent_index: int) -> Tensor:
        return self.observations[:, :, agent_index].reshape(self.sample_count, -1)

    def agent_actions(self, agent_index: int) -> Tensor:
        return self.actions[:, :, agent_index].reshape(self.sample_count, -1)

    def agent_returns(self, agent_index: int) -> Tensor:
        return self.returns[:, :, agent_index].reshape(self.sample_count)

    def critic_inputs(self) -> Tensor:
        observations = self.observations.reshape(self.sample_count, -1)
        actions = self.actions.reshape(self.sample_count, -1)
        return torch.cat((observations, actions), dim=-1)


class RolloutBuffer:
    """Collect one rollout segment and compute discounted local returns."""

    def __init__(self) -> None:
        self._observations: list[Tensor] = []
        self._actions: list[Tensor] = []
        self._rewards: list[Tensor] = []
        self._dones: list[Tensor] = []

    def __len__(self) -> int:
        return len(self._observations)

    def add(self, observations: Tensor, actions: Tensor, rewards: Tensor, dones: Tensor) -> None:
        if observations.ndim != 3:
            raise ValueError("observations must have shape [env, agent, feature]")
        if actions.ndim != 3:
            raise ValueError("actions must have shape [env, agent, feature]")
        if rewards.ndim != 2:
            raise ValueError("rewards must have shape [env, agent]")
        if dones.ndim != 1:
            raise ValueError("dones must have shape [env]")
        if observations.shape[:2] != actions.shape[:2] or observations.shape[:2] != rewards.shape:
            raise ValueError("environment and agent dimensions must match")
        if observations.shape[0] != dones.shape[0]:
            raise ValueError("done count must equal number of environments")
        self._observations.append(observations.detach())
        self._actions.append(actions.detach())
        self._rewards.append(rewards.detach())
        self._dones.append(dones.detach().to(dtype=torch.bool))

    def build(self, gamma: float) -> RolloutBatch:
        if not self._observations:
            raise RuntimeError("cannot build an empty rollout")
        if not 0.0 <= gamma < 1.0:
            raise ValueError("gamma must satisfy 0 <= gamma < 1")
        observations = torch.stack(self._observations)
        actions = torch.stack(self._actions)
        rewards = torch.stack(self._rewards)
        dones = torch.stack(self._dones)
        returns = torch.zeros_like(rewards)
        running = torch.zeros_like(rewards[0])
        for time_index in range(rewards.shape[0] - 1, -1, -1):
            continuation = (~dones[time_index]).to(rewards.dtype).unsqueeze(-1)
            running = rewards[time_index] + gamma * running * continuation
            returns[time_index] = running
        return RolloutBatch(
            observations=observations,
            actions=actions,
            rewards=rewards,
            dones=dones,
            returns=returns,
        )
