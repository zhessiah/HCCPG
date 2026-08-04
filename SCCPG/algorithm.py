"""Multi-agent orchestration for Self-Centered Clipped Critic Policy Gradient."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import torch
from torch import Tensor

from .agent import SCCPGAgent
from .buffer import RolloutBatch
from .config import SCCPGConfig
from .topology import CommunicationGraph


class SCCPG:
    """A decentralized collection of local actors and exchanged scalar critics."""

    def __init__(
        self,
        *,
        num_agents: int,
        observation_dim: int,
        action_dim: int,
        action_low: Tensor,
        action_high: Tensor,
        graph: CommunicationGraph,
        config: SCCPGConfig,
        device: str | torch.device = "cpu",
    ) -> None:
        if graph.num_agents != num_agents:
            raise ValueError("communication graph size must equal num_agents")
        self.num_agents = num_agents
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.graph = graph
        self.config = config
        self.device = torch.device(device)
        critic_input_dim = num_agents * (observation_dim + action_dim)
        low = torch.as_tensor(action_low, dtype=torch.float32)
        high = torch.as_tensor(action_high, dtype=torch.float32)
        self.agents = [
            SCCPGAgent(
                agent_index=index,
                observation_dim=observation_dim,
                action_dim=action_dim,
                critic_input_dim=critic_input_dim,
                action_low=low,
                action_high=high,
                config=config,
                device=self.device,
            )
            for index in range(num_agents)
        ]

    @torch.no_grad()
    def act(self, observations: Tensor, *, deterministic: bool = False) -> tuple[Tensor, Tensor]:
        if observations.ndim != 3:
            raise ValueError("observations must have shape [env, agent, feature]")
        if observations.shape[1:] != (self.num_agents, self.observation_dim):
            raise ValueError(
                f"expected observation shape [env, {self.num_agents}, {self.observation_dim}], "
                f"got {tuple(observations.shape)}"
            )
        actions = []
        log_probabilities = []
        for index, agent in enumerate(self.agents):
            action, log_probability = agent.act(observations[:, index], deterministic=deterministic)
            actions.append(action)
            log_probabilities.append(log_probability)
        return torch.stack(actions, dim=1), torch.stack(log_probabilities, dim=1)

    def update(self, batch: RolloutBatch) -> dict[str, float]:
        if batch.num_agents != self.num_agents:
            raise ValueError("rollout agent count does not match algorithm")
        critic_inputs = batch.critic_inputs().to(self.device)
        critic_losses = []
        for index, agent in enumerate(self.agents):
            critic_losses.append(
                agent.fit_critic(critic_inputs, batch.agent_returns(index).to(self.device))
            )

        critic_vectors = {index: agent.critic_vector() for index, agent in enumerate(self.agents)}
        actor_losses = []
        entropies = []
        accepted_displacements = []
        for receiver, agent in enumerate(self.agents):
            weights = self.graph.weights_for(receiver)
            incoming = {sender: critic_vectors[sender] for sender in weights}
            actor_loss, entropy, displacement = agent.update_actor(
                observations=batch.agent_observations(receiver).to(self.device),
                actions=batch.agent_actions(receiver).to(self.device),
                critic_inputs=critic_inputs,
                receiver_vector=critic_vectors[receiver],
                incoming_vectors=incoming,
                weights=weights,
            )
            actor_losses.append(actor_loss)
            entropies.append(entropy)
            accepted_displacements.append(displacement)

        return {
            "critic_loss": sum(critic_losses) / len(critic_losses),
            "actor_loss": sum(actor_losses) / len(actor_losses),
            "entropy": sum(entropies) / len(entropies),
            "accepted_displacement": sum(accepted_displacements) / len(accepted_displacements),
        }

    def checkpoint(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "num_agents": self.num_agents,
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "actors": [agent.actor.state_dict() for agent in self.agents],
            "critics": [agent.critic.state_dict() for agent in self.agents],
        }
