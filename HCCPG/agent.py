"""Local SCCPG actor-critic agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn

from .clipping import flatten_parameters, load_parameters_from_vector, reconstruct_critic
from .config import SCCPGConfig
from .networks import GaussianActor, ScalarCritic


@dataclass(slots=True)
class AgentUpdateMetrics:
    critic_loss: float
    actor_loss: float
    entropy: float
    accepted_displacement: float


class SCCPGAgent:
    """One decentralized actor and its ordinary scalar critic."""

    def __init__(
        self,
        *,
        agent_index: int,
        observation_dim: int,
        action_dim: int,
        critic_input_dim: int,
        action_low: Tensor,
        action_high: Tensor,
        config: SCCPGConfig,
        device: torch.device,
    ) -> None:
        self.agent_index = agent_index
        self.config = config
        self.device = device
        self.actor = GaussianActor(
            observation_dim,
            action_dim,
            config.hidden_sizes,
            action_low,
            action_high,
        ).to(device)
        self.critic = ScalarCritic(critic_input_dim, config.hidden_sizes).to(device)
        self.critic_evaluator = ScalarCritic(critic_input_dim, config.hidden_sizes).to(device)
        self.critic_evaluator.requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)

    @torch.no_grad()
    def act(self, observations: Tensor, *, deterministic: bool = False) -> tuple[Tensor, Tensor]:
        actions, log_probability, _ = self.actor.sample(observations, deterministic=deterministic)
        return actions, log_probability

    def critic_vector(self) -> Tensor:
        return flatten_parameters(self.critic)

    def fit_critic(self, critic_inputs: Tensor, targets: Tensor) -> float:
        if critic_inputs.shape[0] != targets.shape[0]:
            raise ValueError("critic input and target sample counts must match")
        sample_count = critic_inputs.shape[0]
        losses: list[float] = []
        for _ in range(self.config.critic_epochs):
            permutation = torch.randperm(sample_count, device=critic_inputs.device)
            for start in range(0, sample_count, self.config.critic_batch_size):
                indices = permutation[start : start + self.config.critic_batch_size]
                prediction = self.critic(critic_inputs[indices])
                loss = nn.functional.mse_loss(prediction, targets[indices])
                self.critic_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
                self.critic_optimizer.step()
                self._project_parameters(self.critic)
                losses.append(float(loss.detach()))
        return sum(losses) / max(len(losses), 1)

    def update_actor(
        self,
        *,
        observations: Tensor,
        actions: Tensor,
        critic_inputs: Tensor,
        receiver_vector: Tensor,
        incoming_vectors: Mapping[int, Tensor],
        weights: Mapping[int, float],
    ) -> tuple[float, float, float]:
        if set(incoming_vectors) != set(weights):
            raise ValueError("incoming critic vectors and aggregation weights must have identical keys")
        if any(weight < 0.0 for weight in weights.values()):
            raise ValueError("aggregation weights must be nonnegative")
        weight_sum = sum(weights.values())
        if abs(weight_sum - 1.0) > 1e-6:
            raise ValueError(f"aggregation weights must sum to one, got {weight_sum}")
        sample_count = observations.shape[0]
        batch_size = min(self.config.actor_batch_size, sample_count)
        indices = torch.randperm(sample_count, device=observations.device)[:batch_size]
        selected_observations = observations[indices]
        selected_actions = actions[indices]
        selected_critic_inputs = critic_inputs[indices]

        aggregated_values = torch.zeros(batch_size, device=self.device)
        displacements: list[float] = []
        with torch.no_grad():
            for sender, weight in weights.items():
                reconstructed = reconstruct_critic(
                    receiver_vector,
                    incoming_vectors[sender],
                    self.config.clipping_radius,
                )
                displacement = torch.linalg.vector_norm(reconstructed - receiver_vector)
                displacements.append(float(displacement))
                load_parameters_from_vector(self.critic_evaluator, reconstructed)
                aggregated_values.add_(self.critic_evaluator(selected_critic_inputs), alpha=float(weight))

        log_probability, entropy = self.actor.log_prob(selected_observations, selected_actions)
        policy_gradient_scale = 1.0 / (1.0 - self.config.gamma)
        actor_loss = -(log_probability * aggregated_values.detach() * policy_gradient_scale).mean()
        actor_loss = actor_loss - self.config.entropy_coef * entropy.mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
        self.actor_optimizer.step()
        self._project_parameters(self.actor)

        return (
            float(actor_loss.detach()),
            float(entropy.detach().mean()),
            sum(displacements) / max(len(displacements), 1),
        )

    @torch.no_grad()
    def _project_parameters(self, module: nn.Module) -> None:
        bound = self.config.parameter_bound
        for parameter in module.parameters():
            parameter.clamp_(-bound, bound)
