#!/usr/bin/env python3
"""Train SCCPG on the built-in CACC interface or official VMAS scenarios."""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch

from SCCPG.algorithm import SCCPG
from SCCPG.buffer import RolloutBuffer
from SCCPG.config import SCCPGConfig
from SCCPG.envs import make_environment
from SCCPG.topology import CommunicationGraph


ENVIRONMENT_CHOICES = (
    "cacc-catchup",
    "cacc-slowdown",
    "vmas-dropout",
    "vmas-dispersion",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Self-Centered Clipped Critic Policy Gradient")
    parser.add_argument("--env", choices=ENVIRONMENT_CHOICES, default="cacc-catchup")
    parser.add_argument("--steps", type=int, default=1_000_000, help="Total vector-environment transitions")
    parser.add_argument("--num-agents", type=int, default=5)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--neighborhood-size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", help="cpu, cuda, mps, or auto")
    parser.add_argument("--actor-lr", type=float, default=5e-4)
    parser.add_argument("--critic-lr", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--actor-batch-size", type=int, default=60)
    parser.add_argument("--critic-batch-size", type=int, default=60)
    parser.add_argument("--critic-epochs", type=int, default=4)
    parser.add_argument("--entropy-coef", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=3.0)
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--parameter-bound", type=float, default=10.0)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.num_agents < 2:
        parser.error("--num-agents must be at least 2")
    if args.num_envs is not None and args.num_envs <= 0:
        parser.error("--num-envs must be positive")
    if args.horizon <= 0:
        parser.error("--horizon must be positive")
    return args


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_algorithm(args: argparse.Namespace, environment, device: torch.device) -> SCCPG:
    graph = (
        CommunicationGraph.line(environment.num_agents, args.neighborhood_size)
        if args.env.startswith("cacc-")
        else CommunicationGraph.ring(environment.num_agents, args.neighborhood_size)
    )
    config = SCCPGConfig(
        gamma=args.gamma,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        clipping_radius=args.radius,
        actor_batch_size=args.actor_batch_size,
        critic_batch_size=args.critic_batch_size,
        critic_epochs=args.critic_epochs,
        parameter_bound=args.parameter_bound,
    )
    return SCCPG(
        num_agents=environment.num_agents,
        observation_dim=environment.observation_dim,
        action_dim=environment.action_dim,
        action_low=environment.action_low,
        action_high=environment.action_high,
        graph=graph,
        config=config,
        device=device,
    )


def train(args: argparse.Namespace) -> SCCPG:
    seed_everything(args.seed)
    device = resolve_device(args.device)
    num_envs = args.num_envs
    if num_envs is None:
        num_envs = 1 if args.env.startswith("cacc-") else 32
    environment = make_environment(
        args.env,
        num_agents=args.num_agents,
        num_envs=num_envs,
        horizon=args.horizon,
        device=device,
        seed=args.seed,
    )
    algorithm = build_algorithm(args, environment, device)

    observations = environment.reset()
    running_returns = torch.zeros(
        environment.num_envs,
        environment.num_agents,
        dtype=torch.float32,
        device=device,
    )
    recent_episode_returns: list[float] = []
    total_steps = 0
    update_index = 0

    print(
        f"env={args.env} device={device} agents={environment.num_agents} "
        f"parallel_envs={environment.num_envs} radius={args.radius:g}"
    )

    while total_steps < args.steps:
        remaining = args.steps - total_steps
        rollout_steps = min(args.horizon, max(1, math.ceil(remaining / environment.num_envs)))
        buffer = RolloutBuffer()

        for _ in range(rollout_steps):
            actions, _ = algorithm.act(observations)
            next_observations, rewards, dones, _ = environment.step(actions)
            running_returns += rewards

            reset_all = bool(dones.any().item())
            stored_dones = torch.ones_like(dones) if reset_all else dones
            buffer.add(observations, actions, rewards, stored_dones)
            total_steps += environment.num_envs

            if reset_all:
                recent_episode_returns.extend(
                    running_returns.mean(dim=1).detach().cpu().tolist()
                )
                recent_episode_returns = recent_episode_returns[-100:]
                running_returns.zero_()
                observations = environment.reset()
            else:
                observations = next_observations

            if total_steps >= args.steps:
                break

        metrics = algorithm.update(buffer.build(algorithm.config.gamma))
        update_index += 1
        if update_index % args.log_interval == 0:
            if recent_episode_returns:
                mean_return = float(np.mean(recent_episode_returns))
            else:
                mean_return = float(running_returns.mean().detach().cpu())
            print(
                f"steps={min(total_steps, args.steps):>8d} "
                f"return={mean_return:>9.4f} "
                f"critic_loss={metrics['critic_loss']:>10.5f} "
                f"actor_loss={metrics['actor_loss']:>10.5f} "
                f"entropy={metrics['entropy']:>8.4f} "
                f"accepted_disp={metrics['accepted_displacement']:>7.4f}"
            )

    if args.checkpoint is not None:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        payload = algorithm.checkpoint()
        payload["environment"] = args.env
        payload["training_steps"] = total_steps
        payload["seed"] = args.seed
        torch.save(payload, args.checkpoint)
        print(f"checkpoint={args.checkpoint}")

    return algorithm


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
