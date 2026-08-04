"""Self-Centered Clipped Critic Policy Gradient."""

from .algorithm import SCCPG
from .config import CACCConfig, SCCPGConfig, TrainConfig, VMASConfig
from .envs import CACCEnv, MultiAgentEnv, VMASEnv, make_environment

__all__ = [
    "CACCConfig",
    "CACCEnv",
    "MultiAgentEnv",
    "SCCPG",
    "SCCPGConfig",
    "TrainConfig",
    "VMASConfig",
    "VMASEnv",
    "make_environment",
]

__version__ = "0.1.0"
