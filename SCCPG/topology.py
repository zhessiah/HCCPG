"""Fixed sparse communication graphs for SCCPG."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple


@dataclass(frozen=True, slots=True)
class CommunicationGraph:
    """Receiver-specific aggregation neighborhoods and normalized weights."""

    num_agents: int
    neighborhoods: Tuple[Tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if self.num_agents <= 0:
            raise ValueError("num_agents must be positive")
        if len(self.neighborhoods) != self.num_agents:
            raise ValueError("one neighborhood is required per receiver")
        for receiver, neighbors in enumerate(self.neighborhoods):
            if not neighbors:
                raise ValueError(f"receiver {receiver} has an empty neighborhood")
            if receiver not in neighbors:
                raise ValueError(f"receiver {receiver} must include itself")
            if len(set(neighbors)) != len(neighbors):
                raise ValueError(f"receiver {receiver} has duplicate neighbors")
            if any(neighbor < 0 or neighbor >= self.num_agents for neighbor in neighbors):
                raise ValueError(f"receiver {receiver} has an out-of-range neighbor")

    @classmethod
    def line(cls, num_agents: int, neighborhood_size: int = 3) -> "CommunicationGraph":
        """Construct nearest-neighbor aggregation on a line."""

        return cls._from_distance(
            num_agents=num_agents,
            neighborhood_size=neighborhood_size,
            distance=lambda i, j: abs(i - j),
        )

    @classmethod
    def ring(cls, num_agents: int, neighborhood_size: int = 3) -> "CommunicationGraph":
        """Construct nearest-neighbor aggregation on a ring."""

        return cls._from_distance(
            num_agents=num_agents,
            neighborhood_size=neighborhood_size,
            distance=lambda i, j: min(abs(i - j), num_agents - abs(i - j)),
        )

    @classmethod
    def _from_distance(
        cls,
        *,
        num_agents: int,
        neighborhood_size: int,
        distance,
    ) -> "CommunicationGraph":
        if num_agents <= 0:
            raise ValueError("num_agents must be positive")
        if neighborhood_size <= 0:
            raise ValueError("neighborhood_size must be positive")
        size = min(num_agents, neighborhood_size)
        neighborhoods = []
        for receiver in range(num_agents):
            ordered = sorted(range(num_agents), key=lambda sender: (distance(receiver, sender), sender))
            neighborhoods.append(tuple(ordered[:size]))
        return cls(num_agents=num_agents, neighborhoods=tuple(neighborhoods))

    def neighbors(self, receiver: int) -> Tuple[int, ...]:
        self._validate_receiver(receiver)
        return self.neighborhoods[receiver]

    def weights_for(self, receiver: int) -> Dict[int, float]:
        """Return uniform nonnegative weights that sum to one."""

        neighbors = self.neighbors(receiver)
        weight = 1.0 / len(neighbors)
        return {sender: weight for sender in neighbors}

    def _validate_receiver(self, receiver: int) -> None:
        if receiver < 0 or receiver >= self.num_agents:
            raise IndexError(f"receiver index {receiver} is outside [0, {self.num_agents})")

    def directed_edges(self) -> Iterable[tuple[int, int]]:
        for receiver, neighbors in enumerate(self.neighborhoods):
            for sender in neighbors:
                if sender != receiver:
                    yield sender, receiver
