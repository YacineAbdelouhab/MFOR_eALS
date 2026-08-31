from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EALSConfig:
    rank: int = 16
    regularization: float = 0.01
    max_iterations: int = 10
    inner_sweeps: int = 1
    alpha: float = 0.5
    c0: float = 1.0
    positive_confidence_base: float = 1.0
    positive_confidence_scale: float = 2.0
    convergence_tolerance: float = 1e-4
    convergence_patience: int = 2
    seed: int = 42
    factor_scale: float = 0.01
    partitions: int = 0
    checkpoint_interval: int = 2
    max_factor_norm: float = 100.0
    initialisation: str = "aleatoire"

    def __post_init__(self) -> None:
        if self.rank <= 0 or self.max_iterations <= 0 or self.inner_sweeps <= 0:
            raise ValueError("rank and iteration counts must be positive")
        if self.regularization <= 0 or self.c0 <= 0:
            raise ValueError("regularization and c0 must be positive")
        if self.alpha < 0 or self.positive_confidence_scale < 0:
            raise ValueError("alpha and confidence scale must be non-negative")
        if self.positive_confidence_base <= 0:
            raise ValueError("positive confidence base must be positive")
        if not 0 <= self.convergence_tolerance < 1:
            raise ValueError("convergence_tolerance must be in [0, 1)")
        if self.factor_scale <= 0 or self.max_factor_norm <= 0:
            raise ValueError("factor scales must be positive")
        if self.partitions < 0 or self.checkpoint_interval < 0:
            raise ValueError("partition/checkpoint settings must be non-negative")
        if self.convergence_patience <= 0:
            raise ValueError("convergence_patience must be positive")
        if self.initialisation not in ("aleatoire", "svd"):
            raise ValueError("initialisation must be 'aleatoire' or 'svd'")

    @classmethod
    def from_json(cls, path: str | Path) -> "EALSConfig":
        with Path(path).open("r", encoding="utf-8") as stream:
            return cls(**json.load(stream))

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "EALSConfig":
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
