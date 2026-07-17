"""Central configuration loading and path management for DriveIntent."""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

RANDOM_SEED = 42


def repo_root() -> Path:
    """Repository root: env override, else walk up from this file."""
    env = os.environ.get("DRIVEINTENT_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "configs" / "config.yaml").exists():
            return parent
    # fallback: two levels above src/driveintent
    return here.parents[2]


@dataclass
class Config:
    raw: dict[str, Any] = field(default_factory=dict)
    root: Path = field(default_factory=repo_root)

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, *keys: str, default: Any = None) -> Any:
        node: Any = self.raw
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node

    # ---- resolved paths -------------------------------------------------
    def path(self, key: str) -> Path:
        p = self.root / self.raw["paths"][key]
        return p

    @property
    def raw_data(self) -> Path:
        return self.path("raw_data")

    @property
    def processed_data(self) -> Path:
        return self.path("processed_data")

    @property
    def database(self) -> Path:
        return self.path("database")

    @property
    def artifacts(self) -> Path:
        return self.path("artifacts")

    @property
    def seed(self) -> int:
        return int(self.get("project", "random_seed", default=RANDOM_SEED))

    def data_gen(self, small: bool = False) -> dict[str, Any]:
        key = "test_data_generation" if small else "data_generation"
        return dict(self.raw[key])

    def ensure_dirs(self) -> None:
        for key in ("raw_data", "interim_data", "processed_data"):
            self.path(key).mkdir(parents=True, exist_ok=True)
        self.database.parent.mkdir(parents=True, exist_ok=True)
        for sub in ("regression", "classification", "recommender", "ranking",
                    "encoders", "metrics", "reports"):
            (self.artifacts / sub).mkdir(parents=True, exist_ok=True)


def load_config(path: str | Path | None = None) -> Config:
    root = repo_root()
    cfg_path = Path(path) if path else root / "configs" / "config.yaml"
    with open(cfg_path) as fh:
        raw = yaml.safe_load(fh)
    return Config(raw=raw, root=root)


def set_seed(seed: int = RANDOM_SEED) -> np.random.Generator:
    """Seed python/numpy and return a numpy Generator for local use."""
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)
