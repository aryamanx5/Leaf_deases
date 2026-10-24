"""Load configs/default.yaml and resolve paths relative to the repo root."""

from __future__ import annotations

import random
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"


@lru_cache(maxsize=4)
def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def path(key: str, cfg: dict | None = None) -> Path:
    """Absolute path for an entry under `paths:` in the config."""
    cfg = cfg or load_config()
    return ROOT / cfg["paths"][key]


def seed_everything(seed: int | None = None) -> int:
    seed = load_config()["seed"] if seed is None else seed
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass
    return seed
