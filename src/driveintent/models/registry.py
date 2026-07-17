"""Lightweight local model registry: artifacts + metadata under artifacts/."""
from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib

from driveintent.config import Config


class ModelArtifactNotFoundError(Exception):
    pass


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                              text=True, timeout=5).stdout.strip() or None
    except Exception:
        return None


def save_model(cfg: Config, category: str, name: str, model: Any,
               features: list[str], cat_features: list[str],
               metrics: dict[str, Any], extra: dict[str, Any] | None = None,
               version: str = "v1") -> Path:
    import catboost
    import sklearn
    folder = cfg.artifacts / category / f"{name}_{version}"
    folder.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, folder / "model.joblib")
    meta = dict(
        model_name=name, version=f"{name}_{version}",
        algorithm=type(model).__name__,
        trained_at=datetime.now().isoformat(timespec="seconds"),
        train_end=cfg.get("splits", "train_end"),
        validation_end=cfg.get("splits", "validation_end"),
        test_end=cfg.get("splits", "test_end"),
        features=features, cat_features=cat_features,
        metrics=metrics, random_seed=cfg.seed,
        library_versions=dict(catboost=catboost.__version__, sklearn=sklearn.__version__),
        git_commit=_git_commit(cfg.root),
        **(extra or {}),
    )
    (folder / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    return folder


def load_model(cfg: Config, category: str, name: str, version: str = "v1") -> tuple[Any, dict]:
    folder = cfg.artifacts / category / f"{name}_{version}"
    if not (folder / "model.joblib").exists():
        raise ModelArtifactNotFoundError(
            f"{name} model artifact not found at {folder}. "
            f"Run `python scripts/train_all_models.py --model {name.split('_')[0]}`.")
    model = joblib.load(folder / "model.joblib")
    meta = json.loads((folder / "metadata.json").read_text())
    return model, meta
