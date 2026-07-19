"""Lightweight local model registry: artifacts + metadata under artifacts/."""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib

from driveintent.config import Config


class ModelArtifactNotFoundError(Exception):
    pass


class ModelArtifactIntegrityError(Exception):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def feature_contract_sha256(features: list[str], cat_features: list[str]) -> str:
    payload = json.dumps({"features": features, "cat_features": cat_features},
                         sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _upsert_manifest(cfg: Config, key: str, entry: dict[str, Any]) -> None:
    path = cfg.artifacts / "model_manifest.json"
    manifest = {"schema_version": 1, "artifacts": {}}
    if path.exists():
        manifest = json.loads(path.read_text())
        manifest.setdefault("artifacts", {})
    manifest["artifacts"][key] = entry
    path.write_text(json.dumps(manifest, indent=2, default=str))


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
    model_path = folder / "model.joblib"
    joblib.dump(model, model_path)
    artifact_sha = sha256_file(model_path)
    contract_sha = feature_contract_sha256(features, cat_features)
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
        artifact_path=str(model_path.relative_to(cfg.root)),
        artifact_sha256=artifact_sha,
        feature_contract_sha256=contract_sha,
        **(extra or {}),
    )
    metadata_path = folder / "metadata.json"
    metadata_path.write_text(json.dumps(meta, indent=2, default=str))
    _upsert_manifest(cfg, f"{category}/{name}", {
        "version": meta["version"], "artifact_path": meta["artifact_path"],
        "metadata_path": str(metadata_path.relative_to(cfg.root)),
        "artifact_sha256": artifact_sha, "feature_contract_sha256": contract_sha,
        "trained_at": meta["trained_at"],
    })
    return folder


def load_model(cfg: Config, category: str, name: str, version: str = "v1") -> tuple[Any, dict]:
    folder = cfg.artifacts / category / f"{name}_{version}"
    model_path, metadata_path = folder / "model.joblib", folder / "metadata.json"
    if not model_path.exists() or not metadata_path.exists():
        raise ModelArtifactNotFoundError(
            f"{name} model artifact not found at {folder}. "
            f"Run `python scripts/train_all_models.py --model {name.split('_')[0]}`.")
    meta = json.loads(metadata_path.read_text())
    actual_artifact = sha256_file(model_path)
    actual_contract = feature_contract_sha256(list(meta.get("features", [])),
                                               list(meta.get("cat_features", [])))
    if meta.get("artifact_sha256") != actual_artifact:
        raise ModelArtifactIntegrityError(f"Checksum verification failed for {name} at {model_path}.")
    if meta.get("feature_contract_sha256") != actual_contract:
        raise ModelArtifactIntegrityError(f"Feature contract verification failed for {name}.")
    model = joblib.load(model_path)
    return model, meta


def register_external_artifact(cfg: Config, key: str, artifact_path: Path,
                               extra: dict[str, Any] | None = None) -> Path:
    checksum = sha256_file(artifact_path)
    metadata_path = artifact_path.with_name(f"{artifact_path.stem}.metadata.json")
    metadata = {"artifact_path": str(artifact_path.relative_to(cfg.root)),
                "artifact_sha256": checksum,
                "created_at": datetime.now().isoformat(timespec="seconds"), **(extra or {})}
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str))
    _upsert_manifest(cfg, key, {**metadata, "metadata_path": str(metadata_path.relative_to(cfg.root))})
    return metadata_path


def verify_external_artifact(artifact_path: Path, metadata_path: Path) -> dict[str, Any]:
    if not artifact_path.exists() or not metadata_path.exists():
        raise ModelArtifactNotFoundError(f"Artifact or metadata missing at {artifact_path}")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("artifact_sha256") != sha256_file(artifact_path):
        raise ModelArtifactIntegrityError(f"Checksum verification failed for {artifact_path}.")
    return metadata
