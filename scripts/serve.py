"""Production-style process entrypoint with optional deterministic demo bootstrap."""
from __future__ import annotations

import argparse
import fcntl
import logging
import os
import subprocess
import sys

from driveintent.config import load_config
from driveintent.models import registry

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def missing_requirements() -> list[str]:
    cfg = load_config()
    missing = []
    if not cfg.database.exists():
        missing.append("database")
    for category, name in (("regression", "price"), ("classification", "booking"),
                           ("classification", "sellthrough"), ("ranking", "ranker")):
        try:
            registry.load_model(cfg, category, name)
        except (registry.ModelArtifactNotFoundError, registry.ModelArtifactIntegrityError):
            missing.append(f"{category}/{name}")
    try:
        from driveintent.models.recommender import RecommenderBundle
        RecommenderBundle.load(cfg)
    except (registry.ModelArtifactNotFoundError, registry.ModelArtifactIntegrityError):
        missing.append("recommender/bundle")
    return missing


def ensure_ready() -> None:
    cfg = load_config()
    cfg.ensure_dirs()
    missing = missing_requirements()
    if not missing:
        return
    if os.environ.get("DRIVEINTENT_BOOTSTRAP_DEMO") != "1":
        raise SystemExit(
            f"required runtime assets are unavailable: {', '.join(missing)}. "
            "Run `python scripts/run_pipeline.py --small`, or explicitly set "
            "DRIVEINTENT_BOOTSTRAP_DEMO=1 for a synthetic demo deployment."
        )
    lock_path = cfg.artifacts / ".bootstrap.lock"
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        missing = missing_requirements()
        if missing:
            logging.info("bootstrapping deterministic demo assets: %s", ", ".join(missing))
            subprocess.run([sys.executable, str(cfg.root / "scripts" / "run_pipeline.py"), "--small"],
                           cwd=cfg.root, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=["api", "dashboard"])
    args = parser.parse_args()
    ensure_ready()
    if args.role == "api":
        command = [sys.executable, "-m", "uvicorn", "driveintent.api.main:app", "--host", "0.0.0.0",
                   "--port", os.environ.get("PORT", "8000")]
    else:
        command = [sys.executable, "-m", "streamlit", "run", "src/driveintent/dashboard/app.py",
                   "--server.address", "0.0.0.0", "--server.port", os.environ.get("PORT", "8501"),
                   "--server.headless", "true"]
    os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
