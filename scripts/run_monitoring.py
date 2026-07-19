"""Generate feature-drift and model-quality monitoring artifacts."""
import json
import logging

from driveintent.config import load_config
from driveintent.monitoring.monitor import run_monitoring

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

if __name__ == "__main__":
    cfg = load_config()
    cfg.ensure_dirs()
    report = run_monitoring(cfg)
    logging.info("monitoring status:\n%s", json.dumps(report, indent=2))
