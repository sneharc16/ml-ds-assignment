"""Fail closed when any configured offline model-quality gate is missed."""
import logging

from driveintent.config import load_config
from driveintent.monitoring.monitor import evaluate_quality_gates

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

if __name__ == "__main__":
    report = evaluate_quality_gates(load_config())
    for check in report["checks"]:
        logging.info("%s value=%s %s %s passed=%s", check["gate"], check["value"],
                     check["operator"], check["threshold"], check["passed"])
    if report["status"] != "pass":
        raise SystemExit("model quality gate failed")
    logging.info("all %d model-quality gates passed", report["total"])
