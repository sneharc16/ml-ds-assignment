"""Generate the synthetic marketplace dataset."""
import argparse
import logging

from driveintent.config import load_config
from driveintent.data.generate import build_all
from driveintent.data.validate import validate_all

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--small", action="store_true", help="use test-scale profile")
    args = ap.parse_args()
    cfg = load_config(args.config)
    tables = build_all(cfg, small=args.small)
    for k, v in tables.items():
        logging.info("generated %s: %s rows", k, len(v))
    validate_all(tables)
    logging.info("validation passed; raw data written to %s", cfg.raw_data)
