"""Build the processed feature datasets."""
import logging

from driveintent.config import load_config
from driveintent.features.build import build_all

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

if __name__ == "__main__":
    cfg = load_config()
    ds = build_all(cfg)
    for k, v in ds.items():
        logging.info("feature dataset %s: %s", k, v.shape)
