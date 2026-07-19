"""Transform a flattened GA4 BigQuery export into DriveIntent event tables."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from driveintent.data.ga4 import import_flattened_ga4

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="Flattened GA4 .csv or .parquet")
    parser.add_argument("--output-dir", type=Path, default=Path("data/ga4_canonical"))
    parser.add_argument("--unknown-events", choices=["drop", "error"], default="drop")
    args = parser.parse_args()
    events, sessions = import_flattened_ga4(args.input, args.output_dir, args.unknown_events)
    logging.info("wrote %s and %s", events, sessions)


if __name__ == "__main__":
    main()
