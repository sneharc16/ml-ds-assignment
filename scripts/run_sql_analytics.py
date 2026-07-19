"""Execute the SQL portfolio, export results, and fail on warehouse violations."""
from __future__ import annotations

import logging
import time

import pandas as pd

from driveintent.config import load_config
from driveintent.data import load_database as db

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    cfg = load_config()
    output = cfg.artifacts / "reports" / "sql"
    output.mkdir(parents=True, exist_ok=True)
    catalog = []
    for path in db.all_analytics_files(cfg):
        started = time.perf_counter()
        result = db.run_sql_file(cfg, path.name)
        elapsed_ms = (time.perf_counter() - started) * 1000
        result.to_csv(output / f"{path.stem}.csv", index=False)
        catalog.append({"query": path.stem, "rows": len(result),
                        "columns": len(result.columns), "runtime_ms": round(elapsed_ms, 2)})
        logging.info("%-32s rows=%d runtime_ms=%.2f", path.name, len(result), elapsed_ms)
    quality_frames = [db.run_quality_file(cfg, path.name) for path in db.all_quality_files(cfg)]
    quality = pd.concat(quality_frames, ignore_index=True) if quality_frames else pd.DataFrame()
    quality.to_csv(output / "data_quality_checks.csv", index=False)
    pd.DataFrame(catalog).to_csv(output / "query_catalog.csv", index=False)
    violations = int(quality["violations"].sum()) if len(quality) else 0
    if violations:
        raise SystemExit(f"SQL data-quality checks found {violations} violations")
    logging.info("SQL portfolio complete: %d queries; data-quality checks passed", len(catalog))


if __name__ == "__main__":
    main()
