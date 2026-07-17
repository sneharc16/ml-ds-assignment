"""Create the DuckDB database from raw parquet."""
import logging

from driveintent.config import load_config
from driveintent.data import load_database as db

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

if __name__ == "__main__":
    cfg = load_config()
    db.initialize(cfg)
    logging.info("database initialized at %s", cfg.database)
    for f in db.all_analytics_files(cfg):
        n = len(db.run_sql_file(cfg, f.name))
        logging.info("verified %s -> %d rows", f.name, n)
