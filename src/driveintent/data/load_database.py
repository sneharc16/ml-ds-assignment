"""Load raw parquet tables into DuckDB and register analytics SQL."""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from driveintent.config import Config

TABLES = ["cars", "users", "campaigns", "sessions", "events", "impressions"]


def connect(cfg: Config, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    cfg.database.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(cfg.database), read_only=read_only)


def initialize(cfg: Config) -> None:
    if cfg.database.exists():
        cfg.database.unlink()
    con = connect(cfg)
    ddl = (cfg.root / "sql" / "ddl" / "create_tables.sql").read_text()
    con.execute(ddl)
    for t in TABLES:
        pq = cfg.raw_data / f"{t}.parquet"
        con.execute(f"DELETE FROM {t}")
        con.execute(f"INSERT INTO {t} SELECT * FROM read_parquet('{pq.as_posix()}')")
    for mart in sorted((cfg.root / "sql" / "marts").glob("*.sql")):
        con.execute(mart.read_text())
    con.close()


def run_sql_file(cfg: Config, name: str) -> pd.DataFrame:
    """Execute an analytics SQL file (relative to sql/analytics) and return a DataFrame."""
    path = cfg.root / "sql" / "analytics" / name
    con = connect(cfg, read_only=True)
    try:
        return con.execute(path.read_text()).df()
    finally:
        con.close()


def run_quality_file(cfg: Config, name: str) -> pd.DataFrame:
    path = cfg.root / "sql" / "quality" / name
    con = connect(cfg, read_only=True)
    try:
        return con.execute(path.read_text()).df()
    finally:
        con.close()


def query(cfg: Config, sql: str) -> pd.DataFrame:
    con = connect(cfg, read_only=True)
    try:
        return con.execute(sql).df()
    finally:
        con.close()


def all_analytics_files(cfg: Config) -> list[Path]:
    return sorted((cfg.root / "sql" / "analytics").glob("*.sql"))


def all_quality_files(cfg: Config) -> list[Path]:
    return sorted((cfg.root / "sql" / "quality").glob("*.sql"))
