"""Cached data access for the dashboard."""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from driveintent.config import Config, load_config
from driveintent.data import load_database as db


@st.cache_resource
def load_cfg() -> Config:
    return load_config()


@st.cache_data
def raw_table(name: str) -> pd.DataFrame:
    cfg = load_config()
    return pd.read_parquet(cfg.raw_data / f"{name}.parquet")


@st.cache_data
def processed_table(name: str) -> pd.DataFrame:
    cfg = load_config()
    return pd.read_parquet(cfg.processed_data / f"{name}.parquet")


@st.cache_data
def sql(name: str) -> pd.DataFrame:
    return db.run_sql_file(load_config(), name)


@st.cache_data
def metrics_json(name: str) -> dict:
    cfg = load_config()
    p = cfg.artifacts / "metrics" / name
    return json.loads(p.read_text()) if p.exists() else {}


@st.cache_data
def metrics_csv(name: str) -> pd.DataFrame:
    cfg = load_config()
    p = cfg.artifacts / "metrics" / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def table_counts(cfg: Config) -> dict:
    return {t: len(raw_table(t)) for t in ("cars", "users", "sessions", "events")}
