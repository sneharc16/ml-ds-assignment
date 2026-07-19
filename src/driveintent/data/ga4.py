"""Adapter for flattened GA4/BigQuery exports.

GA4 exports are intentionally mapped at the boundary so the feature and model
layers only depend on DriveIntent's canonical event contract. The adapter does
not call Google APIs; production extraction can write flattened Parquet/CSV and
invoke this deterministic transformation.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

CANONICAL_EVENTS = {
    "session_start", "view_home", "search", "apply_filter", "remove_filter",
    "view_search_results", "select_item", "view_item", "view_gallery",
    "view_inspection_report", "compare_car", "calculate_emi", "view_finance_offer",
    "add_to_wishlist", "remove_from_wishlist", "request_callback", "book_test_drive",
    "begin_checkout", "booking_complete", "purchase", "session_end",
}

GA4_EVENT_MAP = {
    "page_view": "view_home",
    "view_search_results": "view_search_results",
    "search": "search",
    "select_item": "select_item",
    "view_item": "view_item",
    "view_item_list": "view_search_results",
    "add_to_wishlist": "add_to_wishlist",
    "begin_checkout": "begin_checkout",
    "purchase": "purchase",
    "generate_lead": "request_callback",
    "session_start": "session_start",
}

REQUIRED = {"event_timestamp", "event_name", "user_pseudo_id", "ga_session_id"}


class GA4ContractError(ValueError):
    """Raised when a flattened export cannot satisfy the canonical contract."""


def _stable_id(prefix: str, *parts: object) -> str:
    raw = "|".join(str(part) for part in parts).encode()
    return f"{prefix}_{hashlib.sha256(raw).hexdigest()[:20]}"


def _read(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise GA4ContractError("GA4 input must be .csv or .parquet")


def transform_flattened_ga4(frame: pd.DataFrame, unknown_events: str = "drop") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return canonical ``events`` and ``sessions`` from a flattened GA4 frame.

    Expected columns mirror a common BigQuery flattening query. Optional fields
    are filled safely, while identity and time fields are strict.
    """
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise GA4ContractError(f"missing required GA4 fields: {sorted(missing)}")
    if unknown_events not in {"drop", "error"}:
        raise GA4ContractError("unknown_events must be 'drop' or 'error'")

    source = frame.copy()
    source["event_timestamp"] = pd.to_datetime(source["event_timestamp"], utc=True, errors="coerce").dt.tz_localize(None)
    if source["event_timestamp"].isna().any():
        raise GA4ContractError("event_timestamp contains invalid values")
    source["canonical_event"] = source["event_name"].map(GA4_EVENT_MAP).fillna(
        source["event_name"].where(source["event_name"].isin(CANONICAL_EVENTS))
    )
    unknown = source["canonical_event"].isna()
    if unknown.any() and unknown_events == "error":
        names = sorted(source.loc[unknown, "event_name"].astype(str).unique())
        raise GA4ContractError(f"unmapped GA4 events: {names}")
    source = source.loc[~unknown].copy()
    if source.empty:
        raise GA4ContractError("no canonical events remained after mapping")

    def optional(name: str, default=None):
        return source[name] if name in source else pd.Series(default, index=source.index)

    source["user_id"] = source["user_pseudo_id"].astype(str).map(lambda x: _stable_id("GA4U", x))
    source["session_id"] = [
        _stable_id("GA4S", user, session)
        for user, session in zip(source["user_pseudo_id"], source["ga_session_id"])
    ]
    original_event_id = optional("event_id")
    source["event_id"] = [
        str(eid) if pd.notna(eid) and str(eid) else _stable_id("GA4E", sid, ts, name, ordinal)
        for ordinal, (eid, sid, ts, name) in enumerate(zip(
            original_event_id, source["session_id"], source["event_timestamp"], source["canonical_event"]
        ))
    ]

    events = pd.DataFrame({
        "event_id": source["event_id"],
        "event_timestamp": source["event_timestamp"],
        "event_date": source["event_timestamp"].dt.date,
        "user_id": source["user_id"],
        "session_id": source["session_id"],
        "event_name": source["canonical_event"],
        "car_id": optional("item_id"),
        "item_list_id": optional("item_list_id"),
        "list_position": pd.to_numeric(optional("item_list_index"), errors="coerce").astype("Int64"),
        "search_term": optional("search_term"),
        "filter_name": optional("filter_name"),
        "filter_value": optional("filter_value"),
        "engagement_time_seconds": pd.to_numeric(optional("engagement_time_msec", 0), errors="coerce").fillna(0) / 1000,
        "page_location": optional("page_location"),
        "source": optional("traffic_source", "(direct)").fillna("(direct)"),
        "medium": optional("traffic_medium", "(none)").fillna("(none)"),
        "campaign_id": optional("campaign_id", "GA4_UNATTRIBUTED").fillna("GA4_UNATTRIBUTED"),
        "device_category": optional("device_category", "unknown").fillna("unknown"),
        "city": optional("geo_city", "unknown").fillna("unknown"),
    }).sort_values(["session_id", "event_timestamp", "event_id"]).reset_index(drop=True)
    if events["event_id"].duplicated().any():
        raise GA4ContractError("event_id is not unique")

    sessions = (events.groupby("session_id", as_index=False)
                .agg(user_id=("user_id", "first"), session_start=("event_timestamp", "min"),
                     session_end=("event_timestamp", "max"), campaign_id=("campaign_id", "first"),
                     source=("source", "first"), medium=("medium", "first"),
                     device_category=("device_category", "first"), city=("city", "first")))
    sessions = sessions.sort_values(["user_id", "session_start", "session_id"])
    sessions["session_sequence_number"] = sessions.groupby("user_id").cumcount() + 1
    sessions["is_returning_user"] = sessions["session_sequence_number"] > 1
    sessions = sessions[["session_id", "user_id", "session_start", "session_end", "campaign_id",
                         "source", "medium", "device_category", "city", "is_returning_user",
                         "session_sequence_number"]].reset_index(drop=True)
    return events, sessions


def import_flattened_ga4(input_path: Path, output_dir: Path, unknown_events: str = "drop") -> tuple[Path, Path]:
    events, sessions = transform_flattened_ga4(_read(input_path), unknown_events=unknown_events)
    output_dir.mkdir(parents=True, exist_ok=True)
    event_path, session_path = output_dir / "events.parquet", output_dir / "sessions.parquet"
    events.to_parquet(event_path, index=False)
    sessions.to_parquet(session_path, index=False)
    return event_path, session_path
