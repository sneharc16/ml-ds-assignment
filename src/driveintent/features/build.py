"""Feature engineering: builds the processed model datasets.

Temporal hygiene rules:
- price dataset: only attributes known at listing time; target = transaction_price of sold cars
- booking dataset: one row per impression; user/session history features computed
  strictly from data BEFORE the impression's session
- sellthrough dataset: inventory snapshots at ages {0,7,14,30,45}; engagement
  counters use only events up to the snapshot date; label = sale within 30 days
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from driveintent.config import Config

PRICE_FEATURES = [
    "make", "model", "variant", "body_type", "fuel_type", "transmission",
    "manufacturing_year", "registration_year", "vehicle_age", "kilometres_driven",
    "kilometres_per_year", "owner_count", "city", "state", "engine_cc",
    "claimed_mileage_kmpl", "insurance_valid", "service_history_available",
    "accident_history", "inspection_score", "exterior_score", "interior_score",
    "engine_score", "tyre_score", "number_of_features", "inventory_entry_month",
    "market_demand_index", "local_supply_index", "model_popularity",
]
PRICE_CAT = ["make", "model", "variant", "body_type", "fuel_type", "transmission", "city", "state"]

BOOKING_CAT = ["body_type", "fuel_type", "transmission", "make", "device_category",
               "source", "medium", "campaign_id", "city"]
SELL_CAT = ["make", "model", "body_type", "fuel_type", "transmission", "city"]


def car_base_features(cars: pd.DataFrame) -> pd.DataFrame:
    df = cars.copy()
    entry = pd.to_datetime(df["inventory_entry_date"])
    df["vehicle_age"] = (entry.dt.year - df["manufacturing_year"]).clip(lower=0)
    df["kilometres_per_year"] = df["kilometres_driven"] / df["vehicle_age"].clip(lower=1)
    df["inventory_entry_month"] = entry.dt.month

    # demand index: engaged sessions per (city, body); supply index: live inventory share
    seg = df.groupby(["city", "body_type"]).size().rename("segment_supply").reset_index()
    city_tot = seg.groupby("city")["segment_supply"].transform("sum")
    seg["local_supply_index"] = seg["segment_supply"] / city_tot
    df = df.merge(seg[["city", "body_type", "local_supply_index"]], on=["city", "body_type"], how="left")
    return df


def attach_market_context(rows: pd.DataFrame, cars: pd.DataFrame,
                          events: pd.DataFrame, observation_col: str) -> pd.DataFrame:
    """Attach demand and live-supply features known at each row's observation time."""
    out = rows.copy()
    observed = pd.to_datetime(out[observation_col]).to_numpy(dtype="datetime64[ns]")
    car_segments = cars[["car_id", "city", "body_type"]]
    views = (events.loc[events["event_name"] == "view_item", ["car_id", "event_timestamp"]]
             .dropna(subset=["car_id"])
             .merge(car_segments, on="car_id", how="inner"))
    views["event_timestamp"] = pd.to_datetime(views["event_timestamp"])
    global_view_times = np.sort(views["event_timestamp"].to_numpy(dtype="datetime64[ns]"))
    segment_count = max(len(cars[["city", "body_type"]].drop_duplicates()), 1)
    total_views = np.searchsorted(global_view_times, observed, side="right")

    demand_count = np.zeros(len(out), dtype=float)
    for key, idx in out.groupby(["city", "body_type"]).groups.items():
        times = np.sort(views.loc[
            (views["city"] == key[0]) & (views["body_type"] == key[1]),
            "event_timestamp",
        ].to_numpy(dtype="datetime64[ns]"))
        demand_count[np.asarray(list(idx))] = np.searchsorted(times, observed[list(idx)], side="right")
    average_demand = total_views / segment_count
    out["market_demand_index"] = np.divide(
        demand_count, average_demand, out=np.full(len(out), 0.5), where=average_demand > 0
    )

    inventory = cars[["city", "body_type", "inventory_entry_date", "inventory_exit_date"]].copy()
    inventory["inventory_entry_date"] = pd.to_datetime(inventory["inventory_entry_date"])
    inventory["inventory_exit_date"] = pd.to_datetime(inventory["inventory_exit_date"])

    def live_count(frame: pd.DataFrame, times: np.ndarray) -> np.ndarray:
        entries = np.sort(frame["inventory_entry_date"].to_numpy(dtype="datetime64[ns]"))
        exits = np.sort(frame["inventory_exit_date"].dropna().to_numpy(dtype="datetime64[ns]"))
        return (np.searchsorted(entries, times, side="right")
                - np.searchsorted(exits, times, side="left"))

    city_supply = np.zeros(len(out), dtype=float)
    for city, idx in out.groupby("city").groups.items():
        city_supply[np.asarray(list(idx))] = live_count(
            inventory[inventory["city"] == city], observed[list(idx)]
        )
    segment_supply = np.zeros(len(out), dtype=float)
    for key, idx in out.groupby(["city", "body_type"]).groups.items():
        segment_supply[np.asarray(list(idx))] = live_count(
            inventory[(inventory["city"] == key[0]) & (inventory["body_type"] == key[1])],
            observed[list(idx)],
        )
    out["local_supply_index"] = np.divide(
        segment_supply, city_supply, out=np.zeros(len(out)), where=city_supply > 0
    )
    return out


def attach_demand_index(cars_f: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Compatibility helper for ranking code; Phase 3 replaces its global profile."""
    ev = events[events["event_name"] == "view_item"].dropna(subset=["car_id"])
    ev = ev.merge(cars_f[["car_id", "body_type"]], on="car_id", how="left")
    dem = ev.groupby(["city", "body_type"]).size().rename("demand_events").reset_index()
    dem["market_demand_index"] = dem["demand_events"] / dem["demand_events"].mean()
    out = cars_f.merge(dem[["city", "body_type", "market_demand_index"]],
                       on=["city", "body_type"], how="left")
    out["market_demand_index"] = out["market_demand_index"].fillna(0.5)
    return out


# --------------------------------------------------------------------------
def build_price_dataset(cars: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    f = car_base_features(cars)
    f["observation_date"] = pd.to_datetime(f["inventory_entry_date"])
    f = f.drop(columns=["local_supply_index"])
    f = attach_market_context(f, cars, events, "observation_date")
    ds = f[f["sold_flag"]].copy()
    ds["target_price"] = ds["transaction_price"]
    return ds[["car_id", "observation_date", "target_price", "listed_price"] + PRICE_FEATURES]


# --------------------------------------------------------------------------
def build_booking_dataset(impressions: pd.DataFrame, sessions: pd.DataFrame,
                          users: pd.DataFrame, cars: pd.DataFrame,
                          events: pd.DataFrame) -> pd.DataFrame:
    cars_f = car_base_features(cars).drop(columns=["local_supply_index"])
    imp = impressions.merge(sessions, on=["session_id", "user_id"], how="left",
                            suffixes=("", "_s"))
    imp = imp.merge(users, on="user_id", how="left")
    imp = imp.merge(cars_f.drop(columns=["sold_flag", "transaction_price",
                                         "inventory_exit_date", "days_in_inventory",
                                         "acquisition_price"]),
                    on="car_id", how="left", suffixes=("", "_car"))
    imp["observation_date"] = pd.to_datetime(imp["event_timestamp"])
    imp = attach_market_context(imp, cars, events, "observation_date")

    # user history strictly before this session (leak-free)
    sess_book = (imp.groupby(["user_id", "session_sequence_number", "session_start"])["booked"]
                    .max().reset_index().sort_values(["user_id", "session_start"]))
    sess_book["prior_sessions"] = sess_book.groupby("user_id").cumcount()
    sess_book["prior_bookings"] = sess_book.groupby("user_id")["booked"].transform(
        lambda values: values.astype(int).shift(fill_value=0).cumsum()
    )
    imp = imp.merge(sess_book[["user_id", "session_sequence_number",
                               "prior_sessions", "prior_bookings"]],
                    on=["user_id", "session_sequence_number"], how="left")
    imp["prior_booking_rate"] = imp["prior_bookings"] / imp["prior_sessions"].clip(lower=1)

    # Session features stop at the impression decision time. Outcome/deep-
    # engagement events later in the session must never enter model features.
    decision_time = imp.groupby("session_id")["observation_date"].min().rename("decision_time")
    predecision = events.merge(decision_time, on="session_id", how="inner")
    predecision = predecision[pd.to_datetime(predecision["event_timestamp"]) <= predecision["decision_time"]]
    sess_stats = (predecision.groupby("session_id")
                  .agg(session_searches=("event_name", lambda s: (s == "search").sum()),
                       session_filters=("event_name", lambda s: (s == "apply_filter").sum()),
                       session_events=("event_id", "count")).reset_index())
    imp = imp.merge(sess_stats, on="session_id", how="left")
    imp["session_duration_s"] = (pd.to_datetime(imp["observation_date"])
                                 - pd.to_datetime(imp["session_start"])).dt.total_seconds()

    # user-car match features
    imp["price_budget_gap"] = (imp["listed_price"] - imp["maximum_budget"]) / imp["maximum_budget"]
    est_emi = imp["listed_price"] / 60.0
    imp["emi_budget_gap"] = (est_emi - imp["maximum_emi"]) / imp["maximum_emi"]
    imp["brand_match"] = (imp["make"] == imp["preferred_makes"]).astype(int)
    imp["body_match"] = (imp["body_type"] == imp["preferred_body_types"]).astype(int)
    imp["fuel_match"] = (imp["fuel_type"] == imp["preferred_fuel_types"]).astype(int)
    imp["trans_match"] = (imp["transmission"] == imp["preferred_transmissions"]).astype(int)
    imp["city_match"] = (imp["city"] == imp["home_city"]).astype(int)
    imp["age_over_tolerance"] = (imp["vehicle_age"] - imp["vehicle_age_tolerance"]).clip(lower=0)
    imp["km_over_tolerance"] = ((imp["kilometres_driven"] - imp["kilometre_tolerance"])
                                .clip(lower=0) / 10000)
    imp["hard_violations"] = ((imp["price_budget_gap"] > 0.10).astype(int)
                              + (imp["age_over_tolerance"] > 2).astype(int))

    imp["label_booked"] = imp["booked"].astype(int)
    return imp


BOOKING_FEATURES = [
    # user
    "prior_sessions", "prior_booking_rate", "is_returning_user",
    "purchase_urgency", "price_sensitivity", "quality_sensitivity",
    "finance_interest", "brand_loyalty", "first_time_buyer_probability",
    # session
    "session_duration_s", "session_searches", "session_filters",
    "session_events", "session_sequence_number", "device_category",
    "source", "medium", "campaign_id",
    # match
    "price_budget_gap", "emi_budget_gap", "brand_match", "body_match",
    "fuel_match", "trans_match", "city_match", "age_over_tolerance",
    "km_over_tolerance", "hard_violations",
    # car
    "listed_price", "inspection_score", "vehicle_age", "kilometres_driven",
    "market_demand_index", "local_supply_index", "finance_available",
    "delivery_available", "model_popularity", "body_type", "fuel_type",
    "transmission", "make", "city",
    # exposure
    "list_position",
]


# --------------------------------------------------------------------------
def build_sellthrough_dataset(cars: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    cars_f = car_base_features(cars).drop(columns=["local_supply_index"])
    ev = events.dropna(subset=["car_id"]).copy()
    ev["event_timestamp"] = pd.to_datetime(ev["event_timestamp"])

    snapshots = []
    entry = pd.to_datetime(cars_f["inventory_entry_date"])
    max_age = cars_f["days_in_inventory"]
    for snap_age in (0, 7, 14, 30, 45):
        mask = max_age >= snap_age  # car still in inventory at snapshot
        snap = cars_f[mask].copy()
        snap["snapshot_age"] = snap_age
        snap["snapshot_date"] = entry[mask] + pd.to_timedelta(snap_age, unit="D")
        # label: sale within 30 days of snapshot
        snap["label_sold_30d"] = (snap["sold_flag"]
                                  & (snap["days_in_inventory"] <= snap_age + 30)).astype(int)
        snapshots.append(snap)
    ds = pd.concat(snapshots, ignore_index=True)
    ds = attach_market_context(ds, cars, events, "snapshot_date")

    # engagement counters strictly before snapshot_date
    counts = []
    for name, col in [("view_item", "views"), ("select_item", "clicks"),
                      ("view_inspection_report", "inspection_views"),
                      ("add_to_wishlist", "wishlists"),
                      ("book_test_drive", "bookings_cnt")]:
        sub = ev[ev["event_name"] == name][["car_id", "event_timestamp"]]
        counts.append((col, sub))
    for col, sub in counts:
        merged = ds[["car_id", "snapshot_date"]].reset_index().merge(sub, on="car_id", how="left")
        n = (merged[merged["event_timestamp"] <= merged["snapshot_date"]]
             .groupby("index").size())
        ds[col] = n.reindex(ds.index).fillna(0).astype(int)
    ds["click_rate"] = ds["clicks"] / ds["views"].clip(lower=1)
    ds["booking_rate"] = ds["bookings_cnt"] / ds["clicks"].clip(lower=1)

    comp = ds.groupby(["city", "body_type", "snapshot_age"]).size().rename("comparable_inventory")
    ds = ds.merge(comp.reset_index(), on=["city", "body_type", "snapshot_age"], how="left")
    ds["observation_date"] = ds["snapshot_date"]
    return ds


SELL_FEATURES = [
    "snapshot_age", "listed_price", "vehicle_age", "kilometres_driven",
    "owner_count", "inspection_score", "number_of_features",
    "views", "clicks", "inspection_views", "wishlists", "bookings_cnt",
    "click_rate", "booking_rate", "market_demand_index", "local_supply_index",
    "comparable_inventory", "finance_available", "delivery_available",
    "model_popularity", "make", "model", "body_type", "fuel_type",
    "transmission", "city",
]


# --------------------------------------------------------------------------
def build_all(cfg: Config) -> dict[str, pd.DataFrame]:
    raw = cfg.raw_data
    cars = pd.read_parquet(raw / "cars.parquet")
    users = pd.read_parquet(raw / "users.parquet")
    sessions = pd.read_parquet(raw / "sessions.parquet")
    events = pd.read_parquet(raw / "events.parquet")
    impressions = pd.read_parquet(raw / "impressions.parquet")

    price_ds = build_price_dataset(cars, events)
    booking_ds = build_booking_dataset(impressions, sessions, users, cars, events)
    sell_ds = build_sellthrough_dataset(cars, events)

    out = cfg.processed_data
    price_ds.to_parquet(out / "price_dataset.parquet", index=False)
    keep = (["session_id", "user_id", "car_id", "observation_date", "label_booked",
             "clicked", "examined"] + BOOKING_FEATURES)
    booking_ds[keep].to_parquet(out / "booking_dataset.parquet", index=False)
    sell_ds[["car_id", "observation_date", "label_sold_30d"] + SELL_FEATURES].to_parquet(
        out / "sellthrough_dataset.parquet", index=False)
    return dict(price=price_ds, booking=booking_ds[keep], sellthrough=sell_ds)
