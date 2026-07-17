"""Explicit data-quality validation for the synthetic marketplace tables."""
from __future__ import annotations

import pandas as pd

FORBIDDEN_FEATURES = {"_latent_fair_price", "_deal_latent", "latent_fair_price",
                      "transaction_price", "inventory_exit_date", "sold_flag"}


class ValidationError(Exception):
    pass


def _check(cond: bool, msg: str, errors: list[str]) -> None:
    if not cond:
        errors.append(msg)


def validate_cars(cars: pd.DataFrame) -> list[str]:
    e: list[str] = []
    _check(cars["car_id"].is_unique, "car_id not unique", e)
    _check((cars["listed_price"] > 0).all(), "non-positive listed_price", e)
    _check((cars["acquisition_price"] > 0).all(), "non-positive acquisition_price", e)
    _check(cars["manufacturing_year"].between(2005, 2026).all(), "manufacturing_year out of range", e)
    _check((cars["registration_year"] >= cars["manufacturing_year"]).all(),
           "registration_year before manufacturing_year", e)
    _check((cars["kilometres_driven"] >= 0).all(), "negative kilometres", e)
    _check(cars["inspection_score"].between(0, 100).all(), "inspection_score out of [0,100]", e)
    sold = cars["sold_flag"]
    _check(cars.loc[sold, "transaction_price"].notna().all(), "sold car missing transaction_price", e)
    _check(cars.loc[~sold, "transaction_price"].isna().all(), "unsold car has transaction_price", e)
    _check(cars.loc[~sold, "inventory_exit_date"].isna().all(), "unsold car has exit date", e)
    _check(cars["body_type"].isin(["Hatchback", "Sedan", "Compact SUV", "SUV", "MPV"]).all(),
           "invalid body_type", e)
    _check(cars["owner_count"].between(1, 6).all(), "owner_count out of range", e)
    return e


def validate_users(users: pd.DataFrame) -> list[str]:
    e: list[str] = []
    _check(users["user_id"].is_unique, "user_id not unique", e)
    for col in ["first_time_buyer_probability", "quality_sensitivity", "price_sensitivity",
                "brand_loyalty", "exploration_tendency", "finance_interest", "purchase_urgency"]:
        _check(users[col].between(0, 1).all(), f"{col} outside [0,1]", e)
    _check((users["maximum_budget"] >= users["ideal_budget"]).all(),
           "maximum_budget below ideal_budget", e)
    return e


def validate_sessions_events(sessions: pd.DataFrame, events: pd.DataFrame,
                             users: pd.DataFrame, cars: pd.DataFrame) -> list[str]:
    e: list[str] = []
    _check(sessions["session_id"].is_unique, "session_id not unique", e)
    _check(events["event_id"].is_unique, "event_id not unique", e)
    _check((sessions["session_end"] >= sessions["session_start"]).all(),
           "session_end before session_start", e)
    _check(events["user_id"].isin(set(users["user_id"])).all(), "event user FK broken", e)
    _check(events["session_id"].isin(set(sessions["session_id"])).all(), "event session FK broken", e)
    car_events = events["car_id"].dropna()
    _check(car_events.isin(set(cars["car_id"])).all(), "event car FK broken", e)
    # events within session boundaries (allow tiny tolerance)
    bounds = sessions.set_index("session_id")[["session_start", "session_end"]]
    ev = events.join(bounds, on="session_id")
    _check((ev["event_timestamp"] >= ev["session_start"]).all(), "event before session start", e)
    _check((ev["event_timestamp"] <= ev["session_end"] + pd.Timedelta(seconds=1)).all(),
           "event after session end", e)
    valid_events = {"session_start", "view_home", "search", "apply_filter", "remove_filter",
                    "view_search_results", "select_item", "view_item", "view_gallery",
                    "view_inspection_report", "compare_car", "calculate_emi", "view_finance_offer",
                    "add_to_wishlist", "remove_from_wishlist", "request_callback",
                    "book_test_drive", "begin_checkout", "booking_complete", "purchase",
                    "session_end"}
    _check(events["event_name"].isin(valid_events).all(), "invalid event_name", e)
    return e


def validate_feature_list(features: list[str]) -> list[str]:
    bad = FORBIDDEN_FEATURES.intersection(features)
    return [f"leakage feature present: {sorted(bad)}"] if bad else []


def validate_all(tables: dict[str, pd.DataFrame], raise_on_error: bool = True) -> list[str]:
    errors: list[str] = []
    errors += validate_cars(tables["cars"])
    errors += validate_users(tables["users"])
    errors += validate_sessions_events(tables["sessions"], tables["events"],
                                       tables["users"], tables["cars"])
    if errors and raise_on_error:
        raise ValidationError("; ".join(errors))
    return errors
