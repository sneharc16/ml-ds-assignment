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
                             users: pd.DataFrame, cars: pd.DataFrame,
                             campaigns: pd.DataFrame | None = None) -> list[str]:
    e: list[str] = []
    _check(sessions["session_id"].is_unique, "session_id not unique", e)
    _check(events["event_id"].is_unique, "event_id not unique", e)
    _check(sessions["user_id"].isin(set(users["user_id"])).all(), "session user FK broken", e)
    session_users = sessions.set_index("session_id")["user_id"]
    _check(events["user_id"].eq(events["session_id"].map(session_users)).all(),
           "event user does not match session user", e)
    signup = users.set_index("user_id")["signup_date"].pipe(pd.to_datetime)
    _check((sessions["session_start"] >= sessions["user_id"].map(signup)).all(),
           "session before user signup", e)
    chronological = sessions.sort_values(["user_id", "session_start"])
    expected_sequence = chronological.groupby("user_id").cumcount() + 1
    _check(chronological["session_sequence_number"].eq(expected_sequence).all(),
           "session sequence is not chronological", e)
    if campaigns is not None:
        campaign_ids = set(campaigns["campaign_id"])
        _check(sessions["campaign_id"].isin(campaign_ids).all(), "session campaign FK broken", e)
        _check(events["campaign_id"].isin(campaign_ids).all(), "event campaign FK broken", e)
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
    purchases = events[events["event_name"] == "purchase"].dropna(subset=["car_id"])
    _check(purchases["car_id"].is_unique, "car has multiple purchase events", e)
    if len(purchases):
        exit_dates = cars.set_index("car_id")["inventory_exit_date"].pipe(pd.to_datetime)
        _check(pd.to_datetime(purchases["event_timestamp"]).dt.normalize().eq(
            purchases["car_id"].map(exit_dates)).all(), "purchase does not match inventory exit date", e)
    return e


def validate_impressions(impressions: pd.DataFrame, sessions: pd.DataFrame,
                         events: pd.DataFrame, cars: pd.DataFrame) -> list[str]:
    e: list[str] = []
    keys = ["session_id", "car_id"]
    _check(~impressions.duplicated(keys).any(), "impression key not unique", e)
    _check(impressions["session_id"].isin(set(sessions["session_id"])).all(),
           "impression session FK broken", e)
    _check(impressions["car_id"].isin(set(cars["car_id"])).all(),
           "impression car FK broken", e)
    session_users = sessions.set_index("session_id")["user_id"]
    _check(impressions["user_id"].eq(impressions["session_id"].map(session_users)).all(),
           "impression user does not match session user", e)
    _check((~impressions["clicked"] | impressions["examined"]).all(),
           "clicked impression was not examined", e)
    _check((~impressions["booked"] | impressions["clicked"]).all(),
           "booked impression was not clicked", e)
    _check((~impressions["purchased"] | impressions["booked"]).all(),
           "purchased impression was not booked", e)
    event_keys = set(map(tuple, events[["session_id", "car_id", "event_name"]].dropna().to_numpy()))
    for flag, event_name in (("booked", "booking_complete"), ("purchased", "purchase")):
        flagged = impressions[impressions[flag]]
        _check(all((r.session_id, r.car_id, event_name) in event_keys
                   for r in flagged.itertuples()), f"{flag} impression missing {event_name} event", e)
    return e


def validate_feature_list(features: list[str]) -> list[str]:
    bad = FORBIDDEN_FEATURES.intersection(features)
    return [f"leakage feature present: {sorted(bad)}"] if bad else []


def validate_all(tables: dict[str, pd.DataFrame], raise_on_error: bool = True) -> list[str]:
    errors: list[str] = []
    errors += validate_cars(tables["cars"])
    errors += validate_users(tables["users"])
    errors += validate_sessions_events(tables["sessions"], tables["events"],
                                       tables["users"], tables["cars"], tables["campaigns"])
    errors += validate_impressions(tables["impressions"], tables["sessions"],
                                   tables["events"], tables["cars"])
    if errors and raise_on_error:
        raise ValidationError("; ".join(errors))
    return errors
