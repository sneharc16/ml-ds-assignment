import pandas as pd

from driveintent.data.validate import validate_feature_list
from driveintent.features.build import (
    BOOKING_FEATURES,
    PRICE_FEATURES,
    SELL_FEATURES,
    build_booking_dataset,
    build_price_dataset,
)
from driveintent.features.intent import entropy, infer_profile


def test_no_leakage_in_feature_lists():
    for feats in (PRICE_FEATURES, BOOKING_FEATURES, SELL_FEATURES):
        assert validate_feature_list(feats) == []


def test_entropy_bounds():
    h, hn = entropy({"a": 1.0})
    assert hn == 0.0
    h, hn = entropy({"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25})
    assert abs(hn - 1.0) < 1e-9


def test_intent_inference(tables):
    ev = tables["events"]
    cars = tables["cars"]
    uid = ev["user_id"].value_counts().index[0]
    ue = ev[ev["user_id"] == uid]
    sid = ue.sort_values("event_timestamp")["session_id"].iloc[-1]
    prof = infer_profile(ue, cars, session_id=sid)
    assert 0.0 <= prof.session_entropy <= 1.0
    assert 0.0 <= prof.confidence <= 1.0
    for dist in prof.long_term.values():
        if dist:
            assert abs(sum(dist.values()) - 1.0) < 0.02


def test_sellthrough_no_future_engagement(pipeline):
    cfg = pipeline
    ds = pd.read_parquet(cfg.processed_data / "sellthrough_dataset.parquet")
    # day-0 snapshots should have (near) zero engagement counters
    d0 = ds[ds["snapshot_age"] == 0]
    assert d0["views"].median() == 0


def test_price_observation_is_listing_time(tables):
    ds = build_price_dataset(tables["cars"], tables["events"])
    entry = pd.to_datetime(tables["cars"].set_index("car_id")["inventory_entry_date"])
    assert ds["observation_date"].eq(ds["car_id"].map(entry)).all()


def test_booking_features_ignore_post_decision_events(tables):
    args = [tables[name] for name in ("impressions", "sessions", "users", "cars")]
    baseline = build_booking_dataset(*args, tables["events"])
    events = tables["events"].copy()
    future = events[events["event_name"] == "session_end"].copy()
    future["event_id"] = "FUTURE_" + future["event_id"]
    future["event_name"] = "booking_complete"
    changed = build_booking_dataset(*args, pd.concat([events, future], ignore_index=True))
    cols = ["session_duration_s", "session_searches", "session_filters", "session_events",
            "market_demand_index", "local_supply_index"]
    pd.testing.assert_frame_equal(baseline[cols], changed[cols])
