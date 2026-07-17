import pandas as pd

from driveintent.data.validate import validate_feature_list
from driveintent.features.build import BOOKING_FEATURES, PRICE_FEATURES, SELL_FEATURES
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
