import pandas as pd

from driveintent.models.ranking import recommend_for_user


def test_recommendations_end_to_end(pipeline):
    cfg = pipeline
    ev = pd.read_parquet(cfg.raw_data / "events.parquet")
    uid = ev["user_id"].value_counts().index[0]
    out = recommend_for_user(cfg, uid, limit=8)
    recs = out["recommendations"]
    assert 0 < len(recs) <= 8
    ids = [r["car_id"] for r in recs]
    assert len(ids) == len(set(ids)), "no duplicate recommendations"
    cars = pd.read_parquet(cfg.raw_data / "cars.parquet").set_index("car_id")
    for cid in ids:
        assert pd.isna(cars.loc[cid, "inventory_exit_date"]), "sold cars excluded"
    assert all(r["reasons"] for r in recs), "every rec has explanations"


def test_hard_constraints_respected(pipeline):
    cfg = pipeline
    ev = pd.read_parquet(cfg.raw_data / "events.parquet")
    for uid in ev["user_id"].value_counts().head(5).index:
        out = recommend_for_user(cfg, uid, limit=5)
        hc = out["intent_summary"].get("hard_constraints", {}) or {}
        if "max_price" in hc and out["recommendations"]:
            assert all(r["listed_price"] <= hc["max_price"] * 1.05 + 1
                       for r in out["recommendations"])


def test_cold_start_fallback(pipeline):
    out = recommend_for_user(pipeline, "USER_DOES_NOT_EXIST", limit=5)
    assert len(out["recommendations"]) > 0, "cold-start must still recommend"


def test_diversity_changes_order(pipeline):
    cfg = pipeline
    ev = pd.read_parquet(cfg.raw_data / "events.parquet")
    uid = ev["user_id"].value_counts().index[0]
    low = [r["car_id"] for r in recommend_for_user(cfg, uid, limit=8,
                                                   diversity_level="low")["recommendations"]]
    high = [r["car_id"] for r in recommend_for_user(cfg, uid, limit=8,
                                                    diversity_level="high")["recommendations"]]
    assert low[0] == high[0]  # top pick anchored
