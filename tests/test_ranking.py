import numpy as np
import pandas as pd

from driveintent.models.ranking import RANK_FEATURES, historical_recommender_scores, ndcg_at_k
from driveintent.models.recommender import RecommenderBundle


def test_ndcg_math():
    assert ndcg_at_k(np.array([3, 2, 1, 0]), 4) == 1.0
    assert ndcg_at_k(np.array([0, 1, 2, 3]), 4) < 1.0


def test_ranker_excludes_logging_and_in_sample_prediction_features():
    assert "list_position_propensity" not in RANK_FEATURES
    assert "booking_probability" not in RANK_FEATURES


def test_historical_profiles_ignore_future_events(cfg, tables):
    bundle = RecommenderBundle(cfg)
    bundle.content.fit(tables["cars"])
    bundle.collab = None
    imp = tables["impressions"].head(80).copy()
    baseline = historical_recommender_scores(imp, tables["events"], bundle)
    future = tables["events"].dropna(subset=["car_id"]).head(20).copy()
    future["event_id"] = "FUTURE_" + future["event_id"]
    future["event_timestamp"] = pd.to_datetime(imp["event_timestamp"]).max() + pd.Timedelta(days=30)
    changed = historical_recommender_scores(
        imp, pd.concat([tables["events"], future], ignore_index=True), bundle
    )
    cols = ["content_score", "session_score", "collaborative_score"]
    pd.testing.assert_frame_equal(baseline[cols], changed[cols])


def test_ipw_weights_clipped(pipeline):
    import pandas as pd
    cfg = pipeline
    df = pd.read_parquet(cfg.processed_data / "ranking_dataset.parquet")
    wmax = cfg.get("recommendation", "maximum_propensity_weight")
    assert df["ipw"].max() <= wmax + 1e-9
    assert (df["ipw"] >= 1.0).all()


def test_validation_selected_champion_is_deployed(pipeline):
    import json
    metrics = json.loads((pipeline.artifacts / "metrics" / "ranking_metrics.json").read_text())
    assert metrics["deployed_champion"]["ndcg_at_10"] >= 0.50
    assert metrics["validation_selection"]["basis"].startswith("maximum NDCG@10")
