import numpy as np
import pandas as pd

from driveintent.models import classifiers, price
from driveintent.models.registry import load_model


def test_price_quantiles_ordered(pipeline):
    cfg = pipeline
    ds = pd.read_parquet(cfg.processed_data / "price_dataset.parquet").head(30)
    pred = price.predict(cfg, ds)
    assert np.all(pred["lower_price"] <= pred["predicted_fair_price"] + 1e-6)
    assert np.all(pred["predicted_fair_price"] <= pred["upper_price"] + 1e-6)


def test_price_beats_baseline(pipeline):
    import json
    cfg = pipeline
    m = json.loads((cfg.artifacts / "metrics" / "price_metrics.json").read_text())
    assert m["catboost"]["mae"] < m["baseline_global_median"]["mae"]


def test_booking_probabilities_valid(pipeline):
    cfg = pipeline
    ds = pd.read_parquet(cfg.processed_data / "booking_dataset.parquet").head(50)
    p = classifiers.predict_proba(cfg, "booking", ds)
    assert p.shape == (50,)
    assert np.all((p >= 0) & (p <= 1))


def test_artifacts_reload_and_agree(pipeline):
    cfg = pipeline
    ds = pd.read_parquet(cfg.processed_data / "booking_dataset.parquet").head(10)
    p1 = classifiers.predict_proba(cfg, "booking", ds)
    bundle, meta = load_model(cfg, "classification", "booking")
    assert meta["features"] == bundle["features"]
    p2 = classifiers.predict_proba(cfg, "booking", ds)
    np.testing.assert_allclose(p1, p2)


def test_shap_explanations(pipeline):
    cfg = pipeline
    ds = pd.read_parquet(cfg.processed_data / "price_dataset.parquet")
    out = price.explain(cfg, ds.iloc[0].to_dict(), top_k=3)
    assert len(out) == 3 and {"feature", "direction", "importance"} <= set(out[0])
