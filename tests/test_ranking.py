import json

import numpy as np

from driveintent.models.ranking import ndcg_at_k


def test_ndcg_math():
    assert ndcg_at_k(np.array([3, 2, 1, 0]), 4) == 1.0
    assert ndcg_at_k(np.array([0, 1, 2, 3]), 4) < 1.0


def test_ranker_beats_popularity(pipeline):
    cfg = pipeline
    m = json.loads((cfg.artifacts / "metrics" / "ranking_metrics.json").read_text())
    assert m["ranker"]["ndcg_at_10"] > m["popularity"]["ndcg_at_10"]


def test_ipw_weights_clipped(pipeline):
    import pandas as pd
    cfg = pipeline
    df = pd.read_parquet(cfg.processed_data / "ranking_dataset.parquet")
    wmax = cfg.get("recommendation", "maximum_propensity_weight")
    assert df["ipw"].max() <= wmax + 1e-9
    assert (df["ipw"] >= 1.0).all()
