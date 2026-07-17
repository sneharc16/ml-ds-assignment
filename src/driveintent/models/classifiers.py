"""Booking and sell-through classifiers: baselines, CatBoost, calibration, SHAP.

Both use temporal splits; calibration (isotonic if enough positives, else
sigmoid/Platt) is fit on the validation window; thresholds selected on
validation only.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.compose import ColumnTransformer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from driveintent.config import Config
from driveintent.features.build import BOOKING_FEATURES, SELL_FEATURES
from driveintent.models import registry
from driveintent.models.common import (
    classification_metrics,
    lift_table,
    reliability_curve,
    temporal_split,
    threshold_table,
)

SPECS = {
    "booking": dict(
        dataset="booking_dataset.parquet", label="label_booked",
        features=BOOKING_FEATURES,
        cat=["device_category", "source", "medium", "campaign_id", "body_type",
             "fuel_type", "transmission", "make", "city"],
        cfg_key="booking", category="classification"),
    "sellthrough": dict(
        dataset="sellthrough_dataset.parquet", label="label_sold_30d",
        features=SELL_FEATURES,
        cat=["make", "model", "body_type", "fuel_type", "transmission", "city"],
        cfg_key="sellthrough", category="classification"),
}


class _Sigmoid:
    """Platt scaling calibrator."""
    def __init__(self):
        self.lr = LogisticRegression(max_iter=1000)

    def fit(self, p, y):
        self.lr.fit(np.log(np.clip(p, 1e-6, 1 - 1e-6)).reshape(-1, 1) -
                    np.log(1 - np.clip(p, 1e-6, 1 - 1e-6)).reshape(-1, 1), y)
        return self

    def predict(self, p):
        z = (np.log(np.clip(p, 1e-6, 1 - 1e-6)) - np.log(1 - np.clip(p, 1e-6, 1 - 1e-6)))
        return self.lr.predict_proba(z.reshape(-1, 1))[:, 1]


def _pool(df: pd.DataFrame, spec: dict, with_label: bool = True) -> Pool:
    X = df[spec["features"]].copy()
    for c in spec["cat"]:
        X[c] = X[c].astype(str)
    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(int)
    return Pool(X, label=df[spec["label"]] if with_label else None, cat_features=spec["cat"])


def train(cfg: Config, which: str) -> dict:
    spec = SPECS[which]
    ds = pd.read_parquet(cfg.processed_data / spec["dataset"])
    train_df, val_df, test_df = temporal_split(ds, cfg)
    if len(val_df) < 50 or val_df[spec["label"]].sum() < 5:
        cut = max(len(train_df) // 5, 50)
        train_df, val_df = train_df.iloc[:-cut], train_df.iloc[-cut:]

    y_tr = train_df[spec["label"]].to_numpy()
    y_va = val_df[spec["label"]].to_numpy()
    y_te = test_df[spec["label"]].to_numpy()
    results: dict[str, dict] = {}

    # baseline: prevalence
    results["baseline_prevalence"] = classification_metrics(
        y_te, np.full(len(y_te), y_tr.mean()))

    # baseline: logistic regression
    num = [f for f in spec["features"] if f not in spec["cat"]]
    pre = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", max_categories=30), spec["cat"])])
    lr = Pipeline([("pre", pre), ("m", LogisticRegression(max_iter=2000, class_weight="balanced"))])
    Xtr = train_df[spec["features"]].copy(); Xte = test_df[spec["features"]].copy()
    for c in spec["cat"]:
        Xtr[c] = Xtr[c].astype(str); Xte[c] = Xte[c].astype(str)
    Xtr = Xtr.fillna(0); Xte = Xte.fillna(0)
    lr.fit(Xtr, y_tr)
    results["baseline_logistic"] = classification_metrics(y_te, lr.predict_proba(Xte)[:, 1])

    # main: CatBoost with class weighting
    p = cfg.get("models", spec["cfg_key"], "catboost")
    w1 = float((len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1))
    cb = CatBoostClassifier(iterations=p["iterations"], learning_rate=p["learning_rate"],
                            depth=p["depth"], loss_function="Logloss",
                            class_weights=[1.0, min(w1, 20.0)],
                            random_seed=cfg.seed, verbose=False, allow_writing_files=False,
                            early_stopping_rounds=40, eval_metric="PRAUC")
    cb.fit(_pool(train_df, spec), eval_set=_pool(val_df, spec))
    raw_va = cb.predict_proba(_pool(val_df, spec, with_label=False))[:, 1]
    raw_te = cb.predict_proba(_pool(test_df, spec, with_label=False))[:, 1]

    # calibration on validation
    if y_va.sum() >= 50:
        cal = IsotonicRegression(out_of_bounds="clip").fit(raw_va, y_va)
        cal_method = "isotonic"
        cal_te = cal.predict(raw_te)
    else:
        cal = _Sigmoid().fit(raw_va, y_va)
        cal_method = "sigmoid"
        cal_te = cal.predict(raw_te)

    m_raw = classification_metrics(y_te, raw_te)
    m_cal = classification_metrics(y_te, cal_te)
    results["catboost_raw"] = m_raw
    results["catboost_calibrated"] = m_cal
    results["calibration_method"] = cal_method

    # threshold selection on VALIDATION calibrated probs (never test)
    cal_va = cal.predict(raw_va)
    tt_val = threshold_table(y_va, cal_va)
    best = tt_val.loc[tt_val["f1"].idxmax()]
    results["selected_threshold"] = dict(threshold=float(best["threshold"]),
                                         val_f1=float(best["f1"]),
                                         basis="max F1 on validation window")

    # persist tables for dashboard
    mdir = cfg.artifacts / "metrics"
    threshold_table(y_te, cal_te).to_csv(mdir / f"{which}_threshold_table.csv", index=False)
    lift_table(y_te, cal_te).to_csv(mdir / f"{which}_lift_table.csv", index=False)
    reliability_curve(y_te, cal_te).to_csv(mdir / f"{which}_reliability.csv", index=False)
    pd.DataFrame({"y": y_te, "p_raw": raw_te, "p_cal": cal_te}).to_parquet(
        mdir / f"{which}_test_predictions.parquet", index=False)

    bundle = dict(model=cb, calibrator=cal, calibration_method=cal_method,
                  features=spec["features"], cat_features=spec["cat"],
                  threshold=float(best["threshold"]))
    registry.save_model(cfg, spec["category"], which, bundle,
                        spec["features"], spec["cat"], metrics=m_cal,
                        extra=dict(calibration_method=cal_method,
                                   selected_threshold=float(best["threshold"])))
    (mdir / f"{which}_metrics.json").write_text(json.dumps(results, indent=2, default=str))
    return results


def predict_proba(cfg: Config, which: str, rows: pd.DataFrame) -> np.ndarray:
    bundle, _ = registry.load_model(cfg, "classification", which)
    spec = SPECS[which]
    df = rows.copy()
    for f in spec["features"]:
        if f not in df.columns:
            df[f] = 0
    df[spec["label"]] = 0
    raw = bundle["model"].predict_proba(_pool(df, spec, with_label=False))[:, 1]
    return np.clip(bundle["calibrator"].predict(raw), 0, 1)


def explain(cfg: Config, which: str, row: pd.DataFrame, top_k: int = 5) -> list[dict]:
    bundle, _ = registry.load_model(cfg, "classification", which)
    spec = SPECS[which]
    df = row.copy()
    for f in spec["features"]:
        if f not in df.columns:
            df[f] = 0
    df[spec["label"]] = 0
    shap = bundle["model"].get_feature_importance(
        _pool(df, spec, with_label=False), type="ShapValues")[0]
    contrib = dict(zip(spec["features"], shap[:-1]))
    top = sorted(contrib.items(), key=lambda kv: -abs(kv[1]))[:top_k]
    total = sum(abs(v) for v in contrib.values()) or 1.0
    return [dict(feature=f, direction="increases_probability" if v > 0 else "decreases_probability",
                 importance=round(abs(v) / total, 3)) for f, v in top]
