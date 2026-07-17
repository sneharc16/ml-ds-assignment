"""Fair-price regression: baselines, CatBoost point + quantile models, SHAP.

Target = transaction_price of sold cars. latent_fair_price is never used.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from driveintent.config import Config
from driveintent.data.validate import validate_feature_list
from driveintent.features.build import PRICE_CAT, PRICE_FEATURES
from driveintent.models import registry
from driveintent.models.common import interval_metrics, regression_metrics, temporal_split

NUM_FEATURES = [f for f in PRICE_FEATURES if f not in PRICE_CAT]


def _pool(df: pd.DataFrame, with_label: bool = True) -> Pool:
    X = df[PRICE_FEATURES].copy()
    for c in PRICE_CAT:
        X[c] = X[c].astype(str)
    return Pool(X, label=df["target_price"] if with_label else None, cat_features=PRICE_CAT)


def train(cfg: Config) -> dict:
    leak = validate_feature_list(PRICE_FEATURES)
    if leak:
        raise ValueError(f"Leakage in price features: {leak}")
    ds = pd.read_parquet(cfg.processed_data / "price_dataset.parquet")
    train_df, val_df, test_df = temporal_split(ds, cfg)
    if len(val_df) < 20:  # small profiles: fold val into train tail
        train_df, val_df = train_df.iloc[:-max(len(train_df)//5, 10)], train_df.iloc[-max(len(train_df)//5, 10):]

    y_tr, y_va, y_te = (d["target_price"].to_numpy() for d in (train_df, val_df, test_df))
    results: dict[str, dict] = {}

    # baseline 1: global median
    results["baseline_global_median"] = regression_metrics(y_te, np.full(len(y_te), np.median(y_tr)))
    # baseline 2: make-model median
    mm = train_df.groupby(["make", "model"])["target_price"].median()
    pred_mm = test_df.set_index(["make", "model"]).index.map(mm).to_numpy(dtype=float)
    pred_mm = np.where(np.isnan(pred_mm), np.median(y_tr), pred_mm)
    results["baseline_make_model_median"] = regression_metrics(y_te, pred_mm)
    # baseline 3: ridge
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUM_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", max_categories=40), PRICE_CAT)])
    ridge = Pipeline([("pre", pre), ("model", Ridge(alpha=1.0))])
    Xtr = train_df[PRICE_FEATURES].copy(); Xte = test_df[PRICE_FEATURES].copy()
    for c in PRICE_CAT:
        Xtr[c] = Xtr[c].astype(str); Xte[c] = Xte[c].astype(str)
    ridge.fit(Xtr, y_tr)
    results["baseline_ridge"] = regression_metrics(y_te, ridge.predict(Xte))

    # main model: CatBoost
    p = cfg.get("models", "price", "catboost")
    cb = CatBoostRegressor(iterations=p["iterations"], learning_rate=p["learning_rate"],
                           depth=p["depth"], loss_function="RMSE",
                           random_seed=cfg.seed, verbose=False, allow_writing_files=False,
                           early_stopping_rounds=40)
    cb.fit(_pool(train_df), eval_set=_pool(val_df))
    pred_te = cb.predict(_pool(test_df, with_label=False))
    m = regression_metrics(y_te, pred_te)

    # quantiles
    qmodels = {}
    qpreds = {}
    for q in cfg.get("models", "price", "quantiles"):
        qm = CatBoostRegressor(iterations=p["iterations"], learning_rate=p["learning_rate"],
                               depth=p["depth"], loss_function=f"Quantile:alpha={q}",
                               random_seed=cfg.seed, verbose=False, allow_writing_files=False)
        qm.fit(_pool(train_df))
        qmodels[f"q{int(q*100)}"] = qm
        qpreds[f"q{int(q*100)}"] = qm.predict(_pool(test_df, with_label=False))

    # Conformalized quantile regression (Romano et al. 2019): calibrate interval
    # width on the validation window so empirical P10-P90 coverage hits the target.
    q10_va = qmodels["q10"].predict(_pool(val_df, with_label=False))
    q90_va = qmodels["q90"].predict(_pool(val_df, with_label=False))
    y_va = val_df["target_price"].to_numpy()
    conformity = np.maximum(q10_va - y_va, y_va - q90_va)
    target = float(cfg.get("models", "price", "interval_coverage") or 0.80)
    n = len(conformity)
    k = min(n - 1, int(np.ceil((n + 1) * target))) if n > 1 else 0
    conformal_delta = float(np.sort(conformity)[k]) if n > 1 else 0.0
    conformal_delta = max(conformal_delta, 0.0)

    lo = np.minimum(qpreds["q10"], qpreds["q50"]) - conformal_delta
    hi = np.maximum(qpreds["q90"], qpreds["q50"]) + conformal_delta
    lo = np.maximum(lo, 0.0)
    m.update(interval_metrics(y_te, lo, hi))
    m["conformal_delta"] = conformal_delta
    results["catboost"] = m
    results["interval_coverage_target"] = target

    # error slices
    slices = {}
    for dim in ("body_type", "city"):
        s = test_df.assign(pred=pred_te).groupby(dim).apply(
            lambda g: regression_metrics(g["target_price"], g["pred"])["mae"],
            include_groups=False)
        slices[f"mae_by_{dim}"] = s.round(0).to_dict()
    results["slices"] = slices

    bundle = dict(point=cb, quantiles=qmodels, features=PRICE_FEATURES,
                  cat_features=PRICE_CAT, conformal_delta=conformal_delta)
    registry.save_model(cfg, "regression", "price", bundle,
                        PRICE_FEATURES, PRICE_CAT, metrics=results["catboost"])
    (cfg.artifacts / "metrics" / "price_metrics.json").write_text(
        json.dumps(results, indent=2, default=str))

    # residual diagnostics data for dashboard
    diag = test_df[["car_id", "target_price", "listed_price", "body_type", "city"]].copy()
    diag["predicted"] = pred_te
    diag["p10"], diag["p90"] = lo, hi
    diag.to_parquet(cfg.artifacts / "metrics" / "price_test_predictions.parquet", index=False)
    return results


def predict(cfg: Config, car_row: dict | pd.DataFrame) -> dict:
    bundle, meta = registry.load_model(cfg, "regression", "price")
    df = pd.DataFrame([car_row]) if isinstance(car_row, dict) else car_row.copy()
    for f in PRICE_FEATURES:
        if f not in df.columns:
            df[f] = np.nan
    df["target_price"] = 0.0
    pool = _pool(df, with_label=False)
    delta = float(bundle.get("conformal_delta", 0.0))
    p50 = float(bundle["point"].predict(pool)[0]) if len(df) == 1 else bundle["point"].predict(pool)
    q10 = bundle["quantiles"]["q10"].predict(pool) - delta
    q90 = bundle["quantiles"]["q90"].predict(pool) + delta
    if isinstance(p50, float):
        lo, hi = float(max(min(q10[0], p50), 0.0)), float(max(q90[0], p50))
        return dict(predicted_fair_price=round(p50, 0), lower_price=round(lo, 0),
                    upper_price=round(hi, 0), model_version=meta["version"])
    return dict(predicted_fair_price=p50,
                lower_price=np.maximum(np.minimum(q10, p50), 0.0),
                upper_price=np.maximum(q90, p50), model_version=meta["version"])


def explain(cfg: Config, car_row: dict, top_k: int = 5) -> list[dict]:
    """Per-prediction SHAP explanation using CatBoost native SHAP values."""
    bundle, _ = registry.load_model(cfg, "regression", "price")
    df = pd.DataFrame([car_row])
    for f in PRICE_FEATURES:
        if f not in df.columns:
            df[f] = np.nan
    df["target_price"] = 0.0
    pool = _pool(df, with_label=False)
    shap = bundle["point"].get_feature_importance(pool, type="ShapValues")[0]
    contrib = dict(zip(PRICE_FEATURES, shap[:-1]))
    top = sorted(contrib.items(), key=lambda kv: -abs(kv[1]))[:top_k]
    total = sum(abs(v) for _, v in contrib.items()) or 1.0
    return [dict(feature=f, direction="increases_price" if v > 0 else "decreases_price",
                 importance=round(abs(v) / total, 3)) for f, v in top]


def global_importance(cfg: Config) -> pd.DataFrame:
    bundle, _ = registry.load_model(cfg, "regression", "price")
    imp = bundle["point"].get_feature_importance()
    return (pd.DataFrame({"feature": PRICE_FEATURES, "importance": imp})
            .sort_values("importance", ascending=False).reset_index(drop=True))
