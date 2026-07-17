"""Shared model utilities: temporal splits, metrics, calibration, bootstrap CIs."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from driveintent.config import Config


def temporal_split(df: pd.DataFrame, cfg: Config,
                   date_col: str = "observation_date") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = pd.to_datetime(df[date_col])
    tr_end = pd.Timestamp(cfg.get("splits", "train_end"))
    va_end = pd.Timestamp(cfg.get("splits", "validation_end"))
    te_end = pd.Timestamp(cfg.get("splits", "test_end"))
    train = df[d <= tr_end]
    val = df[(d > tr_end) & (d <= va_end)]
    test = df[(d > va_end) & (d <= te_end)]
    return train, val, test


# ---------------------- regression -----------------------------------------
def regression_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y, p = np.asarray(y, float), np.asarray(p, float)
    resid = y - p
    safe = np.clip(np.abs(y), 1e4, None)
    return dict(
        mae=float(mean_absolute_error(y, p)),
        rmse=float(np.sqrt(mean_squared_error(y, p))),
        r2=float(r2_score(y, p)),
        median_ae=float(np.median(np.abs(resid))),
        mape=float(np.mean(np.abs(resid) / safe)),
        smape=float(np.mean(2 * np.abs(resid) / (np.abs(y) + np.abs(p) + 1e-9))),
        residual_mean=float(resid.mean()),
        residual_std=float(resid.std()),
    )


def interval_metrics(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, float)
    cover = float(np.mean((y >= lo) & (y <= hi)))
    return dict(p10_p90_coverage=cover,
                mean_interval_width=float(np.mean(hi - lo)),
                mean_relative_width=float(np.mean((hi - lo) / np.clip(y, 1e4, None))))


# ---------------------- classification -------------------------------------
def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= 1.0)
        if m.sum():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def classification_metrics(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y, p = np.asarray(y, int), np.asarray(p, float)
    yhat = (p >= threshold).astype(int)
    out = dict(
        roc_auc=float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        pr_auc=float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        log_loss=float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))),
        brier=float(brier_score_loss(y, p)),
        ece=expected_calibration_error(y, p),
        precision=float(precision_score(y, yhat, zero_division=0)),
        recall=float(recall_score(y, yhat, zero_division=0)),
        f1=float(f1_score(y, yhat, zero_division=0)),
        base_rate=float(y.mean()),
        threshold=threshold,
    )
    for frac, name in [(0.05, "top5pct"), (0.10, "top10pct")]:
        k = max(int(len(p) * frac), 1)
        top = np.argsort(-p)[:k]
        prec = float(y[top].mean())
        out[f"precision_{name}"] = prec
        out[f"recall_{name}"] = float(y[top].sum() / max(y.sum(), 1))
        out[f"lift_{name}"] = prec / max(y.mean(), 1e-9)
    return out


def lift_table(y: np.ndarray, p: np.ndarray, deciles: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y, int), "p": np.asarray(p, float)})
    df = df.sort_values("p", ascending=False).reset_index(drop=True)
    df["decile"] = (df.index * deciles // len(df)) + 1
    g = df.groupby("decile").agg(n=("y", "size"), mean_prediction=("p", "mean"),
                                 actual_rate=("y", "mean"), conversions=("y", "sum"))
    g["cumulative_conversions"] = g["conversions"].cumsum()
    g["cumulative_gain"] = g["cumulative_conversions"] / max(df["y"].sum(), 1)
    g["lift"] = g["actual_rate"] / max(df["y"].mean(), 1e-9)
    return g.reset_index()


def threshold_table(y: np.ndarray, p: np.ndarray,
                    thresholds: np.ndarray | None = None) -> pd.DataFrame:
    thresholds = thresholds if thresholds is not None else np.round(np.arange(0.05, 0.95, 0.05), 2)
    rows = []
    y = np.asarray(y, int)
    for t in thresholds:
        yhat = (p >= t).astype(int)
        rows.append(dict(threshold=float(t),
                         precision=float(precision_score(y, yhat, zero_division=0)),
                         recall=float(recall_score(y, yhat, zero_division=0)),
                         f1=float(f1_score(y, yhat, zero_division=0)),
                         predicted_positive_rate=float(yhat.mean()),
                         leads_selected=int(yhat.sum())))
    return pd.DataFrame(rows)


def reliability_curve(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y": y, "p": p})
    df["bin"] = np.clip((df["p"] * bins).astype(int), 0, bins - 1)
    return (df.groupby("bin").agg(mean_predicted=("p", "mean"),
                                  observed_rate=("y", "mean"), n=("y", "size"))
            .reset_index())


# ---------------------- bootstrap ------------------------------------------
def bootstrap_ci(values_fn, units: np.ndarray, data: pd.DataFrame,
                 n_boot: int = 200, seed: int = 42, alpha: float = 0.05) -> dict[str, float]:
    """Cluster bootstrap: resample `units` (e.g. user_ids), compute metric via values_fn(sub_df)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(units)
    groups = {u: data[units == u] for u in uniq} if len(uniq) < 5000 else None
    stats = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        if groups is not None:
            sub = pd.concat([groups[u] for u in pick], ignore_index=True)
        else:
            sub = data[np.isin(units, pick)]
        try:
            stats.append(values_fn(sub))
        except Exception:
            continue
    stats = np.array(stats)
    return dict(mean=float(stats.mean()),
                ci_low=float(np.quantile(stats, alpha / 2)),
                ci_high=float(np.quantile(stats, 1 - alpha / 2)),
                n_boot=len(stats))
