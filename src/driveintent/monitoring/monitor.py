"""Feature-drift reports and deterministic deployment quality gates."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from driveintent.config import Config
from driveintent.features.build import BOOKING_FEATURES, PRICE_FEATURES, SELL_FEATURES
from driveintent.models.ranking import RANK_FEATURES


def _psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    ref = pd.to_numeric(reference, errors="coerce").dropna().to_numpy()
    cur = pd.to_numeric(current, errors="coerce").dropna().to_numpy()
    if not len(ref) or not len(cur):
        return 0.0
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_p = np.histogram(ref, bins=edges)[0] / len(ref)
    cur_p = np.histogram(cur, bins=edges)[0] / len(cur)
    ref_p, cur_p = np.clip(ref_p, 1e-6, None), np.clip(cur_p, 1e-6, None)
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))


def _js(reference: pd.Series, current: pd.Series) -> float:
    ref = reference.fillna("__MISSING__").astype(str).value_counts(normalize=True)
    cur = current.fillna("__MISSING__").astype(str).value_counts(normalize=True)
    labels = ref.index.union(cur.index)
    p = np.clip(ref.reindex(labels, fill_value=0).to_numpy(), 1e-12, None)
    q = np.clip(cur.reindex(labels, fill_value=0).to_numpy(), 1e-12, None)
    p, q = p / p.sum(), q / q.sum()
    m = (p + q) / 2
    return float(0.5 * np.sum(p * np.log2(p / m)) + 0.5 * np.sum(q * np.log2(q / m)))


def _severity(cfg: Config, metric: str, value: float) -> str:
    warning = float(cfg.get("monitoring", f"{metric}_warning") or 0.1)
    critical = float(cfg.get("monitoring", f"{metric}_critical") or 0.25)
    return "critical" if value >= critical else "warning" if value >= warning else "ok"


def _split(frame: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    time_col = next((col for col in ("observation_date", "event_timestamp", "snapshot_date")
                     if col in frame.columns), None)
    if time_col:
        dt = pd.to_datetime(frame[time_col])
        reference = frame[dt <= pd.Timestamp(cfg.get("splits", "train_end"))]
        current = frame[(dt > pd.Timestamp(cfg.get("splits", "validation_end"))) &
                        (dt <= pd.Timestamp(cfg.get("splits", "test_end")))]
    else:
        cut = max(int(len(frame) * 0.7), 1)
        reference, current = frame.iloc[:cut], frame.iloc[cut:]
    if reference.empty or current.empty:
        cut = max(int(len(frame) * 0.7), 1)
        reference, current = frame.iloc[:cut], frame.iloc[cut:]
    return reference, current


def build_drift_report(cfg: Config) -> dict[str, Any]:
    datasets = {
        "price": ("price_dataset.parquet", PRICE_FEATURES),
        "booking": ("booking_dataset.parquet", BOOKING_FEATURES),
        "sellthrough": ("sellthrough_dataset.parquet", SELL_FEATURES),
        "ranking": ("ranking_dataset.parquet", RANK_FEATURES),
    }
    rows: list[dict[str, Any]] = []
    for model, (filename, features) in datasets.items():
        path = cfg.processed_data / filename
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        reference, current = _split(frame, cfg)
        for feature in features:
            if feature not in frame:
                continue
            missing_delta = float(abs(current[feature].isna().mean() - reference[feature].isna().mean()))
            if pd.api.types.is_numeric_dtype(frame[feature]) and not pd.api.types.is_bool_dtype(frame[feature]):
                metric, value = "psi", _psi(reference[feature], current[feature])
            else:
                metric, value = "js", _js(reference[feature], current[feature])
            severity = max(_severity(cfg, metric, value), _severity(cfg, "missing_rate_delta", missing_delta),
                           key={"ok": 0, "warning": 1, "critical": 2}.get)
            rows.append({"model": model, "feature": feature, "metric": metric,
                         "value": round(value, 6), "missing_rate_delta": round(missing_delta, 6),
                         "severity": severity, "reference_rows": len(reference),
                         "current_rows": len(current)})
    counts = {level: sum(row["severity"] == level for row in rows)
              for level in ("ok", "warning", "critical")}
    status = "critical" if counts["critical"] else "warning" if counts["warning"] else "ok"
    report = {"generated_at": datetime.now(timezone.utc).isoformat(), "status": status,
              "summary": {"feature_checks": len(rows), **counts}, "features": rows}
    path = cfg.artifacts / "monitoring" / "drift_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    return report


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def evaluate_quality_gates(cfg: Config) -> dict[str, Any]:
    metrics = cfg.artifacts / "metrics"
    price = _read_json(metrics / "price_metrics.json")
    booking = _read_json(metrics / "booking_metrics.json")
    sell = _read_json(metrics / "sellthrough_metrics.json")
    ranking = _read_json(metrics / "ranking_metrics.json")
    checks: list[dict[str, Any]] = []

    def add(name: str, value: float, threshold: float, operator: str) -> None:
        passed = value >= threshold if operator == ">=" else value <= threshold
        checks.append({"gate": name, "value": round(float(value), 6),
                       "operator": operator, "threshold": float(threshold), "passed": bool(passed)})

    p = price.get("deployed_champion", {})
    pb = price.get("baseline_global_median", {})
    pcfg = cfg.get("quality_gates", "price")
    add("price_r2", p.get("r2", -np.inf), pcfg["minimum_r2"], ">=")
    add("price_mae", p.get("mae", np.inf), pcfg["maximum_mae"], "<=")
    add("price_vs_global_median", p.get("mae", np.inf) / max(pb.get("mae", 0), 1),
        pcfg["maximum_global_median_mae_ratio"], "<=")
    for name, source in (("booking", booking), ("sellthrough", sell)):
        deployed = source.get("catboost_calibrated", {})
        prevalence = source.get("baseline_prevalence", {}).get("base_rate", 0)
        ccfg = cfg.get("quality_gates", name)
        add(f"{name}_roc_auc", deployed.get("roc_auc", -np.inf), ccfg["minimum_roc_auc"], ">=")
        add(f"{name}_pr_auc_lift", deployed.get("pr_auc", 0) / max(prevalence, 1e-9),
            ccfg["minimum_pr_auc_lift"], ">=")
        add(f"{name}_top5_lift", deployed.get("lift_top5pct", -np.inf),
            ccfg["minimum_top5_lift"], ">=")
    deployed = ranking.get("deployed_champion", {})
    session = ranking.get("session_only", {})
    rcfg = cfg.get("quality_gates", "ranking")
    add("ranking_ndcg_at_10", deployed.get("ndcg_at_10", -np.inf), rcfg["minimum_ndcg_at_10"], ">=")
    degradation = session.get("ndcg_at_10", 0) - deployed.get("ndcg_at_10", -np.inf)
    add("ranking_session_baseline_degradation", degradation,
        rcfg["maximum_session_baseline_degradation"], "<=")
    report = {"generated_at": datetime.now(timezone.utc).isoformat(),
              "status": "pass" if all(c["passed"] for c in checks) else "fail",
              "passed": sum(c["passed"] for c in checks), "total": len(checks), "checks": checks}
    path = cfg.artifacts / "monitoring" / "quality_gates.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    return report


def run_monitoring(cfg: Config) -> dict[str, Any]:
    drift = build_drift_report(cfg)
    quality = evaluate_quality_gates(cfg)
    status = {"generated_at": datetime.now(timezone.utc).isoformat(),
              "deployment_ready": quality["status"] == "pass",
              "quality_gates": {key: quality[key] for key in ("status", "passed", "total")},
              "drift": {"status": drift["status"], **drift["summary"]}}
    (cfg.artifacts / "monitoring" / "status.json").write_text(json.dumps(status, indent=2))
    return status
