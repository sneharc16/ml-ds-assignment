"""Interpretable hard/soft preference inference from observed events.

Long-term preferences use event-weighted, recency-decayed history.
Session preferences use only the current session with filter emphasis.
Hard constraints are detected from repeated/low-entropy filter behaviour.
Only observed events are used - never generator latents.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

EVENT_WEIGHTS = {
    "view_search_results": 0.5,
    "select_item": 2.0,
    "view_item": 3.0,
    "view_gallery": 4.0,
    "view_inspection_report": 6.0,
    "compare_car": 7.0,
    "calculate_emi": 8.0,
    "add_to_wishlist": 10.0,
    "request_callback": 14.0,
    "book_test_drive": 18.0,
    "booking_complete": 22.0,
    "purchase": 30.0,
}

DECAY_LAMBDA_PER_DAY = 0.03  # long-term recency decay


def entropy(probs: dict[str, float]) -> tuple[float, float]:
    """Shannon entropy and normalized entropy of a categorical distribution."""
    p = np.array([v for v in probs.values() if v > 0], dtype=float)
    if len(p) <= 1:
        return 0.0, 0.0
    p = p / p.sum()
    h = float(-(p * np.log(p)).sum())
    return h, h / math.log(len(p))


@dataclass
class PreferenceProfile:
    long_term: dict[str, dict[str, float]] = field(default_factory=dict)
    session: dict[str, dict[str, float]] = field(default_factory=dict)
    hard_constraints: dict[str, Any] = field(default_factory=dict)
    soft_preferences: dict[str, Any] = field(default_factory=dict)
    entropy_by_dim: dict[str, float] = field(default_factory=dict)
    session_entropy: float = 1.0
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return dict(long_term=self.long_term, session=self.session,
                    hard_constraints=self.hard_constraints,
                    soft_preferences=self.soft_preferences,
                    entropy_by_dim=self.entropy_by_dim,
                    session_entropy=round(self.session_entropy, 3),
                    confidence=round(self.confidence, 3))


def _weighted_dist(ev: pd.DataFrame, cars: pd.DataFrame, dim: str,
                   ref_time: pd.Timestamp, decay: float) -> dict[str, float]:
    e = ev.dropna(subset=["car_id"]).merge(cars[["car_id", dim]], on="car_id", how="left")
    if e.empty:
        return {}
    w = e["event_name"].map(EVENT_WEIGHTS).fillna(0.5).to_numpy()
    dt = (ref_time - pd.to_datetime(e["event_timestamp"])).dt.total_seconds().to_numpy() / 86400.0
    w = w * np.exp(-decay * np.clip(dt, 0, None))
    dist = pd.Series(w).groupby(e[dim].to_numpy()).sum()
    total = dist.sum()
    return {} if total <= 0 else (dist / total).round(4).to_dict()


def infer_profile(user_events: pd.DataFrame, cars: pd.DataFrame,
                  session_id: str | None = None,
                  ref_time: pd.Timestamp | None = None,
                  session_half_life_minutes: float = 20.0) -> PreferenceProfile:
    """Infer long-term + session preferences, hard constraints, entropy."""
    prof = PreferenceProfile()
    if user_events.empty:
        return prof
    ref_time = ref_time or pd.to_datetime(user_events["event_timestamp"]).max()
    dims = ["body_type", "make", "fuel_type", "transmission"]

    hist = user_events if session_id is None else user_events[user_events["session_id"] != session_id]
    for dim in dims:
        prof.long_term[dim] = _weighted_dist(hist, cars, dim, ref_time, DECAY_LAMBDA_PER_DAY)

    sess_ev = (user_events[user_events["session_id"] == session_id]
               if session_id is not None else
               user_events[user_events["session_id"] == user_events.iloc[-1]["session_id"]])
    lam = math.log(2) / max(session_half_life_minutes, 1e-6)  # per-minute
    for dim in dims:
        e = sess_ev.dropna(subset=["car_id"]).merge(cars[["car_id", dim]], on="car_id", how="left")
        if e.empty:
            prof.session[dim] = {}
            continue
        w = e["event_name"].map(EVENT_WEIGHTS).fillna(0.5).to_numpy()
        dt_min = (ref_time - pd.to_datetime(e["event_timestamp"])).dt.total_seconds().to_numpy() / 60.0
        w = w * np.exp(-lam * np.clip(dt_min, 0, None))
        dist = pd.Series(w).groupby(e[dim].to_numpy()).sum()
        prof.session[dim] = (dist / dist.sum()).round(4).to_dict() if dist.sum() > 0 else {}

    # ---- hard constraints from filter behaviour --------------------------
    filters = sess_ev[sess_ev["event_name"] == "apply_filter"]
    fmap = {"body_type": "body_type", "transmission": "transmission",
            "fuel_type": "fuel_type", "max_price": "max_price"}
    for fname, dim in fmap.items():
        vals = filters.loc[filters["filter_name"] == fname, "filter_value"]
        if vals.empty:
            continue
        top = vals.mode().iloc[0]
        if fname == "max_price":
            prof.hard_constraints["max_price"] = float(top)
            continue
        # hard if the categorical session distribution is concentrated on the filtered value
        dist = prof.session.get(dim, {})
        _, hnorm = entropy(dist) if dist else (0.0, 1.0)
        share = dist.get(top, 0.0)
        if share >= 0.6 or hnorm <= 0.35 or len(vals) >= 2:
            prof.hard_constraints[dim] = top
        else:
            prof.soft_preferences[dim] = top

    # ---- soft preferences: top long-term categories not already hard -----
    for dim in dims:
        dist = prof.long_term.get(dim, {})
        if dist and dim not in prof.hard_constraints:
            top_v = max(dist, key=dist.get)
            if dist[top_v] >= 0.35:
                prof.soft_preferences.setdefault(dim, top_v)

    # ---- entropy & confidence ---------------------------------------------
    hs = []
    for dim in dims:
        dist = prof.session.get(dim) or prof.long_term.get(dim) or {}
        _, hn = entropy(dist)
        prof.entropy_by_dim[dim] = round(hn, 3)
        hs.append(hn)
    prof.session_entropy = float(np.mean(hs)) if hs else 1.0
    n_ev = len(user_events)
    prof.confidence = float(np.clip((1 - prof.session_entropy) * min(n_ev / 30, 1.0) + 0.15, 0, 1))
    return prof


def budget_from_events(user_events: pd.DataFrame, cars: pd.DataFrame,
                       fallback: float | None = None) -> float | None:
    """Infer max budget: explicit price filter > engaged price distribution."""
    filt = user_events[(user_events["event_name"] == "apply_filter")
                       & (user_events["filter_name"] == "max_price")]
    if not filt.empty:
        return float(filt["filter_value"].astype(float).median())
    e = user_events.dropna(subset=["car_id"]).merge(
        cars[["car_id", "listed_price"]], on="car_id", how="left")
    if not e.empty:
        return float(e["listed_price"].quantile(0.85))
    return fallback
