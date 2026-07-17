"""DriveIntent FastAPI service."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from driveintent.config import Config, load_config
from driveintent.models import classifiers
from driveintent.models import price as price_model
from driveintent.models.registry import ModelArtifactNotFoundError

log = logging.getLogger("driveintent.api")


def _jsonable(obj):
    """Recursively replace NaN/inf with None for JSON compliance."""
    import math

    import numpy as np
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj
STATE: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    STATE["cfg"] = cfg
    try:  # eager-load artifacts through the registry
        from driveintent.models import registry
        registry.load_model(cfg, "regression", "price")
        registry.load_model(cfg, "classification", "booking")
        registry.load_model(cfg, "classification", "sellthrough")
        registry.load_model(cfg, "ranking", "ranker")
        STATE["models_loaded"] = True
    except ModelArtifactNotFoundError as e:
        log.warning("Model artifacts missing: %s", e)
        STATE["models_loaded"] = False
    yield
    STATE.clear()


app = FastAPI(title="DriveIntent API", version="1.0.0", lifespan=lifespan)


def _cfg() -> Config:
    return STATE["cfg"]


# ---------------------------- schemas ---------------------------------------
class CarInput(BaseModel):
    make: str
    model: str
    variant: str = "Mid"
    body_type: str
    fuel_type: str
    transmission: str
    manufacturing_year: int = Field(ge=2005, le=2026)
    registration_year: Optional[int] = None
    kilometres_driven: float = Field(ge=0)
    owner_count: int = Field(default=1, ge=1, le=6)
    city: str
    state: str = ""
    engine_cc: float = 1200
    claimed_mileage_kmpl: float = 18
    insurance_valid: bool = True
    service_history_available: bool = True
    accident_history: bool = False
    inspection_score: float = Field(default=80, ge=0, le=100)
    exterior_score: float = 80
    interior_score: float = 80
    engine_score: float = 80
    tyre_score: float = 75
    number_of_features: int = 10
    listed_price: Optional[float] = None


class BookingInput(BaseModel):
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    car_id: Optional[str] = None
    features: dict[str, Any] = Field(default_factory=dict,
                                     description="Override any booking-model feature")


class SellthroughInput(BaseModel):
    car_id: str
    snapshot_age: int = 0


class BudgetInput(BaseModel):
    total_budget: Optional[float] = None


# ---------------------------- endpoints -------------------------------------
@app.get("/health")
def health():
    cfg = _cfg()
    return dict(status="healthy",
                database="connected" if cfg.database.exists() else "missing",
                models_loaded=STATE.get("models_loaded", False),
                version="1.0.0")


@app.post("/api/v1/pricing/predict")
def predict_price(car: CarInput):
    cfg = _cfg()
    row = car.model_dump()
    row["registration_year"] = row["registration_year"] or row["manufacturing_year"]
    row["vehicle_age"] = max(2026 - row["manufacturing_year"], 0)
    row["kilometres_per_year"] = row["kilometres_driven"] / max(row["vehicle_age"], 1)
    row["inventory_entry_month"] = 7
    row["market_demand_index"] = 1.0
    row["local_supply_index"] = 0.2
    row["model_popularity"] = 0.75
    try:
        pred = price_model.predict(cfg, row)
        factors = price_model.explain(cfg, row, top_k=5)
    except ModelArtifactNotFoundError as e:
        raise HTTPException(503, str(e))
    fair = pred["predicted_fair_price"]
    out = dict(pred)
    if car.listed_price:
        deal = (fair - car.listed_price) / fair
        out["listed_price"] = car.listed_price
        out["deal_score"] = round(deal, 4)
        out["deal_label"] = ("great_deal" if deal > 0.07 else "good_deal" if deal > 0.02
                             else "fair_price" if deal > -0.05 else "overpriced")
    out["top_factors"] = factors
    return out


@app.post("/api/v1/conversion/booking-probability")
def booking_probability(inp: BookingInput):
    cfg = _cfg()
    ds = pd.read_parquet(cfg.processed_data / "booking_dataset.parquet")
    if inp.session_id and inp.car_id:
        rows = ds[(ds["session_id"] == inp.session_id) & (ds["car_id"] == inp.car_id)]
        if rows.empty:
            raise HTTPException(404, "session_id/car_id pair not found in feature store")
    else:
        rows = ds.sample(1, random_state=0)
    rows = rows.head(1).copy()
    for k, v in inp.features.items():
        if k in rows.columns:
            rows[k] = v
    try:
        p = float(classifiers.predict_proba(cfg, "booking", rows)[0])
        factors = classifiers.explain(cfg, "booking", rows, top_k=5)
    except ModelArtifactNotFoundError as e:
        raise HTTPException(503, str(e))
    quality = "high" if p > 0.15 else "medium" if p > 0.05 else "low"
    action = {"high": "prioritize_callback", "medium": "standard_queue",
              "low": "nurture_campaign"}[quality]
    return dict(booking_probability=round(p, 4), calibrated_probability=round(p, 4),
                lead_quality=quality, recommended_action=action, top_factors=factors,
                model_version="booking_v1")


@app.post("/api/v1/conversion/sellthrough-probability")
def sellthrough_probability(inp: SellthroughInput):
    cfg = _cfg()
    ds = pd.read_parquet(cfg.processed_data / "sellthrough_dataset.parquet")
    rows = ds[ds["car_id"] == inp.car_id]
    if rows.empty:
        raise HTTPException(404, f"car_id {inp.car_id} not found")
    rows = rows[rows["snapshot_age"] <= inp.snapshot_age].tail(1)
    if rows.empty:
        rows = ds[ds["car_id"] == inp.car_id].head(1)
    try:
        p = float(classifiers.predict_proba(cfg, "sellthrough", rows)[0])
    except ModelArtifactNotFoundError as e:
        raise HTTPException(503, str(e))
    return dict(car_id=inp.car_id, sellthrough_probability_30d=round(p, 4),
                snapshot_age=int(rows["snapshot_age"].iloc[0]),
                model_version="sellthrough_v1")


@app.get("/api/v1/recommendations/{user_id}")
def recommendations(user_id: str, session_id: Optional[str] = None,
                    limit: int = Query(10, ge=1, le=50),
                    diversity_level: Optional[str] = Query(None, pattern="^(low|medium|high)$"),
                    include_explanations: bool = True):
    from driveintent.models.ranking import recommend_for_user
    try:
        out = recommend_for_user(_cfg(), user_id, session_id=session_id,
                                 limit=limit, diversity_level=diversity_level)
    except ModelArtifactNotFoundError as e:
        raise HTTPException(503, str(e))
    if not include_explanations:
        for r in out["recommendations"]:
            r.pop("reasons", None)
    return _jsonable(out)


@app.get("/api/v1/recommendations/{user_id}/intent")
def user_intent(user_id: str, session_id: Optional[str] = None):
    cfg = _cfg()
    from driveintent.features.intent import infer_profile
    events = pd.read_parquet(cfg.raw_data / "events.parquet")
    cars = pd.read_parquet(cfg.raw_data / "cars.parquet")
    ue = events[events["user_id"] == user_id]
    if ue.empty:
        raise HTTPException(404, f"No events for user {user_id}")
    sid = session_id or ue.sort_values("event_timestamp")["session_id"].iloc[-1]
    prof = infer_profile(ue, cars, session_id=sid)
    return _jsonable(dict(user_id=user_id, session_id=sid, **prof.to_dict()))


@app.get("/api/v1/inventory/opportunities")
def inventory_opps(city: Optional[str] = None, body_type: Optional[str] = None,
                   minimum_gap: float = 0.0):
    from driveintent.analytics.analytics import inventory_opportunities
    df = inventory_opportunities(_cfg(), city=city, body_type=body_type,
                                 minimum_gap=minimum_gap)
    return _jsonable(dict(count=len(df), opportunities=df.head(50).to_dict(orient="records")))


@app.get("/api/v1/campaigns/performance")
def campaign_perf():
    from driveintent.analytics.analytics import campaign_performance
    df = campaign_performance(_cfg())
    return _jsonable(dict(campaigns=df.to_dict(orient="records")))


@app.post("/api/v1/campaigns/optimize-budget")
def optimize_budget(inp: BudgetInput):
    from driveintent.analytics.analytics import budget_optimizer
    df = budget_optimizer(_cfg(), total_budget=inp.total_budget)
    return dict(note="Scenario simulation based on synthetic response curves, "
                     "not a production Google Ads bidding system.",
                allocation=_jsonable(df.to_dict(orient="records")))
