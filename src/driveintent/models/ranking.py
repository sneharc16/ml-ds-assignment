"""Learning-to-rank + multi-objective reranking + offline evaluation.

- Ranking dataset: session-grouped impressions with graded relevance and
  hybrid recommender scores as features
- Position bias: inverse-propensity weights (clipped) from the synthetic
  examination-propensity curve; click labels alone are biased because
  higher positions get examined more often regardless of relevance
- Ranker: CatBoostRanker (YetiRank)
- Reranker: configurable weighted multi-objective score + MMR diversity,
  entropy-adaptive lambda
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime

import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool

from driveintent.config import Config
from driveintent.models import registry
from driveintent.models.common import temporal_split
from driveintent.models.recommender import RecommenderBundle

RELEVANCE = {
    "impression_only": 0, "select_item": 1, "view_item": 2, "view_gallery": 3,
    "view_inspection_report": 4, "compare_car": 4, "calculate_emi": 5,
    "add_to_wishlist": 6, "request_callback": 7, "book_test_drive": 8,
    "booking_complete": 9, "purchase": 10,
}

RANK_FEATURES = [
    "content_score", "collaborative_score", "session_score",
    "price_budget_gap", "brand_match", "body_match", "trans_match",
    "deal_score", "inspection_score", "vehicle_age",
    "inventory_age", "market_demand_index",
    "finance_available", "delivery_available", "model_popularity",
]


def relevance_from_impression(row: pd.Series) -> int:
    if row.get("purchased"):
        return RELEVANCE["purchase"]
    if row.get("booked"):
        return RELEVANCE["booking_complete"]
    if row.get("callback"):
        return RELEVANCE["request_callback"]
    if row.get("wishlisted"):
        return RELEVANCE["add_to_wishlist"]
    if row.get("emi_calculated"):
        return RELEVANCE["calculate_emi"]
    if row.get("viewed_inspection"):
        return RELEVANCE["view_inspection_report"]
    if row.get("compared"):
        return RELEVANCE["compare_car"]
    if row.get("viewed_gallery"):
        return RELEVANCE["view_gallery"]
    if row.get("clicked"):
        return RELEVANCE["view_item"]
    return 0


def _scores_for_rows(content, profile: np.ndarray | None,
                     car_ids: pd.Series) -> np.ndarray:
    """Score only the requested cars instead of the full catalog."""
    out = np.zeros(len(car_ids))
    if profile is None:
        return out
    idx = car_ids.map(content.id_to_idx)
    ok = idx.notna().to_numpy()
    if ok.any():
        out[ok] = np.asarray(content.matrix[idx[ok].astype(int)].dot(profile)).ravel()
    return out


def historical_recommender_scores(df: pd.DataFrame, events: pd.DataFrame,
                                  bundle: RecommenderBundle) -> pd.DataFrame:
    """Build recommender features using only events visible at each decision."""
    out = df.copy()
    out["content_score"] = 0.0
    out["session_score"] = 0.0
    out["collaborative_score"] = 0.0
    ev = events.copy()
    ev["event_timestamp"] = pd.to_datetime(ev["event_timestamp"])
    by_user = {uid: g.sort_values("event_timestamp") for uid, g in ev.groupby("user_id")}
    for sid, rows in out.groupby("session_id", sort=False):
        decision = pd.to_datetime(rows["event_timestamp"]).min()
        uid = rows["user_id"].iloc[0]
        user_events = by_user.get(uid, ev.iloc[0:0])
        visible = user_events[user_events["event_timestamp"] <= decision]
        history = visible[visible["session_id"] != sid]
        current = visible[visible["session_id"] == sid]
        long_profile = bundle.content.profile_vector(history, ref_time=decision)
        session_profile = bundle.content.profile_vector(current, ref_time=decision, decay_per_day=50.0)
        out.loc[rows.index, "content_score"] = _scores_for_rows(
            bundle.content, long_profile, rows["car_id"]
        )
        out.loc[rows.index, "session_score"] = _scores_for_rows(
            bundle.content, session_profile, rows["car_id"]
        )
        if bundle.collab is not None and uid in bundle.collab.uidx:
            out.loc[rows.index, "collaborative_score"] = bundle.collab.score_user(
                uid, list(rows["car_id"])
            )
    return out


def build_ranking_dataset(cfg: Config) -> pd.DataFrame:
    """Impressions -> grouped ranking rows with hybrid scores as features."""
    raw = cfg.raw_data
    imp = pd.read_parquet(raw / "impressions.parquet")
    cars = pd.read_parquet(raw / "cars.parquet")
    users = pd.read_parquet(raw / "users.parquet")
    events = pd.read_parquet(raw / "events.parquet")
    booking_ds = pd.read_parquet(cfg.processed_data / "booking_dataset.parquet")

    from driveintent.models import classifiers
    from driveintent.models import price as price_model

    df = imp.merge(cars, on="car_id", how="left")
    df = df.merge(users[["user_id", "maximum_budget", "preferred_makes",
                         "preferred_body_types", "preferred_transmissions",
                         "home_city"]], on="user_id", how="left")
    df["relevance"] = df.apply(relevance_from_impression, axis=1)

    entry = pd.to_datetime(df["inventory_entry_date"])
    df["vehicle_age"] = (entry.dt.year - df["manufacturing_year"]).clip(lower=0)
    df["inventory_age"] = ((pd.to_datetime(df["event_timestamp"]) - entry)
                           .dt.days.clip(lower=0))
    df["price_budget_gap"] = (df["listed_price"] - df["maximum_budget"]) / df["maximum_budget"]
    df["brand_match"] = (df["make"] == df["preferred_makes"]).astype(int)
    df["body_match"] = (df["body_type"] == df["preferred_body_types"]).astype(int)
    df["trans_match"] = (df["transmission"] == df["preferred_transmissions"]).astype(int)
    df["observation_date"] = pd.to_datetime(df["event_timestamp"])
    from driveintent.features.build import attach_market_context
    df = attach_market_context(df, cars, events, "observation_date")

    # hybrid recommender scores: content similarity of car to user long-term profile
    bundle = RecommenderBundle.load(cfg)
    df = historical_recommender_scores(df, events, bundle)

    # model-based features
    pred = price_model.predict(cfg, df)
    df["predicted_fair_price"] = pred["predicted_fair_price"]
    df["deal_score"] = (df["predicted_fair_price"] - df["listed_price"]) / df["predicted_fair_price"]
    df["booking_probability"] = classifiers.predict_proba(
        cfg, "booking",
        booking_ds.merge(df[["session_id", "car_id"]], on=["session_id", "car_id"])
    ) if False else 0.0  # avoided double merge; fill below
    bp = classifiers.predict_proba(cfg, "booking", booking_ds)
    bp_map = pd.Series(bp, index=pd.MultiIndex.from_frame(
        booking_ds[["session_id", "car_id"]]))
    df["booking_probability"] = (pd.MultiIndex.from_frame(df[["session_id", "car_id"]])
                                 .map(bp_map).fillna(float(np.median(bp))))

    props = np.array(cfg.get("position_bias", "propensities"))
    pos = df["list_position"].clip(1, len(props)).astype(int) - 1
    df["list_position_propensity"] = props[pos]
    wmax = cfg.get("recommendation", "maximum_propensity_weight")
    df["ipw"] = np.minimum(1.0 / df["list_position_propensity"], wmax)
    for c in ("finance_available", "delivery_available"):
        df[c] = df[c].astype(int)
    return df


def _rank_pool(df: pd.DataFrame, with_label: bool = True, with_weight: bool = True) -> Pool:
    d = df.sort_values("session_id").reset_index(drop=True)
    return Pool(d[RANK_FEATURES],
                label=d["relevance"] if with_label else None,
                group_id=d["session_id"].astype("category").cat.codes.to_numpy(),
                weight=d["ipw"] if with_weight else None), d


def train_ranker(cfg: Config) -> dict:
    df = build_ranking_dataset(cfg)
    df.to_parquet(cfg.processed_data / "ranking_dataset.parquet", index=False)
    train_df, val_df, test_df = temporal_split(df, cfg)
    if len(val_df) < 100:
        cut = max(len(train_df) // 5, 100)
        train_df, val_df = train_df.iloc[:-cut], train_df.iloc[-cut:]
    p = cfg.get("models", "ranker")
    tr_pool, _ = _rank_pool(train_df)
    va_pool, _ = _rank_pool(val_df)
    ranker = CatBoostRanker(iterations=p["iterations"], learning_rate=p["learning_rate"],
                            depth=p["depth"], loss_function="QueryRMSE",
                            random_seed=cfg.seed, verbose=False, allow_writing_files=False)
    ranker.fit(tr_pool, eval_set=va_pool)

    metrics = evaluate_rankers(cfg, ranker, test_df)
    registry.save_model(cfg, "ranking", "ranker", ranker, RANK_FEATURES, [],
                        metrics=metrics.get("ranker", {}))
    (cfg.artifacts / "metrics" / "ranking_metrics.json").write_text(
        json.dumps(metrics, indent=2, default=str))
    return metrics


# ---------------------- offline ranking evaluation --------------------------
def _dcg(rels: np.ndarray, k: int) -> float:
    rels = rels[:k]
    return float(((2 ** rels - 1) / np.log2(np.arange(2, len(rels) + 2))).sum())


def ndcg_at_k(rels_sorted: np.ndarray, k: int) -> float:
    ideal = np.sort(rels_sorted)[::-1]
    idcg = _dcg(ideal, k)
    return _dcg(rels_sorted, k) / idcg if idcg > 0 else 0.0


def ranking_metrics_for_scores(df: pd.DataFrame, score_col: str,
                               cars: pd.DataFrame | None = None) -> dict:
    ndcg5, ndcg10, rec5, rec10, hit5, hit10, mrr, ap10 = [], [], [], [], [], [], [], []
    shown: set[str] = set()
    brand_conc = []
    make_map = cars.set_index("car_id")["make"] if cars is not None else None
    for _, g in df.groupby("session_id"):
        g = g.sort_values(score_col, ascending=False)
        rels = g["relevance"].to_numpy(dtype=float)
        n_rel = (rels > 0).sum()
        if n_rel == 0:
            continue
        ndcg5.append(ndcg_at_k(rels, 5)); ndcg10.append(ndcg_at_k(rels, 10))
        rec5.append((rels[:5] > 0).sum() / n_rel); rec10.append((rels[:10] > 0).sum() / n_rel)
        hit5.append(float((rels[:5] > 0).any())); hit10.append(float((rels[:10] > 0).any()))
        pos = np.argmax(rels > 0)
        mrr.append(1.0 / (pos + 1))
        prec = [(rels[:i + 1] > 0).mean() for i in range(min(10, len(rels))) if rels[i] > 0]
        ap10.append(float(np.mean(prec)) if prec else 0.0)
        top10 = g.head(10)["car_id"]
        shown.update(top10)
        if make_map is not None:
            makes = top10.map(make_map).dropna()
            if len(makes):
                brand_conc.append(makes.value_counts(normalize=True).iloc[0])
    total_cars = df["car_id"].nunique()
    return dict(
        ndcg_at_5=float(np.mean(ndcg5)), ndcg_at_10=float(np.mean(ndcg10)),
        map_at_10=float(np.mean(ap10)),
        recall_at_5=float(np.mean(rec5)), recall_at_10=float(np.mean(rec10)),
        hitrate_at_5=float(np.mean(hit5)), hitrate_at_10=float(np.mean(hit10)),
        mrr=float(np.mean(mrr)),
        coverage_at_10=float(len(shown) / max(total_cars, 1)),
        brand_concentration=float(np.mean(brand_conc)) if brand_conc else float("nan"),
        n_sessions=len(ndcg10))


def evaluate_rankers(cfg: Config, ranker: CatBoostRanker, test_df: pd.DataFrame) -> dict:
    cars = pd.read_parquet(cfg.raw_data / "cars.parquet")
    d = test_df.copy()
    pool, sorted_df = _rank_pool(d, with_label=False, with_weight=False)
    sorted_df["ranker_score"] = ranker.predict(pool)
    d = sorted_df
    # Catalog popularity is fixed before the test interaction. Never derive a
    # baseline from test clicks, which would leak the labels being evaluated.
    d["popularity_score"] = d["model_popularity"].astype(float)
    d["hybrid_score"] = (0.4 * _z(d["content_score"]) + 0.3 * _z(d["collaborative_score"])
                         + 0.3 * _z(d["session_score"]))
    out = {}
    for name, col in [("popularity", "popularity_score"), ("content_only", "content_score"),
                      ("collaborative_only", "collaborative_score"),
                      ("session_only", "session_score"), ("weighted_hybrid", "hybrid_score"),
                      ("ranker", "ranker_score")]:
        out[name] = ranking_metrics_for_scores(d, col, cars)
    # multi-objective reranker on top of ranker score
    d["mo_score"] = multi_objective_score(cfg, d)
    out["multi_objective_reranker"] = ranking_metrics_for_scores(d, "mo_score", cars)
    return out


def _z(s: pd.Series) -> pd.Series:
    sd = s.std()
    return (s - s.mean()) / sd if sd > 0 else s * 0


# ---------------------- multi-objective reranking ---------------------------
def multi_objective_score(cfg: Config, d: pd.DataFrame,
                          exposure_counts: pd.Series | None = None) -> pd.Series:
    w = cfg.get("reranking")
    ranker = _z(d["ranker_score"]) if "ranker_score" in d else _z(d.get("content_score", pd.Series(0, index=d.index)))
    booking = _z(d["booking_probability"])
    deal = _z(d["deal_score"].clip(-0.5, 0.5))
    inv_age = d.get("inventory_age", pd.Series(0, index=d.index))
    inv_priority = (inv_age / 60.0).clip(upper=1.0)
    # gate inventory boost: relevance above threshold and quality floor
    rel_gate = (ranker.rank(pct=True) >= w["minimum_relevance_for_inventory_boost"] * 0.5)
    q_gate = d["inspection_score"] >= 65
    inv_priority = inv_priority.where(rel_gate & q_gate, 0.0)
    quality = _z(d["inspection_score"])
    constraint_pen = d.get("hard_violations", (d["price_budget_gap"] > 0.10).astype(int))
    if exposure_counts is not None:
        exp_pen = np.log1p(exposure_counts.reindex(d["car_id"]).fillna(0).to_numpy())
    else:
        exp_pen = 0.0
    return (w["ranker_weight"] * ranker
            + w["booking_weight"] * booking
            + w["deal_weight"] * deal
            + w["inventory_weight"] * inv_priority
            + w["quality_weight"] * quality
            - w["constraint_penalty_weight"] * constraint_pen
            - w["exposure_penalty_weight"] * exp_pen)


def mmr_diversify(cand: pd.DataFrame, score_col: str, content, k: int,
                  lam: float) -> pd.DataFrame:
    """Greedy maximal-marginal-relevance selection."""
    remaining = cand.sort_values(score_col, ascending=False).reset_index(drop=True)
    if len(remaining) <= 1:
        return remaining.head(k)
    s = _z(remaining[score_col]).to_numpy()
    selected: list[int] = [0]
    while len(selected) < min(k, len(remaining)):
        best, best_val = None, -np.inf
        for i in range(len(remaining)):
            if i in selected:
                continue
            max_sim = max(content.item_similarity(remaining.loc[i, "car_id"],
                                                  remaining.loc[j, "car_id"])
                          for j in selected)
            val = lam * s[i] - (1 - lam) * max_sim
            if val > best_val:
                best, best_val = i, val
        selected.append(best)
    return remaining.iloc[selected].reset_index(drop=True)


def entropy_adaptive_lambda(cfg: Config, session_entropy: float) -> float:
    w = cfg.get("reranking")
    if session_entropy < 0.35:
        return w["mmr_lambda_low_entropy"]
    if session_entropy < 0.65:
        return w["mmr_lambda_mid_entropy"]
    return w["mmr_lambda_high_entropy"]


# ---------------------- explanations ----------------------------------------
def explain_recommendation(row: pd.Series, profile, budget: float | None) -> list[str]:
    reasons = []
    if row.get("session_score", 0) > 0.5:
        dims = profile.session.get("body_type", {}) if profile else {}
        top = max(dims, key=dims.get) if dims else row["body_type"]
        reasons.append(f"Matches your recent {row['transmission'].lower()} {top} searches.")
    elif row.get("content_score", 0) > 0.5:
        reasons.append("Similar to cars you recently shortlisted.")
    if budget and row["listed_price"] <= budget:
        reasons.append("Within your inferred maximum budget.")
    ds = row.get("deal_score", 0)
    if ds > 0.02:
        reasons.append(f"Priced {abs(ds) * 100:.1f}% below its estimated fair value.")
    if row.get("inspection_score", 0) >= 85:
        reasons.append(f"Strong inspection score of {row['inspection_score']:.0f}/100.")
    if row.get("booking_probability", 0) > 0.15:
        reasons.append("High predicted interest from similar buyers.")
    if "popularity" in str(row.get("candidate_source", "")):
        reasons.append(f"Popular among buyers in {row['city']}.")
    if profile is not None and profile.session_entropy > 0.65:
        reasons.append("Recommended as a diverse option because your current preferences are still exploratory.")
    return reasons[:4] or ["Matches your inferred preferences."]


def recommend_for_user(cfg: Config, user_id: str, session_id: str | None = None,
                       limit: int = 10, diversity_level: str | None = None) -> dict:
    """End-to-end serving path: candidates -> scores -> multi-objective -> MMR."""
    from driveintent.models import classifiers
    from driveintent.models import price as price_model
    from driveintent.models.recommender import generate_candidates, popularity_by_segment

    raw = cfg.raw_data
    users = pd.read_parquet(raw / "users.parquet")
    cars = pd.read_parquet(raw / "cars.parquet")
    events = pd.read_parquet(raw / "events.parquet")
    bundle = RecommenderBundle.load(cfg)
    user_events = events[events["user_id"] == user_id]
    urow = users[users["user_id"] == user_id]
    urow = urow.iloc[0] if len(urow) else None
    if session_id is None and len(user_events):
        session_id = user_events.sort_values("event_timestamp")["session_id"].iloc[-1]

    exit_dt = pd.to_datetime(cars["inventory_exit_date"])
    available = set(cars.loc[exit_dt.isna(), "car_id"])
    pop = popularity_by_segment(events, cars)
    cand = generate_candidates(bundle, user_events, urow, session_id, available, pop, cfg)
    prof = cand.attrs.get("profile") if len(cand) else None
    if cand.empty:
        return dict(user_id=user_id, session_id=session_id, intent_summary={},
                    recommendations=[])

    entry = pd.to_datetime(cand["inventory_entry_date"])
    cand["vehicle_age"] = (entry.dt.year - cand["manufacturing_year"]).clip(lower=0)
    cand["inventory_age"] = (pd.Timestamp.now().normalize() - entry).dt.days.clip(lower=0)
    pred = price_model.predict(cfg, cand)
    cand["predicted_fair_price"] = pred["predicted_fair_price"]
    cand["deal_score"] = (cand["predicted_fair_price"] - cand["listed_price"]) / cand["predicted_fair_price"]

    budget = None
    if urow is not None:
        budget = float(urow["maximum_budget"])
    if prof and "max_price" in prof.hard_constraints:
        budget = prof.hard_constraints["max_price"]
    cand["price_budget_gap"] = ((cand["listed_price"] - budget) / budget) if budget else 0.0
    cand["brand_match"] = (cand["make"] == (urow["preferred_makes"] if urow is not None else "")).astype(int)
    cand["body_match"] = (cand["body_type"] == (urow["preferred_body_types"] if urow is not None else "")).astype(int)
    cand["trans_match"] = (cand["transmission"] == (urow["preferred_transmissions"] if urow is not None else "")).astype(int)
    cand["market_demand_index"] = 1.0
    cand["list_position_propensity"] = 1.0
    for c in ("finance_available", "delivery_available"):
        cand[c] = cand[c].astype(int)

    # booking probability with serve-time defaults for session features
    brow = cand.copy()
    defaults = dict(prior_sessions=len(user_events["session_id"].unique()),
                    prior_booking_rate=0.0, is_returning_user=True,
                    session_duration_s=300, session_searches=1, session_filters=1,
                    session_events=10, session_sequence_number=1,
                    device_category="mobile", source="(direct)", medium="(none)",
                    campaign_id="NA", list_position=1, emi_budget_gap=0.0,
                    fuel_match=0, city_match=1, age_over_tolerance=0.0,
                    km_over_tolerance=0.0, hard_violations=0)
    if urow is not None:
        for k in ("purchase_urgency", "price_sensitivity", "quality_sensitivity",
                  "finance_interest", "brand_loyalty", "first_time_buyer_probability"):
            defaults[k] = float(urow[k])
    for k, v in defaults.items():
        if k not in brow.columns:
            brow[k] = v
    cand["booking_probability"] = classifiers.predict_proba(cfg, "booking", brow)
    try:
        cand["sellthrough_probability"] = classifiers.predict_proba(cfg, "sellthrough", _sell_rows(cand))
    except Exception:
        cand["sellthrough_probability"] = np.nan

    ranker, _ = registry.load_model(cfg, "ranking", "ranker")
    cand["ranker_score"] = ranker.predict(cand[RANK_FEATURES])
    cand["mo_score"] = multi_objective_score(cfg, cand)

    lam = {"low": 0.9, "medium": 0.7, "high": 0.5}.get(
        diversity_level or "", entropy_adaptive_lambda(cfg, prof.session_entropy if prof else 0.7))
    final = mmr_diversify(cand, "mo_score", bundle.content, k=limit, lam=lam)

    recs = []
    for rank, (_, r) in enumerate(final.iterrows(), start=1):
        recs.append(dict(
            car_id=r["car_id"], rank=rank, score=round(float(r["mo_score"]), 3),
            make=r["make"], model=r["model"], body_type=r["body_type"],
            transmission=r["transmission"], fuel_type=r["fuel_type"], city=r["city"],
            listed_price=float(r["listed_price"]),
            predicted_fair_price=float(r["predicted_fair_price"]),
            deal_score=round(float(r["deal_score"]), 4),
            booking_probability=round(float(r["booking_probability"]), 4),
            inspection_score=float(r["inspection_score"]),
            candidate_source=r["candidate_source"],
            reasons=explain_recommendation(r, prof, budget)))
    intent = {}
    if prof is not None:
        body = prof.session.get("body_type") or prof.long_term.get("body_type") or {}
        trans = prof.session.get("transmission") or prof.long_term.get("transmission") or {}
        intent = dict(body_type=max(body, key=body.get) if body else None,
                      transmission=max(trans, key=trans.get) if trans else None,
                      maximum_budget=budget, confidence=round(prof.confidence, 3),
                      entropy=round(prof.session_entropy, 3),
                      hard_constraints=prof.hard_constraints,
                      soft_preferences=prof.soft_preferences)
    return dict(user_id=user_id, session_id=session_id, intent_summary=intent,
                model_version="ranker_v1", generated_at=datetime.now().isoformat(),
                recommendation_id=str(uuid.uuid4()), recommendations=recs)


def _sell_rows(cand: pd.DataFrame) -> pd.DataFrame:
    d = cand.copy()
    fill = dict(snapshot_age=d["inventory_age"], owner_count=d.get("owner_count", 1),
                number_of_features=d.get("number_of_features", 8),
                views=5, clicks=1, inspection_views=0, wishlists=0, bookings_cnt=0,
                click_rate=0.2, booking_rate=0.0, local_supply_index=0.2,
                comparable_inventory=20)
    for k, v in fill.items():
        if k not in d.columns:
            d[k] = v
    return d
