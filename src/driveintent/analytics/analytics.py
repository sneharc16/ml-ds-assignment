"""Business analytics: campaign lead quality, budget optimization simulation,
inventory opportunities, A/B experiment design, ablation study.

The budget optimizer is a scenario simulation on synthetic diminishing-return
curves, NOT a production Google Ads bidding system.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
from scipy import optimize, stats

from driveintent.config import Config
from driveintent.data import load_database as db

RECONDITIONING_COST = 15000.0
OPERATING_COST = 8000.0


# --------------------------- campaigns --------------------------------------
def campaign_performance(cfg: Config) -> pd.DataFrame:
    perf = db.run_sql_file(cfg, "campaign_quality.sql")
    # expected lead value = P(booking) * P(purchase|booking) * expected contribution margin
    cars = db.query(cfg, "SELECT AVG(transaction_price - acquisition_price) AS m FROM cars WHERE sold_flag")
    margin = float(cars["m"].iloc[0]) - RECONDITIONING_COST - OPERATING_COST
    p_purchase_given_booking = (perf["purchases"].sum() / max(perf["bookings"].sum(), 1))
    perf["p_booking_per_lead"] = perf["bookings"] / perf["leads"].clip(lower=1)
    perf["expected_lead_value"] = (perf["p_booking_per_lead"]
                                   * p_purchase_given_booking * margin).round(0)
    perf["predicted_lead_value_total"] = (perf["leads"] * perf["expected_lead_value"]).round(0)
    perf["expected_roas"] = (perf["predicted_lead_value_total"]
                             / perf["click_cost"].clip(lower=1)).round(3)
    return perf


def inventory_match_score(cfg: Config) -> pd.DataFrame:
    """Per-campaign: could current inventory satisfy the demand it generates?"""
    sql = """
    WITH sess_body AS (
        SELECT s.campaign_id, e.session_id, f.filter_value AS body_type, e.city
        FROM sessions s
        JOIN events e ON e.session_id = s.session_id AND e.event_name = 'view_search_results'
        JOIN events f ON f.session_id = s.session_id
             AND f.event_name = 'apply_filter' AND f.filter_name = 'body_type'
    ),
    supply AS (
        SELECT city, body_type, COUNT(*) AS live FROM cars WHERE NOT sold_flag GROUP BY 1,2
    )
    SELECT sb.campaign_id,
           COUNT(*) AS demand_sessions,
           AVG(CASE WHEN COALESCE(su.live,0) >= 3 THEN 1.0 ELSE 0.0 END) AS satisfiable_share,
           MEDIAN(COALESCE(su.live,0)) AS median_eligible_cars
    FROM sess_body sb LEFT JOIN supply su
      ON su.city = sb.city AND su.body_type = sb.body_type
    GROUP BY 1
    """
    return db.query(cfg, sql)


def budget_optimizer(cfg: Config, total_budget: float | None = None) -> pd.DataFrame:
    """Constrained allocation over log diminishing-return curves fitted per campaign."""
    perf = campaign_performance(cfg)
    perf = perf[perf["click_cost"] > 0].reset_index(drop=True)
    if perf.empty:
        return pd.DataFrame()
    spend = perf["click_cost"].to_numpy(float)
    value = perf["predicted_lead_value_total"].clip(lower=1).to_numpy(float)
    # value_c(x) = a_c * log(1 + b_c x), calibrated so curve passes through observed point
    b = 1.0 / np.clip(spend, 1, None)
    a = value / np.log1p(b * spend)
    B = float(total_budget or spend.sum())
    lo = 0.3 * spend
    hi = 3.0 * spend
    if B < lo.sum() or B > hi.sum():
        raise ValueError(
            f"total_budget must be between {lo.sum():.0f} and {hi.sum():.0f} "
            "under the campaign change constraints"
        )

    def neg_total(x):
        return -np.sum(a * np.log1p(b * x))

    res = optimize.minimize(neg_total, x0=spend, method="SLSQP",
                            bounds=list(zip(lo, hi)),
                            constraints=[{"type": "eq", "fun": lambda x: B - x.sum()}])
    if not res.success:
        raise RuntimeError(f"budget optimization failed: {res.message}")
    x = res.x
    out = pd.DataFrame(dict(
        campaign=perf["campaign_name"], campaign_id=perf["campaign_id"],
        current_budget=spend.round(0), recommended_budget=x.round(0)))
    out["absolute_change"] = (out["recommended_budget"] - out["current_budget"]).round(0)
    out["percentage_change"] = (100 * out["absolute_change"] / out["current_budget"]).round(1)
    leads_per_value = perf["leads"].to_numpy(float) / np.clip(value, 1, None)
    inc_value = a * np.log1p(b * x) - value
    out["expected_incremental_value"] = inc_value.round(0)
    out["expected_incremental_leads"] = (inc_value * leads_per_value).round(1)
    out.attrs["note"] = ("Scenario simulation based on synthetic response curves, "
                         "not a production Google Ads bidding system.")
    return out


# --------------------------- inventory --------------------------------------
def inventory_opportunities(cfg: Config, city: str | None = None,
                            body_type: str | None = None,
                            minimum_gap: float = 0.0) -> pd.DataFrame:
    gap = db.run_sql_file(cfg, "demand_supply_gap.sql")
    if city:
        gap = gap[gap["city"] == city]
    if body_type:
        gap = gap[gap["body_type"] == body_type]
    return gap[gap["demand_supply_gap"] >= minimum_gap].reset_index(drop=True)


def price_review_candidates(cfg: Config, top_n: int = 25) -> pd.DataFrame:
    """High-engagement, low-conversion, aging cars = price-review candidates."""
    sql = """
    WITH eng AS (
        SELECT car_id,
               SUM(CASE WHEN event_name='view_item' THEN 1 ELSE 0 END) AS views,
               SUM(CASE WHEN event_name='booking_complete' THEN 1 ELSE 0 END) AS bookings
        FROM events WHERE car_id IS NOT NULL GROUP BY 1
    )
    SELECT c.car_id, c.make, c.model, c.city, c.listed_price, c.inspection_score,
           c.days_in_inventory, e.views, e.bookings
    FROM cars c JOIN eng e USING (car_id)
    WHERE NOT c.sold_flag AND e.views >= 3 AND e.bookings = 0
    ORDER BY e.views DESC, c.days_in_inventory DESC
    """
    return db.query(cfg, sql).head(top_n)


# --------------------------- experimentation --------------------------------
def sample_size_two_proportions(baseline: float, mde_relative: float,
                                alpha: float = 0.05, power: float = 0.8) -> int:
    """Per-arm sample size for detecting a relative lift in conversion."""
    if not 0 < baseline < 1:
        raise ValueError("baseline must be between 0 and 1")
    if mde_relative <= 0:
        raise ValueError("mde_relative must be positive")
    if not 0 < alpha < 1 or not 0 < power < 1:
        raise ValueError("alpha and power must be between 0 and 1")
    p1 = baseline
    p2 = baseline * (1 + mde_relative)
    if p2 >= 1:
        raise ValueError("baseline and relative MDE imply a conversion rate >= 1")
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    pbar = (p1 + p2) / 2
    n = ((z_a * math.sqrt(2 * pbar * (1 - pbar))
          + z_b * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2) / (p2 - p1) ** 2
    return int(math.ceil(n))


def simulate_experiment(cfg: Config, n_per_arm: int = 4000,
                        true_lift: float = 0.15, baseline: float | None = None,
                        seed: int | None = None) -> dict:
    """Simulate a user-randomized A/B test: relevance-only vs intent-aware ranking."""
    if n_per_arm < 2:
        raise ValueError("n_per_arm must be at least 2")
    rng = np.random.default_rng(seed if seed is not None else cfg.seed)
    if baseline is None:
        r = db.query(cfg, """
            SELECT AVG(b) FROM (SELECT user_id,
                MAX(CASE WHEN event_name='booking_complete' THEN 1 ELSE 0 END) AS b
                FROM events GROUP BY user_id)""")
        baseline = float(r.iloc[0, 0])
    if not 0 < baseline < 1:
        raise ValueError("baseline must be between 0 and 1")
    if true_lift <= 0 or baseline * (1 + true_lift) >= 1:
        raise ValueError("true_lift must be positive and imply a treatment rate below 1")
    ctrl = rng.binomial(1, baseline, n_per_arm)
    trt = rng.binomial(1, min(baseline * (1 + true_lift), 0.99), n_per_arm)
    p1, p2 = ctrl.mean(), trt.mean()
    se = math.sqrt(p1 * (1 - p1) / n_per_arm + p2 * (1 - p2) / n_per_arm)
    z = (p2 - p1) / max(se, 1e-12)
    pval = 2 * (1 - stats.norm.cdf(abs(z)))
    ci = ((p2 - p1) - 1.96 * se, (p2 - p1) + 1.96 * se)
    p_alt = baseline * (1 + true_lift)
    delta = p_alt - baseline
    pooled = (baseline + p_alt) / 2
    se_null = math.sqrt(2 * pooled * (1 - pooled) / n_per_arm)
    se_alt = math.sqrt((baseline * (1 - baseline) + p_alt * (1 - p_alt)) / n_per_arm)
    critical = stats.norm.ppf(0.975) * se_null
    attained_power = (stats.norm.cdf((-critical - delta) / se_alt)
                      + 1 - stats.norm.cdf((critical - delta) / se_alt))
    return dict(
        randomization_unit="user_id (session-level randomization risks the same user "
                           "seeing both rankings, contaminating treatment)",
        control_conversion=round(float(p1), 4), treatment_conversion=round(float(p2), 4),
        absolute_lift=round(float(p2 - p1), 4),
        relative_lift=round(float((p2 - p1) / max(p1, 1e-9)), 4),
        ci_95=[round(ci[0], 4), round(ci[1], 4)], p_value=round(float(pval), 5),
        approx_power=round(float(attained_power), 3),
        interpretation=("Statistically significant" if pval < 0.05 else "Not significant")
                       + "; practical significance depends on lead-handling economics.")


# --------------------------- ablation ---------------------------------------
def ablation_study(cfg: Config) -> pd.DataFrame:
    """Recommendation ablation: drop one signal group at a time from the ranker."""
    from catboost import CatBoostRanker

    from driveintent.models.common import temporal_split
    from driveintent.models.ranking import RANK_FEATURES, ranking_metrics_for_scores
    df = pd.read_parquet(cfg.processed_data / "ranking_dataset.parquet")
    cars = pd.read_parquet(cfg.raw_data / "cars.parquet")
    train_df, val_df, test_df = temporal_split(df, cfg)
    if len(test_df) < 100:
        test_df = df.tail(max(len(df) // 5, 100))
        train_df = df.iloc[:len(df) - len(test_df)]
    ablations = {
        "full_model": [],
        "no_session_intent": ["session_score"],
        "no_collaborative": ["collaborative_score"],
        "no_price_deal": ["deal_score"],
        "no_market_context": ["market_demand_index"],
    }
    p = cfg.get("models", "ranker")
    rows = []
    for name, drop in ablations.items():
        feats = [f for f in RANK_FEATURES if f not in drop]
        d = train_df.sort_values("session_id").reset_index(drop=True)
        from catboost import Pool
        tr = Pool(d[feats], label=d["relevance"],
                  group_id=d["session_id"].astype("category").cat.codes.to_numpy(),
                  weight=d["ipw"])
        rk = CatBoostRanker(iterations=min(p["iterations"], 150),
                            learning_rate=p["learning_rate"], depth=p["depth"],
                            loss_function="QueryRMSE", random_seed=cfg.seed, verbose=False, allow_writing_files=False)
        rk.fit(tr)
        te = test_df.copy()
        te["score"] = rk.predict(te[feats])
        m = ranking_metrics_for_scores(te, "score", cars)
        rows.append(dict(ablation=name, ndcg_at_10=round(m["ndcg_at_10"], 4),
                         recall_at_10=round(m["recall_at_10"], 4),
                         coverage_at_10=round(m["coverage_at_10"], 4),
                         brand_concentration=round(m["brand_concentration"], 4)))
    out = pd.DataFrame(rows)
    out.to_csv(cfg.artifacts / "reports" / "ablation_results.csv", index=False)
    return out


def export_reports(cfg: Config) -> None:
    rep = cfg.artifacts / "reports"
    campaign_performance(cfg).to_csv(rep / "campaign_performance.csv", index=False)
    inventory_opportunities(cfg).to_csv(rep / "inventory_opportunities.csv", index=False)
    price_review_candidates(cfg).to_csv(rep / "price_review_candidates.csv", index=False)
    (rep / "experiment_simulation.json").write_text(
        json.dumps(simulate_experiment(cfg), indent=2))
