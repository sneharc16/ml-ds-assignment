"""Synthetic data generation for DriveIntent.

Generates a statistically coherent Indian used-car marketplace:
cars (with a latent fair-price DGP), users (persistent latent preferences),
campaigns (heterogeneous traffic quality), sessions (with session-intent
drift) and a GA4-style event stream with explicit position bias.

Latent variables (latent_fair_price, user/session latent preference vectors)
drive behaviour generation but are NEVER exposed as model features. They are
written to data/raw/_latents_*.parquet purely for documentation/EDA.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from driveintent.config import Config

# --------------------------------------------------------------------------
# Catalog: make -> model -> (body_type, fuels, transmissions, base_new_price_lakh, popularity)
# --------------------------------------------------------------------------
CATALOG: list[dict[str, Any]] = [
    # hatchbacks
    dict(make="Maruti Suzuki", model="Swift", body="Hatchback", fuels=["Petrol", "CNG"], trans=["Manual", "Automatic"], base=8.0, pop=0.95),
    dict(make="Maruti Suzuki", model="Baleno", body="Hatchback", fuels=["Petrol", "CNG"], trans=["Manual", "Automatic"], base=9.0, pop=0.90),
    dict(make="Hyundai", model="i10", body="Hatchback", fuels=["Petrol", "CNG"], trans=["Manual", "Automatic"], base=7.0, pop=0.80),
    dict(make="Hyundai", model="i20", body="Hatchback", fuels=["Petrol", "Diesel"], trans=["Manual", "Automatic"], base=9.5, pop=0.85),
    dict(make="Tata", model="Tiago", body="Hatchback", fuels=["Petrol", "CNG"], trans=["Manual", "Automatic"], base=7.0, pop=0.75),
    dict(make="Renault", model="Kwid", body="Hatchback", fuels=["Petrol"], trans=["Manual", "Automatic"], base=5.5, pop=0.60),
    dict(make="Volkswagen", model="Polo", body="Hatchback", fuels=["Petrol", "Diesel"], trans=["Manual", "Automatic"], base=9.0, pop=0.65),
    # sedans
    dict(make="Honda", model="City", body="Sedan", fuels=["Petrol", "Diesel"], trans=["Manual", "Automatic"], base=13.5, pop=0.85),
    dict(make="Hyundai", model="Verna", body="Sedan", fuels=["Petrol", "Diesel"], trans=["Manual", "Automatic"], base=13.0, pop=0.75),
    dict(make="Maruti Suzuki", model="Ciaz", body="Sedan", fuels=["Petrol"], trans=["Manual", "Automatic"], base=11.0, pop=0.60),
    dict(make="Skoda", model="Slavia", body="Sedan", fuels=["Petrol"], trans=["Manual", "Automatic"], base=13.5, pop=0.60),
    dict(make="Volkswagen", model="Virtus", body="Sedan", fuels=["Petrol"], trans=["Manual", "Automatic"], base=13.5, pop=0.60),
    dict(make="Tata", model="Tigor", body="Sedan", fuels=["Petrol", "CNG"], trans=["Manual", "Automatic"], base=8.0, pop=0.55),
    # compact SUVs
    dict(make="Tata", model="Nexon", body="Compact SUV", fuels=["Petrol", "Diesel", "Electric"], trans=["Manual", "Automatic"], base=12.0, pop=0.92),
    dict(make="Hyundai", model="Venue", body="Compact SUV", fuels=["Petrol", "Diesel"], trans=["Manual", "Automatic"], base=11.0, pop=0.85),
    dict(make="Kia", model="Sonet", body="Compact SUV", fuels=["Petrol", "Diesel"], trans=["Manual", "Automatic"], base=11.5, pop=0.80),
    dict(make="Maruti Suzuki", model="Brezza", body="Compact SUV", fuels=["Petrol", "CNG"], trans=["Manual", "Automatic"], base=11.5, pop=0.90),
    dict(make="Mahindra", model="XUV300", body="Compact SUV", fuels=["Petrol", "Diesel"], trans=["Manual", "Automatic"], base=11.0, pop=0.70),
    dict(make="Nissan", model="Magnite", body="Compact SUV", fuels=["Petrol"], trans=["Manual", "Automatic"], base=8.5, pop=0.60),
    dict(make="Renault", model="Kiger", body="Compact SUV", fuels=["Petrol"], trans=["Manual", "Automatic"], base=8.5, pop=0.55),
    dict(make="Ford", model="EcoSport", body="Compact SUV", fuels=["Petrol", "Diesel"], trans=["Manual", "Automatic"], base=10.0, pop=0.50),
    # SUVs
    dict(make="Hyundai", model="Creta", body="SUV", fuels=["Petrol", "Diesel"], trans=["Manual", "Automatic"], base=16.5, pop=0.95),
    dict(make="Kia", model="Seltos", body="SUV", fuels=["Petrol", "Diesel"], trans=["Manual", "Automatic"], base=16.0, pop=0.88),
    dict(make="Mahindra", model="XUV700", body="SUV", fuels=["Petrol", "Diesel"], trans=["Manual", "Automatic"], base=20.0, pop=0.85),
    dict(make="Mahindra", model="Scorpio", body="SUV", fuels=["Diesel"], trans=["Manual", "Automatic"], base=17.0, pop=0.80),
    dict(make="Tata", model="Harrier", body="SUV", fuels=["Diesel"], trans=["Manual", "Automatic"], base=19.0, pop=0.72),
    dict(make="MG", model="Hector", body="SUV", fuels=["Petrol", "Diesel"], trans=["Manual", "Automatic"], base=18.0, pop=0.65),
    dict(make="Toyota", model="Fortuner", body="SUV", fuels=["Diesel", "Petrol"], trans=["Manual", "Automatic"], base=38.0, pop=0.78),
    dict(make="Jeep", model="Compass", body="SUV", fuels=["Petrol", "Diesel"], trans=["Manual", "Automatic"], base=22.0, pop=0.60),
    # MPVs
    dict(make="Maruti Suzuki", model="Ertiga", body="MPV", fuels=["Petrol", "CNG"], trans=["Manual", "Automatic"], base=11.0, pop=0.85),
    dict(make="Toyota", model="Innova", body="MPV", fuels=["Diesel", "Petrol"], trans=["Manual", "Automatic"], base=22.0, pop=0.90),
    dict(make="Kia", model="Carens", body="MPV", fuels=["Petrol", "Diesel"], trans=["Manual", "Automatic"], base=13.0, pop=0.70),
    dict(make="Renault", model="Triber", body="MPV", fuels=["Petrol"], trans=["Manual", "Automatic"], base=7.5, pop=0.55),
]

VARIANTS = ["Base", "Mid", "Top"]
VARIANT_MULT = {"Base": 0.92, "Mid": 1.00, "Top": 1.12}

# city -> (price_multiplier, automatic_pref, diesel_pref, dominant body demand)
CITIES: dict[str, dict[str, Any]] = {
    "Delhi NCR":  dict(state="Delhi", mult=1.05, at=0.45, diesel=0.20, body={"Hatchback": 0.25, "Sedan": 0.15, "Compact SUV": 0.25, "SUV": 0.25, "MPV": 0.10}, weight=0.16),
    "Mumbai":     dict(state="Maharashtra", mult=1.08, at=0.55, diesel=0.15, body={"Hatchback": 0.30, "Sedan": 0.18, "Compact SUV": 0.24, "SUV": 0.18, "MPV": 0.10}, weight=0.14),
    "Bengaluru":  dict(state="Karnataka", mult=1.10, at=0.60, diesel=0.20, body={"Hatchback": 0.22, "Sedan": 0.15, "Compact SUV": 0.28, "SUV": 0.25, "MPV": 0.10}, weight=0.14),
    "Hyderabad":  dict(state="Telangana", mult=1.04, at=0.45, diesel=0.28, body={"Hatchback": 0.22, "Sedan": 0.16, "Compact SUV": 0.25, "SUV": 0.26, "MPV": 0.11}, weight=0.10),
    "Chennai":    dict(state="Tamil Nadu", mult=1.00, at=0.40, diesel=0.25, body={"Hatchback": 0.28, "Sedan": 0.20, "Compact SUV": 0.22, "SUV": 0.20, "MPV": 0.10}, weight=0.09),
    "Pune":       dict(state="Maharashtra", mult=1.02, at=0.42, diesel=0.22, body={"Hatchback": 0.27, "Sedan": 0.15, "Compact SUV": 0.26, "SUV": 0.21, "MPV": 0.11}, weight=0.09),
    "Kolkata":    dict(state="West Bengal", mult=0.95, at=0.32, diesel=0.22, body={"Hatchback": 0.34, "Sedan": 0.20, "Compact SUV": 0.20, "SUV": 0.16, "MPV": 0.10}, weight=0.07),
    "Ahmedabad":  dict(state="Gujarat", mult=0.97, at=0.35, diesel=0.18, body={"Hatchback": 0.30, "Sedan": 0.16, "Compact SUV": 0.24, "SUV": 0.19, "MPV": 0.11}, weight=0.06),
    "Jaipur":     dict(state="Rajasthan", mult=0.94, at=0.30, diesel=0.30, body={"Hatchback": 0.28, "Sedan": 0.14, "Compact SUV": 0.24, "SUV": 0.24, "MPV": 0.10}, weight=0.05),
    "Chandigarh": dict(state="Chandigarh", mult=0.98, at=0.38, diesel=0.28, body={"Hatchback": 0.24, "Sedan": 0.16, "Compact SUV": 0.24, "SUV": 0.26, "MPV": 0.10}, weight=0.04),
    "Lucknow":    dict(state="Uttar Pradesh", mult=0.92, at=0.28, diesel=0.26, body={"Hatchback": 0.30, "Sedan": 0.16, "Compact SUV": 0.24, "SUV": 0.20, "MPV": 0.10}, weight=0.03),
    "Kochi":      dict(state="Kerala", mult=0.96, at=0.36, diesel=0.30, body={"Hatchback": 0.26, "Sedan": 0.18, "Compact SUV": 0.24, "SUV": 0.20, "MPV": 0.12}, weight=0.03),
}
CITY_NAMES = list(CITIES.keys())
CITY_WEIGHTS = np.array([CITIES[c]["weight"] for c in CITY_NAMES])
CITY_WEIGHTS = CITY_WEIGHTS / CITY_WEIGHTS.sum()

BODY_TYPES = ["Hatchback", "Sedan", "Compact SUV", "SUV", "MPV"]

CHANNELS = [
    # name, source, medium, quality (click quality), intent, inv_match, cpc, weight
    dict(channel="Google Search - Brand", source="google", medium="cpc", quality=1.30, intent=1.40, inv_match=1.10, cpc=25.0),
    dict(channel="Google Search - Generic", source="google", medium="cpc", quality=1.05, intent=1.05, inv_match=1.00, cpc=18.0),
    dict(channel="Google Display", source="google", medium="display", quality=0.70, intent=0.60, inv_match=0.90, cpc=6.0),
    dict(channel="Organic Search", source="google", medium="organic", quality=1.10, intent=1.10, inv_match=1.00, cpc=0.0),
    dict(channel="Direct", source="(direct)", medium="(none)", quality=1.15, intent=1.20, inv_match=1.00, cpc=0.0),
    dict(channel="Referral", source="partner", medium="referral", quality=0.95, intent=0.90, inv_match=1.00, cpc=8.0),
    dict(channel="Social", source="meta", medium="paid_social", quality=0.80, intent=0.70, inv_match=0.95, cpc=9.0),
    dict(channel="Retargeting", source="google", medium="retargeting", quality=1.35, intent=1.55, inv_match=1.15, cpc=14.0),
    dict(channel="Email", source="crm", medium="email", quality=1.20, intent=1.25, inv_match=1.05, cpc=1.0),
    dict(channel="Affiliate", source="affiliate", medium="affiliate", quality=0.85, intent=0.80, inv_match=0.95, cpc=10.0),
]

AGE_BANDS = ["18-24", "25-34", "35-44", "45-54", "55+"]
INCOME_BANDS = ["<5L", "5-10L", "10-20L", "20-35L", "35L+"]
FAMILY_BANDS = ["1", "2", "3-4", "5+"]
DEVICES = ["mobile", "desktop", "tablet"]

EVENT_FUNNEL = [
    "session_start", "view_home", "search", "apply_filter", "view_search_results",
    "select_item", "view_item", "view_gallery", "view_inspection_report",
    "compare_car", "calculate_emi", "view_finance_offer", "add_to_wishlist",
    "request_callback", "book_test_drive", "begin_checkout", "booking_complete",
    "purchase", "session_end",
]


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


# ==========================================================================
# Cars
# ==========================================================================
def generate_cars(cfg: Config, gen_cfg: dict, rng: np.random.Generator) -> pd.DataFrame:
    n = int(gen_cfg["n_cars"])
    start = pd.Timestamp(gen_cfg["start_date"])
    end = pd.Timestamp(gen_cfg["end_date"])
    horizon_days = (end - start).days

    pops = np.array([c["pop"] for c in CATALOG])
    idx = rng.choice(len(CATALOG), size=n, p=pops / pops.sum())
    rows = []
    for i in range(n):
        c = CATALOG[idx[i]]
        city = rng.choice(CITY_NAMES, p=CITY_WEIGHTS)
        cinfo = CITIES[city]
        variant = rng.choice(VARIANTS, p=[0.3, 0.45, 0.25])
        fuel_p = np.ones(len(c["fuels"]))
        for j, f in enumerate(c["fuels"]):
            if f == "Diesel":
                fuel_p[j] = 0.6 + cinfo["diesel"]
            elif f == "CNG":
                fuel_p[j] = 0.35
            elif f == "Electric":
                fuel_p[j] = 0.15
        fuel = rng.choice(c["fuels"], p=fuel_p / fuel_p.sum())
        at_p = cinfo["at"] * (1.25 if c["body"] in ("SUV", "Sedan") else 1.0)
        trans = "Automatic" if (len(c["trans"]) > 1 and rng.random() < min(at_p, 0.9)) else "Manual"

        entry = start + pd.Timedelta(days=int(rng.integers(0, horizon_days - 20)))
        age_years = float(np.clip(rng.gamma(3.0, 1.6), 0.5, 12.0))
        mfg_year = int(entry.year - math.ceil(age_years))
        reg_year = mfg_year + int(rng.random() < 0.25)
        km_per_year = float(np.clip(rng.normal(12000, 4000), 3000, 30000))
        km = float(np.clip(age_years * km_per_year * rng.uniform(0.8, 1.2), 1500, 220000))
        owners = int(np.clip(1 + rng.poisson(max(age_years - 3, 0) * 0.25), 1, 4))
        accident = bool(rng.random() < 0.08)
        service_hist = bool(rng.random() < 0.7)
        insurance = bool(rng.random() < 0.8)

        # inspection scores correlate with age/km/accident
        base_q = np.clip(rng.normal(82 - 1.6 * age_years - km / 25000 - 8 * accident, 6), 40, 99)
        ext = float(np.clip(base_q + rng.normal(0, 4), 35, 100))
        interior = float(np.clip(base_q + rng.normal(0, 4), 35, 100))
        engine = float(np.clip(base_q + rng.normal(0, 5), 35, 100))
        tyre = float(np.clip(base_q + rng.normal(-3, 7), 20, 100))
        insp = float(np.round(0.3 * ext + 0.25 * interior + 0.3 * engine + 0.15 * tyre, 1))

        n_feat = int(np.clip(rng.poisson(8 + 6 * (variant == "Top") + 3 * (variant == "Mid")), 2, 30))
        engine_cc = float({"Hatchback": 1100, "Sedan": 1400, "Compact SUV": 1300,
                           "SUV": 1900, "MPV": 1500}[c["body"]] * rng.uniform(0.85, 1.2))
        mileage = float(np.clip(rng.normal(24 - engine_cc / 250, 2), 8, 30))

        # ---------------- latent fair-price DGP (log-linear) ----------------
        new_price = c["base"] * 1e5 * VARIANT_MULT[variant]
        retention = 0.845 + 0.045 * c["pop"]                    # popular models retain value
        log_fair = (
            math.log(new_price)
            + age_years * math.log(retention)                   # nonlinear (exponential) depreciation
            + math.log(cinfo["mult"])                           # city effect
            - 0.06 * max(km - age_years * 11000, 0) / 30000     # excess-km penalty
            + 0.25 * (insp - 75) / 100                          # quality
            - 0.13 * accident
            - 0.03 * (owners - 1)
            + (0.045 if trans == "Automatic" else 0.0)
            + (0.03 if (fuel == "Diesel" and c["body"] in ("SUV", "MPV") and cinfo["diesel"] > 0.22) else
               (-0.03 if fuel == "Diesel" else 0.0))            # diesel varies by body/city
            + 0.01 * n_feat / 10
        )
        noise_sigma = 0.05 + 0.05 * min(c["base"] / 40.0, 1.0)  # premium models vary more
        latent_fair = math.exp(log_fair + rng.normal(0, noise_sigma))
        listed = latent_fair * rng.uniform(0.96, 1.14)
        acq = latent_fair * rng.uniform(0.84, 0.92)
        deal_latent = (latent_fair - listed) / latent_fair      # >0 => underpriced

        # sale hazard driven by deal attractiveness, demand, quality
        demand = cinfo["body"][c["body"]] / 0.2                 # ~1.0 average
        hazard = 0.011 * math.exp(2.5 * deal_latent) * demand * (0.6 + insp / 150) * (0.7 + 0.5 * c["pop"])
        days_to_sale = rng.exponential(1.0 / max(hazard, 1e-4))
        max_days = (end - entry).days
        sold = days_to_sale <= max_days
        days_in_inv = int(min(days_to_sale, max_days))
        exit_date = entry + pd.Timedelta(days=days_in_inv) if sold else pd.NaT
        txn_price = round(listed * rng.uniform(0.93, 1.0), 0) if sold else np.nan

        rows.append(dict(
            car_id=f"CAR_{i:05d}", make=c["make"], model=c["model"], variant=variant,
            body_type=c["body"], fuel_type=fuel, transmission=trans,
            manufacturing_year=mfg_year, registration_year=reg_year,
            kilometres_driven=round(km, 0), owner_count=owners, city=city,
            state=cinfo["state"], engine_cc=round(engine_cc, 0),
            claimed_mileage_kmpl=round(mileage, 1), insurance_valid=insurance,
            service_history_available=service_hist, accident_history=accident,
            inspection_score=insp, exterior_score=round(ext, 1),
            interior_score=round(interior, 1), engine_score=round(engine, 1),
            tyre_score=round(tyre, 1), number_of_features=n_feat,
            acquisition_price=round(acq, 0), listed_price=round(listed, 0),
            inventory_entry_date=entry.date(), inventory_exit_date=exit_date,
            sold_flag=sold, transaction_price=txn_price,
            days_in_inventory=days_in_inv,
            delivery_available=bool(rng.random() < 0.75),
            finance_available=bool(rng.random() < 0.85),
            _latent_fair_price=round(latent_fair, 0),
            _deal_latent=round(deal_latent, 4),
            _model_pop=c["pop"],
        ))
    df = pd.DataFrame(rows)
    df["inventory_exit_date"] = pd.to_datetime(df["inventory_exit_date"]).dt.date
    return df


# ==========================================================================
# Users
# ==========================================================================
def generate_users(cfg: Config, gen_cfg: dict, rng: np.random.Generator) -> pd.DataFrame:
    n = int(gen_cfg["n_users"])
    start = pd.Timestamp(gen_cfg["start_date"])
    horizon = (pd.Timestamp(gen_cfg["end_date"]) - start).days
    rows = []
    for i in range(n):
        city = rng.choice(CITY_NAMES, p=CITY_WEIGHTS)
        age_band = rng.choice(AGE_BANDS, p=[0.15, 0.35, 0.28, 0.15, 0.07])
        income_i = int(np.clip(rng.normal(2.0 + 0.4 * AGE_BANDS.index(age_band), 1.1), 0, 4))
        income_band = INCOME_BANDS[income_i]
        family_band = rng.choice(FAMILY_BANDS, p=[0.15, 0.25, 0.42, 0.18])

        # correlated body preference
        body_w = np.array([0.28, 0.16, 0.24, 0.20, 0.12])
        if family_band in ("3-4", "5+"):
            body_w += np.array([-0.10, -0.04, 0.02, 0.06, 0.10]) * (1 + (family_band == "5+"))
        body_w += np.array([-0.05, 0.0, 0.02, 0.05, -0.02]) * income_i / 2
        body_w = np.clip(body_w, 0.02, None); body_w /= body_w.sum()
        pref_body = rng.choice(BODY_TYPES, p=body_w)

        at_pref = CITIES[city]["at"] + 0.08 * income_i / 4
        pref_trans = "Automatic" if rng.random() < at_pref else "Manual"
        pref_fuel = "Diesel" if rng.random() < CITIES[city]["diesel"] else ("CNG" if rng.random() < 0.12 else "Petrol")
        makes = sorted({c["make"] for c in CATALOG})
        pref_make = rng.choice(makes)

        budget_mu = {0: 4.5e5, 1: 6.5e5, 2: 9.5e5, 3: 14e5, 4: 20e5}[income_i]
        ideal_budget = float(np.clip(rng.lognormal(math.log(budget_mu), 0.30), 2e5, 45e5))
        max_budget = ideal_budget * rng.uniform(1.12, 1.35)
        ideal_emi = ideal_budget / 60.0
        max_emi = max_budget / 55.0

        first_time = float(np.clip(0.75 - 0.12 * AGE_BANDS.index(age_band) + rng.normal(0, 0.1), 0.05, 0.95))
        price_sens = float(np.clip(rng.beta(3, 2) + 0.15 * first_time - 0.08 * income_i / 4, 0.05, 0.98))
        quality_sens = float(np.clip(rng.beta(2.5, 2.5) + 0.06 * income_i / 4, 0.05, 0.98))
        brand_loyal = float(np.clip(rng.beta(2, 3), 0.02, 0.98))
        explore = float(np.clip(1 - brand_loyal * rng.uniform(0.6, 1.0), 0.05, 0.98))
        finance_int = float(np.clip(rng.beta(2.2, 2.0) + 0.2 * first_time - 0.1 * income_i / 4, 0.05, 0.98))
        urgency = float(np.clip(rng.beta(1.8, 3.5), 0.02, 0.95))

        rows.append(dict(
            user_id=f"USER_{i:05d}",
            signup_date=(start + pd.Timedelta(days=int(rng.integers(0, max(horizon - 30, 1))))).date(),
            home_city=city, age_band=age_band, income_band=income_band,
            family_size_band=family_band, first_time_buyer_probability=round(first_time, 3),
            preferred_body_types=pref_body, preferred_makes=pref_make,
            preferred_fuel_types=pref_fuel, preferred_transmissions=pref_trans,
            ideal_budget=round(ideal_budget, 0), maximum_budget=round(max_budget, 0),
            ideal_emi=round(ideal_emi, 0), maximum_emi=round(max_emi, 0),
            vehicle_age_tolerance=float(np.clip(rng.normal(6, 2), 2, 12)),
            kilometre_tolerance=float(np.clip(rng.normal(80000, 25000), 20000, 180000)),
            quality_sensitivity=round(quality_sens, 3), price_sensitivity=round(price_sens, 3),
            brand_loyalty=round(brand_loyal, 3), exploration_tendency=round(explore, 3),
            finance_interest=round(finance_int, 3), purchase_urgency=round(urgency, 3),
        ))
    return pd.DataFrame(rows)


# ==========================================================================
# Campaigns
# ==========================================================================
def generate_campaigns(cfg: Config, gen_cfg: dict, rng: np.random.Generator) -> pd.DataFrame:
    n = int(gen_cfg["n_campaigns"])
    start = pd.Timestamp(gen_cfg["start_date"])
    end = pd.Timestamp(gen_cfg["end_date"])
    rows = []
    for i in range(n):
        ch = CHANNELS[i % len(CHANNELS)]
        city = rng.choice(CITY_NAMES + ["All"], p=list(CITY_WEIGHTS * 0.6) + [0.4])
        seg_body = rng.choice(BODY_TYPES + ["All"])
        c_start = start + pd.Timedelta(days=int(rng.integers(0, 60)))
        rows.append(dict(
            campaign_id=f"CMP_{i:03d}",
            campaign_name=f"{ch['channel']} | {city} | {seg_body}",
            source=ch["source"], medium=ch["medium"], channel=ch["channel"],
            target_city=city, target_segment=seg_body,
            start_date=c_start.date(), end_date=end.date(),
            daily_budget=round(float(rng.uniform(3000, 40000)), 0),
            cost_per_click=ch["cpc"] * float(rng.uniform(0.8, 1.2)),
            quality_multiplier=ch["quality"] * float(rng.uniform(0.9, 1.1)),
            intent_multiplier=ch["intent"] * float(rng.uniform(0.9, 1.1)),
            # some campaigns deliberately target under-supplied segments
            inventory_match_multiplier=ch["inv_match"] * float(rng.uniform(0.75, 1.1)),
        ))
    return pd.DataFrame(rows)


# ==========================================================================
# Sessions + events (GA4-style) with position bias
# ==========================================================================
def _car_match_score(session_pref: dict, cars: pd.DataFrame) -> np.ndarray:
    """Vectorized relevance of every car to a session's latent intent."""
    s = np.zeros(len(cars))
    s += 1.4 * (cars["body_type"].values == session_pref["body"])
    s += 0.6 * (cars["make"].values == session_pref["make"])
    s += 0.5 * (cars["fuel_type"].values == session_pref["fuel"])
    s += 0.8 * (cars["transmission"].values == session_pref["trans"])
    price_gap = (cars["listed_price"].values - session_pref["budget"]) / session_pref["budget"]
    s -= 2.2 * np.clip(price_gap, 0, None)          # over budget hurts a lot
    s -= 0.4 * np.clip(-price_gap - 0.5, 0, None)   # far below budget slightly less relevant
    s += 1.5 * cars["_deal_latent"].values          # attractively priced
    s += 0.6 * (cars["inspection_score"].values - 75) / 100
    age = session_pref["entry_year"] - cars["manufacturing_year"].values
    s -= 0.15 * np.clip(age - session_pref["age_tol"], 0, None)
    return s


def generate_sessions_events(cfg: Config, gen_cfg: dict, rng: np.random.Generator,
                             users: pd.DataFrame, cars: pd.DataFrame,
                             campaigns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n_sessions = int(gen_cfg["n_sessions"])
    start = pd.Timestamp(gen_cfg["start_date"]) + pd.Timedelta(days=15)
    end = pd.Timestamp(gen_cfg["end_date"])
    start = max(start, pd.to_datetime(users["signup_date"]).min())
    horizon = (end - start).days
    props = np.array(cfg.get("position_bias", "propensities"))

    # Heavy users generate more sessions (zipf-ish).  Generate the timeline
    # first and process it chronologically so sequence numbers and inventory
    # availability describe the same point in time.
    user_w = rng.pareto(1.6, len(users)) + 1
    user_w /= user_w.sum()
    session_starts = [
        start + pd.Timedelta(days=int(rng.integers(0, horizon)),
                             seconds=int(rng.integers(7 * 3600, 23 * 3600)))
        for _ in range(n_sessions)
    ]
    session_starts.sort()
    signup_dates = pd.to_datetime(users["signup_date"]).to_numpy()

    camp_w = campaigns["daily_budget"].values + 5000
    camp_w = camp_w / camp_w.sum()

    cars_by_city = {c: cars[cars["city"] == c].reset_index(drop=True) for c in CITY_NAMES}
    makes = sorted({c["make"] for c in CATALOG})

    sess_rows, event_rows, imp_rows = [], [], []
    seq_counter: dict[str, int] = {}
    purchased_car_ids: set[str] = set()
    eid = 0

    def emit(ts, u, sid, name, car_id=None, pos=None, term=None, fname=None, fval=None,
             eng=None, src=None, med=None, cmp_=None, dev=None, city=None):
        nonlocal eid
        event_rows.append(dict(
            event_id=f"EVT_{eid:08d}", event_timestamp=ts, event_date=ts.date(),
            user_id=u, session_id=sid, event_name=name, car_id=car_id,
            item_list_id="search_results" if pos is not None else None,
            list_position=pos, search_term=term, filter_name=fname, filter_value=fval,
            engagement_time_seconds=eng, page_location=f"/{name}",
            source=src, medium=med, campaign_id=cmp_, device_category=dev, city=city))
        eid += 1

    for si, ts0 in enumerate(session_starts):
        eligible = signup_dates <= ts0.to_datetime64()
        eligible_w = user_w * eligible
        eligible_w /= eligible_w.sum()
        u = users.iloc[rng.choice(len(users), p=eligible_w)]
        uid = u["user_id"]
        seq = seq_counter.get(uid, 0) + 1
        seq_counter[uid] = seq
        returning = seq > 1

        camp = campaigns.iloc[rng.choice(len(campaigns), p=camp_w)]
        # direct/organic more likely for returning users
        if returning and rng.random() < 0.35:
            direct = campaigns[campaigns["medium"].isin(["(none)", "organic"])]
            if len(direct):
                camp = direct.iloc[rng.integers(0, len(direct))]

        sid = f"SES_{si:06d}"
        device = rng.choice(DEVICES, p=[0.68, 0.27, 0.05])
        city = u["home_city"] if rng.random() < 0.92 else rng.choice(CITY_NAMES)

        # ----- session intent: exploit long-term prefs or drift -------------
        drift = rng.random()
        pref = dict(body=u["preferred_body_types"], make=u["preferred_makes"],
                    fuel=u["preferred_fuel_types"], trans=u["preferred_transmissions"],
                    budget=u["maximum_budget"], age_tol=u["vehicle_age_tolerance"],
                    entry_year=ts0.year, drift="exploit")
        if drift < 0.18:      # explore adjacent body type
            adj = {"Hatchback": "Compact SUV", "Sedan": "SUV", "Compact SUV": "SUV",
                   "SUV": "MPV", "MPV": "SUV"}
            pref["body"] = adj[pref["body"]]; pref["drift"] = "explore_body"
        elif drift < 0.28:    # transmission drift
            pref["trans"] = "Automatic" if pref["trans"] == "Manual" else "Manual"
            pref["drift"] = "explore_transmission"
        elif drift < 0.36:    # budget shift
            pref["budget"] *= rng.uniform(0.75, 1.35); pref["drift"] = "budget_shift"
        elif drift < 0.44 and u["exploration_tendency"] > 0.5:
            pref["make"] = rng.choice(makes); pref["drift"] = "explore_brand"
        # campaign steers intent
        if camp["target_segment"] != "All" and rng.random() < 0.5:
            pref["body"] = camp["target_segment"]; pref["drift"] = "campaign_driven"

        intent_boost = camp["intent_multiplier"] * camp["quality_multiplier"]
        emit(ts0, uid, sid, "session_start", src=camp["source"], med=camp["medium"],
             cmp_=camp["campaign_id"], dev=device, city=city)
        t = ts0 + pd.Timedelta(seconds=int(rng.integers(2, 10)))
        emit(t, uid, sid, "view_home", src=camp["source"], med=camp["medium"],
             cmp_=camp["campaign_id"], dev=device, city=city)

        pool_all = cars_by_city.get(city, cars)
        # available at session time
        entry_ok = pd.to_datetime(pool_all["inventory_entry_date"]) <= ts0
        exit_dt = pd.to_datetime(pool_all["inventory_exit_date"])
        not_sold_yet = exit_dt.isna() | (exit_dt >= ts0)
        not_purchased = ~pool_all["car_id"].isin(purchased_car_ids)
        pool = pool_all[entry_ok & not_sold_yet & not_purchased]
        if len(pool) < 5:
            entry_ok = pd.to_datetime(cars["inventory_entry_date"]) <= ts0
            exit_dt = pd.to_datetime(cars["inventory_exit_date"])
            not_sold_yet = exit_dt.isna() | (exit_dt >= ts0)
            not_purchased = ~cars["car_id"].isin(purchased_car_ids)
            pool = cars[entry_ok & not_sold_yet & not_purchased]
        if len(pool) == 0:
            emit(t, uid, sid, "session_end", src=camp["source"], med=camp["medium"],
                 cmp_=camp["campaign_id"], dev=device, city=city)
            sess_rows.append(dict(session_id=sid, user_id=uid, session_start=ts0,
                                  session_end=t, campaign_id=camp["campaign_id"],
                                  source=camp["source"], medium=camp["medium"],
                                  device_category=device, city=city,
                                  is_returning_user=returning, session_sequence_number=seq,
                                  _drift=pref["drift"]))
            continue

        # searches + filters reveal intent
        n_search = 1 + int(rng.random() < 0.5)
        for _ in range(n_search):
            t += pd.Timedelta(seconds=int(rng.integers(5, 25)))
            term = f"{pref['trans'].lower()} {pref['body'].lower()}" if rng.random() < 0.6 else f"{pref['make']} {pref['body'].lower()}"
            emit(t, uid, sid, "search", term=term, src=camp["source"], med=camp["medium"],
                 cmp_=camp["campaign_id"], dev=device, city=city)
        for fname, fval, p in [("body_type", pref["body"], 0.7),
                               ("transmission", pref["trans"], 0.55),
                               ("max_price", str(int(pref["budget"])), 0.6),
                               ("fuel_type", pref["fuel"], 0.3)]:
            if rng.random() < p:
                t += pd.Timedelta(seconds=int(rng.integers(2, 12)))
                emit(t, uid, sid, "apply_filter", fname=fname, fval=fval,
                     src=camp["source"], med=camp["medium"], cmp_=camp["campaign_id"],
                     dev=device, city=city)

        # results list: relevance + noise -> ranked impressions
        rel = _car_match_score(pref, pool) + rng.normal(0, 0.6, len(pool))
        order = np.argsort(-rel)[:20]
        t += pd.Timedelta(seconds=int(rng.integers(2, 8)))
        emit(t, uid, sid, "view_search_results", src=camp["source"], med=camp["medium"],
             cmp_=camp["campaign_id"], dev=device, city=city)

        booked_session = False
        purchased_session = False
        for rank, ci in enumerate(order, start=1):
            car = pool.iloc[ci]
            examined = rng.random() < props[min(rank - 1, len(props) - 1)]
            rel_z = float(rel[ci])
            p_click = _sigmoid(-1.9 + 0.75 * rel_z + 0.5 * (intent_boost - 1.0))
            clicked = bool(examined and rng.random() < p_click)
            row = dict(session_id=sid, user_id=uid, car_id=car["car_id"],
                       list_position=rank, examined=examined, clicked=clicked,
                       viewed_gallery=False, viewed_inspection=False, compared=False,
                       emi_calculated=False, wishlisted=False, callback=False,
                       booked=False, purchased=False,
                       event_timestamp=t, campaign_id=camp["campaign_id"])
            if clicked:
                t += pd.Timedelta(seconds=int(rng.integers(3, 20)))
                emit(t, uid, sid, "select_item", car_id=car["car_id"], pos=rank,
                     src=camp["source"], med=camp["medium"], cmp_=camp["campaign_id"],
                     dev=device, city=city)
                t += pd.Timedelta(seconds=int(rng.integers(5, 40)))
                emit(t, uid, sid, "view_item", car_id=car["car_id"], pos=rank,
                     eng=float(rng.integers(15, 240)), src=camp["source"], med=camp["medium"],
                     cmp_=camp["campaign_id"], dev=device, city=city)

                match_q = _sigmoid(0.8 * rel_z)
                if rng.random() < 0.55:
                    row["viewed_gallery"] = True
                    t += pd.Timedelta(seconds=int(rng.integers(5, 60)))
                    emit(t, uid, sid, "view_gallery", car_id=car["car_id"],
                         src=camp["source"], med=camp["medium"], cmp_=camp["campaign_id"],
                         dev=device, city=city)
                if rng.random() < 0.25 + 0.45 * u["quality_sensitivity"]:
                    row["viewed_inspection"] = True
                    t += pd.Timedelta(seconds=int(rng.integers(10, 90)))
                    emit(t, uid, sid, "view_inspection_report", car_id=car["car_id"],
                         src=camp["source"], med=camp["medium"], cmp_=camp["campaign_id"],
                         dev=device, city=city)
                if rng.random() < 0.18:
                    row["compared"] = True
                    emit(t, uid, sid, "compare_car", car_id=car["car_id"],
                         src=camp["source"], med=camp["medium"], cmp_=camp["campaign_id"],
                         dev=device, city=city)
                if rng.random() < 0.15 + 0.5 * u["finance_interest"]:
                    row["emi_calculated"] = True
                    t += pd.Timedelta(seconds=int(rng.integers(5, 45)))
                    emit(t, uid, sid, "calculate_emi", car_id=car["car_id"],
                         src=camp["source"], med=camp["medium"], cmp_=camp["campaign_id"],
                         dev=device, city=city)
                    if rng.random() < 0.4:
                        emit(t, uid, sid, "view_finance_offer", car_id=car["car_id"],
                             src=camp["source"], med=camp["medium"], cmp_=camp["campaign_id"],
                             dev=device, city=city)
                if rng.random() < 0.10 + 0.35 * match_q:
                    row["wishlisted"] = True
                    t += pd.Timedelta(seconds=int(rng.integers(2, 15)))
                    emit(t, uid, sid, "add_to_wishlist", car_id=car["car_id"],
                         src=camp["source"], med=camp["medium"], cmp_=camp["campaign_id"],
                         dev=device, city=city)

                # booking depends on deep engagement + urgency + quality + budget fit
                price_mismatch = max((car["listed_price"] - pref["budget"]) / pref["budget"], 0)
                p_book = _sigmoid(
                    -3.4
                    + 1.1 * row["viewed_inspection"] + 0.9 * row["wishlisted"]
                    + 0.7 * row["emi_calculated"] + 0.6 * row["compared"]
                    + 1.3 * u["purchase_urgency"] + 0.9 * (car["inspection_score"] - 75) / 25
                    + 0.8 * (intent_boost - 1.0) + 1.2 * car["_deal_latent"]
                    - 2.0 * price_mismatch
                )
                if (not booked_session) and rng.random() < p_book:
                    if rng.random() < 0.5:
                        emit(t, uid, sid, "request_callback", car_id=car["car_id"],
                             src=camp["source"], med=camp["medium"], cmp_=camp["campaign_id"],
                             dev=device, city=city)
                        row["callback"] = True
                    t += pd.Timedelta(seconds=int(rng.integers(10, 60)))
                    emit(t, uid, sid, "book_test_drive", car_id=car["car_id"],
                         src=camp["source"], med=camp["medium"], cmp_=camp["campaign_id"],
                         dev=device, city=city)
                    emit(t, uid, sid, "begin_checkout", car_id=car["car_id"],
                         src=camp["source"], med=camp["medium"], cmp_=camp["campaign_id"],
                         dev=device, city=city)
                    emit(t, uid, sid, "booking_complete", car_id=car["car_id"],
                         src=camp["source"], med=camp["medium"], cmp_=camp["campaign_id"],
                         dev=device, city=city)
                    row["booked"] = True
                    booked_session = True
                    if car["sold_flag"] and rng.random() < 0.45:
                        t += pd.Timedelta(seconds=int(rng.integers(30, 120)))
                        emit(t, uid, sid, "purchase", car_id=car["car_id"],
                             src=camp["source"], med=camp["medium"], cmp_=camp["campaign_id"],
                             dev=device, city=city)
                        row["purchased"] = True
                        purchased_session = True
                        purchased_car_ids.add(car["car_id"])
                        car_idx = cars.index[cars["car_id"] == car["car_id"]][0]
                        cars.at[car_idx, "inventory_exit_date"] = t.date()
                        cars.at[car_idx, "days_in_inventory"] = max(
                            0,
                            (t.normalize() - pd.Timestamp(cars.at[car_idx, "inventory_entry_date"])).days,
                        )
            imp_rows.append(row)

        t += pd.Timedelta(seconds=int(rng.integers(3, 30)))
        emit(t, uid, sid, "session_end", src=camp["source"], med=camp["medium"],
             cmp_=camp["campaign_id"], dev=device, city=city)
        sess_rows.append(dict(session_id=sid, user_id=uid, session_start=ts0,
                              session_end=t, campaign_id=camp["campaign_id"],
                              source=camp["source"], medium=camp["medium"],
                              device_category=device, city=city,
                              is_returning_user=returning, session_sequence_number=seq,
                              _drift=pref["drift"], _booked=booked_session,
                              _purchased=purchased_session))

    sessions = pd.DataFrame(sess_rows)
    events = pd.DataFrame(event_rows)
    impressions = pd.DataFrame(imp_rows)
    return sessions, events, impressions


# ==========================================================================
# Orchestrator
# ==========================================================================
def build_all(cfg: Config, small: bool = False) -> dict[str, pd.DataFrame]:
    gen_cfg = cfg.data_gen(small=small)
    rng = np.random.default_rng(int(gen_cfg.get("random_seed", cfg.seed)))
    cfg.ensure_dirs()

    cars = generate_cars(cfg, gen_cfg, rng)
    users = generate_users(cfg, gen_cfg, rng)
    campaigns = generate_campaigns(cfg, gen_cfg, rng)
    sessions, events, impressions = generate_sessions_events(cfg, gen_cfg, rng, users, cars, campaigns)

    # split hidden latents out of public tables
    latents = cars[["car_id", "_latent_fair_price", "_deal_latent", "_model_pop"]].copy()
    cars_pub = cars.drop(columns=["_latent_fair_price", "_deal_latent"])
    cars_pub = cars_pub.rename(columns={"_model_pop": "model_popularity"})  # observable proxy (catalog popularity)
    sess_pub = sessions.drop(columns=[c for c in sessions.columns if c.startswith("_")])

    out = cfg.raw_data
    cars_pub.to_parquet(out / "cars.parquet", index=False)
    users.to_parquet(out / "users.parquet", index=False)
    campaigns.to_parquet(out / "campaigns.parquet", index=False)
    sess_pub.to_parquet(out / "sessions.parquet", index=False)
    events.to_parquet(out / "events.parquet", index=False)
    impressions.to_parquet(out / "impressions.parquet", index=False)
    latents.to_parquet(out / "_latents_cars.parquet", index=False)
    sessions.to_parquet(out / "_latents_sessions.parquet", index=False)

    return dict(cars=cars_pub, users=users, campaigns=campaigns,
                sessions=sess_pub, events=events, impressions=impressions)
