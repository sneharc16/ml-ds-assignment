"""Hybrid recommendation components.

- ContentRecommender: standardized+one-hot car vectors, cosine similarity,
  long-term and session user profiles (event-weighted, recency-decayed)
- CollaborativeRecommender: implicit ALS on weighted interactions
  (deterministic SVD fallback if implicit is unavailable)
- SessionIntentRecommender: recency-decayed current-session vector
- Popularity/cold-start fallback: city-segment popularity, never global-only
- generate_candidates: dedup union with source flags and hard-constraint filters
"""
from __future__ import annotations

import math

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from driveintent.config import Config
from driveintent.features.intent import EVENT_WEIGHTS, infer_profile

INTERACTION_WEIGHTS = {
    "select_item": 1.0, "view_item": 2.0, "view_gallery": 3.0,
    "view_inspection_report": 5.0, "compare_car": 6.0, "calculate_emi": 7.0,
    "add_to_wishlist": 8.0, "request_callback": 11.0, "book_test_drive": 14.0,
    "booking_complete": 18.0, "purchase": 25.0,
}

CONTENT_NUM = ["listed_price", "vehicle_age", "kilometres_driven",
               "inspection_score", "number_of_features", "estimated_emi"]
CONTENT_CAT = ["body_type", "make", "fuel_type", "transmission", "city"]


class ContentRecommender:
    def __init__(self) -> None:
        self.pre: ColumnTransformer | None = None
        self.matrix: sparse.csr_matrix | None = None
        self.car_ids: np.ndarray | None = None
        self.id_to_idx: dict[str, int] = {}

    def fit(self, cars: pd.DataFrame) -> "ContentRecommender":
        df = cars.copy()
        entry = pd.to_datetime(df["inventory_entry_date"])
        df["vehicle_age"] = (entry.dt.year - df["manufacturing_year"]).clip(lower=0)
        df["estimated_emi"] = df["listed_price"] / 60.0
        self.pre = ColumnTransformer([
            ("num", StandardScaler(), CONTENT_NUM),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CONTENT_CAT)])
        m = self.pre.fit_transform(df)
        m = sparse.csr_matrix(m)
        norms = np.sqrt(m.multiply(m).sum(axis=1)).A.ravel()
        norms[norms == 0] = 1.0
        self.matrix = sparse.csr_matrix(m.multiply(1.0 / norms[:, None]))
        self.car_ids = df["car_id"].to_numpy()
        self.id_to_idx = {c: i for i, c in enumerate(self.car_ids)}
        return self

    def profile_vector(self, events: pd.DataFrame,
                       decay_per_day: float = 0.03,
                       ref_time: pd.Timestamp | None = None) -> np.ndarray | None:
        ev = events.dropna(subset=["car_id"])
        ev = ev[ev["car_id"].isin(self.id_to_idx)]
        if ev.empty:
            return None
        ref_time = ref_time or pd.to_datetime(ev["event_timestamp"]).max()
        w = ev["event_name"].map(EVENT_WEIGHTS).fillna(0.5).to_numpy()
        dt = (ref_time - pd.to_datetime(ev["event_timestamp"])).dt.total_seconds().to_numpy() / 86400
        w = w * np.exp(-decay_per_day * np.clip(dt, 0, None))
        idx = ev["car_id"].map(self.id_to_idx).to_numpy()
        vec = np.zeros(self.matrix.shape[1])
        for i, wt in zip(idx, w):
            vec += wt * self.matrix[i].toarray().ravel()
        n = np.linalg.norm(vec)
        return vec / n if n > 0 else None

    def score(self, profile: np.ndarray | None) -> np.ndarray:
        if profile is None:
            return np.zeros(len(self.car_ids))
        return self.matrix.dot(profile)

    def item_similarity(self, car_a: str, car_b: str) -> float:
        ia, ib = self.id_to_idx.get(car_a), self.id_to_idx.get(car_b)
        if ia is None or ib is None:
            return 0.0
        return float(self.matrix[ia].dot(self.matrix[ib].T).toarray()[0, 0])


class CollaborativeRecommender:
    """Implicit ALS on user-car interaction strengths."""
    def __init__(self, factors: int = 32, regularization: float = 0.05,
                 iterations: int = 15, alpha: float = 20.0, seed: int = 42) -> None:
        self.factors, self.reg = factors, regularization
        self.iters, self.alpha, self.seed = iterations, alpha, seed
        self.user_f: np.ndarray | None = None
        self.item_f: np.ndarray | None = None
        self.user_ids: list[str] = []
        self.car_ids: list[str] = []
        self.uidx: dict[str, int] = {}
        self.cidx: dict[str, int] = {}

    def fit(self, events: pd.DataFrame) -> "CollaborativeRecommender":
        ev = events.dropna(subset=["car_id"])
        ev = ev[ev["event_name"].isin(INTERACTION_WEIGHTS)]
        ev = ev.assign(w=ev["event_name"].map(INTERACTION_WEIGHTS))
        agg = ev.groupby(["user_id", "car_id"])["w"].sum().reset_index()
        self.user_ids = sorted(agg["user_id"].unique())
        self.car_ids = sorted(agg["car_id"].unique())
        self.uidx = {u: i for i, u in enumerate(self.user_ids)}
        self.cidx = {c: i for i, c in enumerate(self.car_ids)}
        mat = sparse.csr_matrix(
            (agg["w"].to_numpy() * self.alpha,
             (agg["user_id"].map(self.uidx), agg["car_id"].map(self.cidx))),
            shape=(len(self.user_ids), len(self.car_ids)))
        try:
            from implicit.als import AlternatingLeastSquares
            als = AlternatingLeastSquares(factors=self.factors, regularization=self.reg,
                                          iterations=self.iters, random_state=self.seed,
                                          use_gpu=False)
            als.fit(mat, show_progress=False)
            self.user_f = np.asarray(als.user_factors)
            self.item_f = np.asarray(als.item_factors)
        except Exception:  # deterministic fallback
            from sklearn.decomposition import TruncatedSVD
            k = min(self.factors, min(mat.shape) - 1)
            svd = TruncatedSVD(n_components=max(k, 2), random_state=self.seed)
            self.user_f = svd.fit_transform(mat)
            self.item_f = svd.components_.T
        return self

    def score_user(self, user_id: str, car_ids: list[str]) -> np.ndarray:
        if user_id not in self.uidx:
            return np.zeros(len(car_ids))
        u = self.user_f[self.uidx[user_id]]
        out = np.zeros(len(car_ids))
        for i, c in enumerate(car_ids):
            j = self.cidx.get(c)
            if j is not None:
                out[i] = float(u @ self.item_f[j])
        return out

    def top_k(self, user_id: str, k: int = 50, exclude: set[str] | None = None) -> list[tuple[str, float]]:
        if user_id not in self.uidx:
            return []
        scores = self.item_f @ self.user_f[self.uidx[user_id]]
        order = np.argsort(-scores)
        out = []
        exclude = exclude or set()
        for j in order:
            c = self.car_ids[j]
            if c not in exclude:
                out.append((c, float(scores[j])))
            if len(out) >= k:
                break
        return out


class RecommenderBundle:
    """Fitted content + collaborative recommenders plus car metadata."""
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.content = ContentRecommender()
        self.collab: CollaborativeRecommender | None = None
        self.cars: pd.DataFrame | None = None

    def fit(self, cars: pd.DataFrame, events_train: pd.DataFrame) -> "RecommenderBundle":
        self.cars = cars.reset_index(drop=True)
        self.content.fit(self.cars)
        p = self.cfg.get("models", "collaborative")
        self.collab = CollaborativeRecommender(
            factors=p["factors"], regularization=p["regularization"],
            iterations=p["iterations"], alpha=p["alpha"], seed=self.cfg.seed
        ).fit(events_train)
        return self

    def save(self) -> None:
        joblib.dump(self, self.cfg.artifacts / "recommender" / "recommender_bundle.joblib")

    @staticmethod
    def load(cfg: Config) -> "RecommenderBundle":
        from driveintent.models.registry import ModelArtifactNotFoundError
        path = cfg.artifacts / "recommender" / "recommender_bundle.joblib"
        if not path.exists():
            raise ModelArtifactNotFoundError(
                f"Recommender bundle not found at {path}. "
                "Run `python scripts/train_all_models.py --model recommender`.")
        return joblib.load(path)


def popularity_by_segment(events: pd.DataFrame, cars: pd.DataFrame) -> pd.DataFrame:
    ev = events[events["event_name"].isin(INTERACTION_WEIGHTS)].dropna(subset=["car_id"])
    ev = ev.assign(w=ev["event_name"].map(INTERACTION_WEIGHTS))
    pop = ev.groupby("car_id")["w"].sum().rename("popularity").reset_index()
    return cars[["car_id", "city", "body_type", "listed_price", "inspection_score"]].merge(
        pop, on="car_id", how="left").fillna({"popularity": 0.0})


def generate_candidates(bundle: RecommenderBundle, user_events: pd.DataFrame,
                        user_row: pd.Series | None, session_id: str | None,
                        available_ids: set[str], pop_table: pd.DataFrame,
                        cfg: Config, city: str | None = None) -> pd.DataFrame:
    """Union of candidate sources with dedup, source flags and hard-constraint filter."""
    rec_cfg = cfg.get("recommendation")
    cars = bundle.cars
    avail_mask = cars["car_id"].isin(available_ids).to_numpy()

    prof = infer_profile(user_events, cars, session_id=session_id,
                         session_half_life_minutes=rec_cfg["session_half_life_minutes"])

    lt_vec = bundle.content.profile_vector(
        user_events[user_events["session_id"] != session_id] if session_id else user_events)
    sess_events = user_events[user_events["session_id"] == session_id] if session_id else user_events.iloc[0:0]
    half_life_days = rec_cfg["session_half_life_minutes"] / (60 * 24)
    sess_vec = bundle.content.profile_vector(sess_events,
                                             decay_per_day=math.log(2) / max(half_life_days, 1e-9))

    content_s = bundle.content.score(lt_vec)
    session_s = bundle.content.score(sess_vec)
    uid = user_row["user_id"] if user_row is not None else (
        user_events["user_id"].iloc[0] if len(user_events) else "")
    collab_s = np.zeros(len(cars))
    if bundle.collab is not None and uid in bundle.collab.uidx:
        collab_s = np.array(bundle.collab.score_user(uid, list(cars["car_id"])))

    def _top(scores: np.ndarray, k: int) -> list[int]:
        s = np.where(avail_mask, scores, -np.inf)
        order = np.argsort(-s)[:k]
        return [int(i) for i in order if np.isfinite(s[i]) and s[i] > 0]

    sources: dict[int, set[str]] = {}
    for name, idxs in [
        ("content", _top(content_s, rec_cfg["content_top_k"])),
        ("collaborative", _top(collab_s, rec_cfg["collaborative_top_k"])),
        ("session", _top(session_s, rec_cfg["session_top_k"]))]:
        for i in idxs:
            sources.setdefault(i, set()).add(name)

    # popularity fallback (city segment + inferred body preference), never global-only
    seg_pop = pop_table[pop_table["car_id"].isin(available_ids)]
    ucity = city or (user_row["home_city"] if user_row is not None else None)
    if ucity is not None and (seg_pop["city"] == ucity).any():
        seg_pop = seg_pop[seg_pop["city"] == ucity]
    pref_body = None
    body_dist = prof.session.get("body_type") or prof.long_term.get("body_type") or {}
    if body_dist:
        pref_body = max(body_dist, key=body_dist.get)
    if pref_body and (seg_pop["body_type"] == pref_body).any():
        seg_pop = pd.concat([seg_pop[seg_pop["body_type"] == pref_body],
                             seg_pop[seg_pop["body_type"] != pref_body]])
    for cid in seg_pop.sort_values("popularity", ascending=False)["car_id"].head(
            rec_cfg["popularity_top_k"]):
        i = bundle.content.id_to_idx.get(cid)
        if i is not None:
            sources.setdefault(i, set()).add("popularity")

    # inventory-opportunity candidates: aging but decent quality
    inv = cars[avail_mask].copy()
    entry = pd.to_datetime(inv["inventory_entry_date"])
    inv["inv_age"] = (pd.Timestamp.now().normalize() - entry).dt.days.clip(lower=0)
    inv = inv[inv["inspection_score"] >= 70].sort_values("inv_age", ascending=False)
    for cid in inv["car_id"].head(rec_cfg["inventory_top_k"]):
        i = bundle.content.id_to_idx.get(cid)
        if i is not None:
            sources.setdefault(i, set()).add("inventory")

    if not sources:
        return pd.DataFrame()
    idxs = list(sources.keys())[: rec_cfg["candidate_limit"]]
    cand = cars.iloc[idxs].copy().reset_index(drop=True)
    cand["content_score"] = content_s[idxs]
    cand["session_score"] = session_s[idxs]
    cand["collaborative_score"] = collab_s[idxs]
    cand["candidate_source"] = ["+".join(sorted(sources[i])) for i in idxs]

    # hard-constraint filtering
    hc = prof.hard_constraints
    keep = np.ones(len(cand), bool)
    if "max_price" in hc:
        keep &= cand["listed_price"] <= hc["max_price"] * 1.05
    for dim in ("body_type", "transmission", "fuel_type"):
        if dim in hc:
            keep &= cand[dim] == hc[dim]
    if keep.sum() >= 5:            # don't empty the pool on aggressive constraints
        cand = cand[keep]
    cand.attrs["profile"] = prof
    return cand.reset_index(drop=True)
