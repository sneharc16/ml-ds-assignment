"""End-to-end smoke test: artifacts load, API responds, recs generate."""
import logging
import sys

import pandas as pd
from fastapi.testclient import TestClient

from driveintent.api.main import app
from driveintent.config import load_config

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

if __name__ == "__main__":
    cfg = load_config()
    failures = []
    with TestClient(app) as client:
        h = client.get("/health").json()
        if not h.get("models_loaded"):
            failures.append("models not loaded")
        r = client.post("/api/v1/pricing/predict", json=dict(
            make="Hyundai", model="Creta", body_type="SUV", fuel_type="Petrol",
            transmission="Automatic", manufacturing_year=2021,
            kilometres_driven=35000, city="Bengaluru", listed_price=1200000))
        if r.status_code != 200:
            failures.append(f"pricing endpoint {r.status_code}")
        ev = pd.read_parquet(cfg.raw_data / "events.parquet")
        uid = ev["user_id"].value_counts().index[0]
        r = client.get(f"/api/v1/recommendations/{uid}?limit=5")
        if r.status_code != 200 or not r.json()["recommendations"]:
            failures.append("recommendations endpoint failed")
    if failures:
        logging.error("SMOKE TEST FAILED: %s", failures)
        sys.exit(1)
    logging.info("SMOKE TEST PASSED")
