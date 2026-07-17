import pandas as pd
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(pipeline):
    from driveintent.api.main import app
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["models_loaded"] is True


def test_pricing_valid(client):
    r = client.post("/api/v1/pricing/predict", json=dict(
        make="Hyundai", model="Creta", body_type="SUV", fuel_type="Petrol",
        transmission="Automatic", manufacturing_year=2021,
        kilometres_driven=35000, city="Bengaluru", listed_price=1200000))
    assert r.status_code == 200
    body = r.json()
    assert body["lower_price"] <= body["predicted_fair_price"] <= body["upper_price"]
    assert body["deal_label"] in {"great_deal", "good_deal", "fair_price", "overpriced"}


def test_pricing_invalid(client):
    assert client.post("/api/v1/pricing/predict", json={"make": "X"}).status_code == 422


def test_booking_probability(client):
    r = client.post("/api/v1/conversion/booking-probability", json={})
    assert r.status_code == 200
    assert 0 <= r.json()["booking_probability"] <= 1


def test_recommendations_known_and_unknown(client, pipeline):
    ev = pd.read_parquet(pipeline.raw_data / "events.parquet")
    uid = ev["user_id"].value_counts().index[0]
    r = client.get(f"/api/v1/recommendations/{uid}?limit=5")
    assert r.status_code == 200 and len(r.json()["recommendations"]) > 0
    r = client.get("/api/v1/recommendations/GHOST?limit=5")
    assert r.status_code == 200  # cold-start fallback still answers


def test_inventory_and_campaigns(client):
    assert client.get("/api/v1/inventory/opportunities").status_code == 200
    assert client.get("/api/v1/campaigns/performance").status_code == 200
    r = client.post("/api/v1/campaigns/optimize-budget", json={})
    assert r.status_code == 200 and "allocation" in r.json()
