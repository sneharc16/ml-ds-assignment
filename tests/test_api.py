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
    assert r.status_code == 422


def test_booking_probability_for_feature_store_pair(client, pipeline):
    ds = pd.read_parquet(pipeline.processed_data / "booking_dataset.parquet")
    row = ds.iloc[0]
    r = client.post("/api/v1/conversion/booking-probability", json={
        "user_id": row["user_id"], "session_id": row["session_id"], "car_id": row["car_id"]
    })
    assert r.status_code == 200
    assert 0 <= r.json()["booking_probability"] <= 1


def test_booking_rejects_partial_or_unknown_inputs(client):
    assert client.post("/api/v1/conversion/booking-probability",
                       json={"session_id": "SES_1"}).status_code == 422
    assert client.post("/api/v1/conversion/booking-probability",
                       json={"features": {"label_booked": 1}}).status_code == 422


def test_recommendations_known_and_unknown(client, pipeline):
    ev = pd.read_parquet(pipeline.raw_data / "events.parquet")
    uid = ev["user_id"].value_counts().index[0]
    r = client.get(f"/api/v1/recommendations/{uid}?limit=5")
    assert r.status_code == 200 and len(r.json()["recommendations"]) > 0
    r = client.get("/api/v1/recommendations/GHOST?limit=5")
    assert r.status_code == 200  # cold-start fallback still answers


def test_recommendations_reject_foreign_session(client, pipeline):
    ev = pd.read_parquet(pipeline.raw_data / "events.parquet")
    users = ev.groupby("user_id")["session_id"].first().head(2)
    r = client.get(f"/api/v1/recommendations/{users.index[0]}?session_id={users.iloc[1]}")
    assert r.status_code == 404


def test_sellthrough_rejects_negative_snapshot_age(client):
    r = client.post("/api/v1/conversion/sellthrough-probability",
                    json={"car_id": "CAR_00000", "snapshot_age": -1})
    assert r.status_code == 422


def test_inventory_and_campaigns(client):
    assert client.get("/api/v1/inventory/opportunities").status_code == 200
    assert client.get("/api/v1/campaigns/performance").status_code == 200
    r = client.post("/api/v1/campaigns/optimize-budget", json={})
    assert r.status_code == 200 and "allocation" in r.json()
    assert client.post("/api/v1/campaigns/optimize-budget",
                       json={"total_budget": -1}).status_code == 422
