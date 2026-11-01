"""End-to-end API tests: consent -> fetch -> score -> card -> recommendation."""
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from msme_health.config import DATA_PATH


@pytest.fixture(scope="module")
def client():
    from api.main import app
    return TestClient(app)


@pytest.fixture(scope="module")
def sample():
    row = pd.read_csv(DATA_PATH, nrows=5).iloc[2].to_dict()
    for k in list(row):
        if k.endswith("_Score") or k in ("Probability_of_Default", "Credit_Risk_Category",
                                         "Credit_Eligible", "Recommended_Credit_Limit_INR"):
            row.pop(k)
    return {k: (None if isinstance(v, float) and pd.isna(v) else v) for k, v in row.items()}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["portfolio_size"] == 50000


def test_consent_fetch_score_flow(client):
    c = client.post("/api/v1/consents", json={"msme_id": "MSME0000010"}).json()
    assert c["status"] == "ACTIVE"
    f = client.post(f"/api/v1/msmes/MSME0000010/fetch?consent_id={c['consent_id']}")
    assert f.status_code == 200 and all(v == "ok" for v in f.json()["sources"].values())
    s = client.post("/api/v1/msmes/MSME0000010/score")
    assert s.status_code == 200
    body = s.json()
    assert 0 <= body["financial_health_score"] <= 100
    assert body["risk_band"] in {"Low", "Medium", "High"}


def test_health_card_contents(client):
    card = client.get("/api/v1/msmes/MSME0000001/health-card").json()
    assert len(card["pillars"]) == 6
    assert card["explanation"]["top_drivers"]
    assert {"strengths", "concerns", "improvement_actions"} <= set(card["insights"])
    assert card["peer_benchmark"]["peer_count"] >= 200


def test_realtime_score_matches_schema(client, sample):
    r = client.post("/api/v1/score", json=sample)
    assert r.status_code == 200, r.text
    assert r.json()["credit_recommendation"]["decision"]


def test_validation_rejects_bad_input(client, sample):
    bad = dict(sample, GST_Filing_Timeliness_Pct=140)
    assert client.post("/api/v1/score", json=bad).status_code == 422


def test_source_ingestion_merges(client):
    r = client.post("/api/v1/msmes/NEW0001/sources/gstn",
                    json={"fields": {"Monthly_GST_Sales_INR": 500000, "GST_Filing_Timeliness_Pct": 92}})
    assert r.status_code == 200 and r.json()["ready_to_score"] is False
    bad = client.post("/api/v1/msmes/NEW0001/sources/gstn", json={"fields": {"Monthly_Payroll_INR": 1}})
    assert bad.status_code == 422


def test_credit_recommendation_and_portfolio(client):
    rec = client.get("/api/v1/msmes/MSME0000006/credit-recommendation").json()
    assert rec["decision"] in {"Approve", "Approve with conditions", "Refer to underwriter", "Decline"}
    if rec["decision"] == "Decline":
        assert rec["recommended_limit_inr"] == 0
    port = client.get("/api/v1/dashboard/portfolio?segment=NTC").json()
    assert port["count"] > 0 and 0 < port["approval_rate"] < 1
