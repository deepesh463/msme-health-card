"""REST API for the MSME Financial Health Card.

Run locally:  uvicorn api.main:app --reload
Docs:         http://localhost:8000/docs
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from msme_health.config import CATEGORICAL, ID_COL, NUMERIC_RAW, PILLARS
from msme_health.connectors import REAL_INTEGRATION, SOURCE_FIELDS, build_connectors
from msme_health.engine import get_engine
from msme_health.schemas import ConsentRequest, MSMEData, ScoreSummary, SourcePayload
from msme_health.store import Store

app = FastAPI(
    title="Cars24 MSME Financial Health Card API",
    version="1.0.0",
    description="Consent-based alternate-data credit assessment for NTC/NTB small businesses.",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

engine = get_engine()
store = Store()
_raw = engine.scored.set_index(ID_COL)[CATEGORICAL + NUMERIC_RAW] if engine.scored is not None else pd.DataFrame()
connectors = build_connectors(_raw)
REQUIRED = set(CATEGORICAL + NUMERIC_RAW) - {"EMI_On_Time_Rate_Pct"}


def _record(msme_id: str) -> dict:
    """Ingested data first; otherwise the pre-loaded portfolio."""
    data = store.get_data(msme_id)
    if data:
        return data
    msme_id = msme_id.upper()  # portfolio IDs are upper case; accept msme0000001 too
    if msme_id in _raw.index:
        row = _raw.loc[msme_id].to_dict()
        row[ID_COL] = msme_id
        return {k: (None if isinstance(v, float) and pd.isna(v) else v) for k, v in row.items()}
    raise HTTPException(404, f"No data for {msme_id}. Ingest data first.")


def _summary(card: dict, ts: str) -> ScoreSummary:
    rec = card["credit_recommendation"]
    return ScoreSummary(
        msme_id=card["msme_id"], financial_health_score=card["financial_health_score"], grade=card["grade"],
        probability_of_default=card["probability_of_default"], risk_band=card["risk_band"],
        decision=rec["decision"], recommended_limit_inr=rec["recommended_limit_inr"], scored_at=ts,
    )


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


@app.get("/health", tags=["system"])
@app.get("/api/v1/health", tags=["system"])
def health():
    return {"status": "ok", "portfolio_size": int(len(_raw))}


# ---------------- Consent & ingestion ----------------
@app.post("/api/v1/consents", tags=["ingestion"])
def create_consent(req: ConsentRequest):
    """Mock Account Aggregator consent artefact. In production this is raised via the AA's consent manager."""
    req.msme_id = req.msme_id.upper()
    cid = "CNS-" + uuid.uuid4().hex[:12].upper()
    now = datetime.now(timezone.utc)
    artefact = {
        "consent_id": cid, "msme_id": req.msme_id, "status": "ACTIVE", "purpose": req.purpose,
        "fi_types": req.fi_types, "data_range_months": req.data_range_months,
        "valid_until": (now + timedelta(days=req.consent_validity_days)).isoformat(timespec="seconds"),
        "fetch_type": "PERIODIC", "data_life": "store until consent expiry",
    }
    store.put_consent(cid, req.msme_id, artefact)
    store.audit(req.msme_id, "consent_created", artefact)
    return artefact


@app.post("/api/v1/msmes", tags=["ingestion"])
def ingest_full(data: MSMEData):
    """Ingest a complete, already-consolidated record for one business."""
    rec = data.model_dump()
    store.put_data(data.MSME_ID, rec)
    store.audit(data.MSME_ID, "data_ingested", {"source": "full_record"})
    return {"msme_id": data.MSME_ID, "status": "ingested", "fields": len(rec)}


@app.post("/api/v1/msmes/{msme_id}/sources/{source}", tags=["ingestion"])
def ingest_source(msme_id: str, source: str, payload: SourcePayload):
    """Push one source's data (e.g. a GSTN or EPFO payload). Merged into the business's record."""
    if source not in SOURCE_FIELDS:
        raise HTTPException(400, f"Unknown source. Use one of {list(SOURCE_FIELDS)}")
    if payload.consent_id:
        _check_consent(msme_id, payload.consent_id)
    unknown = set(payload.fields) - set(SOURCE_FIELDS[source])
    if unknown:
        raise HTTPException(422, f"Fields not provided by {source}: {sorted(unknown)}")
    rec = store.get_data(msme_id) or {ID_COL: msme_id}
    rec.update(payload.fields)
    store.put_data(msme_id, rec)
    store.audit(msme_id, "source_ingested", {"source": source, "consent_id": payload.consent_id})
    missing = sorted(REQUIRED - set(k for k, v in rec.items() if v is not None))
    return {"msme_id": msme_id, "source": source, "ready_to_score": not missing, "missing_fields": missing}


def _check_consent(msme_id: str, consent_id: str) -> None:
    """Consent first: the artefact must exist, belong to this business, be active and not expired."""
    c = store.get_consent(consent_id)
    if not c:
        raise HTTPException(403, "Unknown consent_id. Create one with POST /api/v1/consents.")
    if c["msme_id"] != msme_id:
        raise HTTPException(403, f"Consent {consent_id} was given for {c['msme_id']}, not {msme_id}.")
    if c["status"] != "ACTIVE" or datetime.fromisoformat(c["valid_until"]) < datetime.now(timezone.utc):
        raise HTTPException(403, f"Consent {consent_id} is no longer active.")


@app.post("/api/v1/msmes/{msme_id}/fetch", tags=["ingestion"])
def fetch_all_sources(msme_id: str, consent_id: str = Query(..., description="From POST /api/v1/consents")):
    """Pull every source through its connector (mocked from the dataset) and store the merged record.
    Needs a consent artefact for this business (consent first)."""
    msme_id = msme_id.upper()
    _check_consent(msme_id, consent_id)
    rec, fetched = {ID_COL: msme_id}, {}
    for name, conn in connectors.items():
        try:
            rec.update(conn.fetch(msme_id, consent_id))
            fetched[name] = "ok"
        except KeyError:
            fetched[name] = "not found"
    if all(v != "ok" for v in fetched.values()):
        raise HTTPException(404, f"{msme_id} not found in any source")
    store.put_data(msme_id, rec)
    store.audit(msme_id, "sources_fetched", fetched)
    return {"msme_id": msme_id, "sources": fetched,
            "production_integration": {k: REAL_INTEGRATION[k] for k in fetched}}


# ---------------- Scoring ----------------
@app.post("/api/v1/score", tags=["scoring"])
def score_realtime(data: MSMEData):
    """Stateless real-time scoring: send the data, get the full Health Card back."""
    return engine.health_card(data.model_dump())


@app.post("/api/v1/msmes/{msme_id}/score", response_model=ScoreSummary, tags=["scoring"])
def score_msme(msme_id: str):
    """Score a stored business and save its Health Card."""
    rec = _record(msme_id)
    missing = sorted(REQUIRED - set(k for k, v in rec.items() if v is not None))
    if missing:
        raise HTTPException(422, f"Cannot score yet, missing: {missing}")
    card = engine.health_card(rec)
    ts = store.put_card(msme_id, card)
    store.audit(msme_id, "scored", {"score": card["financial_health_score"], "decision": card["credit_recommendation"]["decision"]})
    return _summary(card, ts)


def _card(msme_id: str) -> dict:
    card, _ = store.get_card(msme_id)
    if card is None:  # score on demand
        card = engine.health_card(_record(msme_id))
        store.put_card(msme_id, card)
    return card


@app.get("/api/v1/msmes/{msme_id}/health-card", tags=["health card"])
def get_health_card(msme_id: str):
    return _card(msme_id)


@app.get("/api/v1/msmes/{msme_id}/credit-recommendation", tags=["credit"])
def get_credit_recommendation(msme_id: str):
    card = _card(msme_id)
    return {"msme_id": msme_id, "financial_health_score": card["financial_health_score"],
            **card["credit_recommendation"], "risk_reason_codes": card["explanation"]["risk_reason_codes"]}


# ---------------- Dashboard integration ----------------
@app.get("/api/v1/dashboard/msmes/{msme_id}", tags=["dashboard"])
def dashboard_msme(msme_id: str):
    """Everything the single-business dashboard needs in one call."""
    card = _card(msme_id)
    rec = _record(msme_id)
    return {"card": card, "raw": rec}


@app.get("/api/v1/dashboard/portfolio", tags=["dashboard"])
def dashboard_portfolio(segment: str | None = None, industry: str | None = None):
    s = engine.scored
    if segment:
        s = s[s.Customer_Segment == segment]
    if industry:
        s = s[s.Industry == industry]
    if s.empty:
        raise HTTPException(404, "No businesses match the filter")
    return {
        "count": int(len(s)),
        "avg_score": round(float(s.score_overall.mean()), 1),
        "risk_band_mix": s.risk_band.value_counts(normalize=True).round(4).to_dict(),
        "approval_rate": round(float((s.risk_band != "High").mean()), 4),
        "approval_rate_by_segment": s.groupby("Customer_Segment").risk_band.apply(lambda x: round(float((x != "High").mean()), 4)).to_dict(),
        "avg_pillars": {name: round(float(s["score_" + c.replace("_Score", "")].mean()), 1) for c, name in PILLARS.items()},
        "flagged_for_review": int(((s.rule_flag_count > 0) | s.is_outlier).sum()),
    }


@app.get("/api/v1/dashboard/msmes", tags=["dashboard"])
def list_msmes(segment: str | None = None, risk_band: str | None = None, industry: str | None = None,
               limit: int = Query(50, le=500), offset: int = 0):
    s = engine.scored
    for col, val in [("Customer_Segment", segment), ("risk_band", risk_band), ("Industry", industry)]:
        if val:
            s = s[s[col] == val]
    page = s.iloc[offset:offset + limit]
    return {"total": int(len(s)), "items": page[[ID_COL, "Customer_Segment", "Industry", "Location_Category",
                                                  "score_overall", "pred_pd", "risk_band"]].round(4).to_dict("records")}
