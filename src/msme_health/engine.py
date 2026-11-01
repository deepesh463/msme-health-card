"""Scoring engine: one business's raw data in, a complete Health Card out.

The API and the dashboard both call this, so they always agree.
"""
from __future__ import annotations

from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from .anomaly import rule_flags
from .config import ARTIFACTS, CATEGORICAL, ID_COL, NUMERIC_RAW, PILLARS, SOURCE_OF
from .features import FEATURE_PILLAR, LABELS, MODEL_FEATURES, add_derived, prepare
from .peers import PEER_METRICS, PeerBenchmark
from .recommend import recommend, risk_band

# Things a business owner can actually change, with the direction that helps.
ACTIONABLE = {
    "GST_Filing_Timeliness_Pct": +1,
    "Overdraft_Usage_Ratio": -1,
    "Avg_Invoice_Payment_Delay_Days": -1,
    "Vendor_Payment_Timeliness_Pct": +1,
    "Salary_Consistency_Pct": +1,
    "Customer_Concentration_Ratio": -1,
    "EMI_On_Time_Rate_Pct": +1,
    "Cashflow_Stability_Index": +1,
    "Average_Bank_Balance_INR": +1,
}
ADVICE = {
    "GST_Filing_Timeliness_Pct": "File GST returns on or before the due date",
    "Overdraft_Usage_Ratio": "Reduce reliance on the overdraft",
    "Avg_Invoice_Payment_Delay_Days": "Collect from customers faster (e.g. TReDS invoice discounting)",
    "Vendor_Payment_Timeliness_Pct": "Pay suppliers on time",
    "Salary_Consistency_Pct": "Pay salaries on a fixed date every month",
    "Customer_Concentration_Ratio": "Spread sales across more customers",
    "EMI_On_Time_Rate_Pct": "Pay every EMI on time",
    "Cashflow_Stability_Index": "Smooth out monthly cash flows",
    "Average_Bank_Balance_INR": "Keep a higher average bank balance",
}


def grade(score: float) -> str:
    for cut, g in [(85, "A+"), (78, "A"), (70, "B"), (62, "C")]:
        if score >= cut:
            return g
    return "D"


def grade_from_pct(pct: float) -> str:
    """Pillar grades are relative: each pillar has a different natural range."""
    for cut, g in [(90, "A+"), (70, "A"), (40, "B"), (15, "C")]:
        if pct >= cut:
            return g
    return "D"


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


def _clean(v):
    if isinstance(v, (np.floating, float)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, np.integer):
        return int(v)
    return v


class HealthCardEngine:
    def __init__(self, artifacts=ARTIFACTS):
        m = joblib.load(artifacts / "models.joblib")
        self.pd_model, self.fhs_model, self.pillar_models = m["pd"], m["fhs"], m["pillars"]
        self.anomaly = joblib.load(artifacts / "anomaly.joblib")
        scored_path = artifacts / "scored.parquet"
        self.scored = pd.read_parquet(scored_path) if scored_path.exists() else None
        if self.scored is not None:  # text columns as categories: saves ~70 MB of memory
            for c in self.scored.columns:
                if self.scored[c].dtype == object or str(self.scored[c].dtype) in ("string", "str"):
                    if c != ID_COL:
                        self.scored[c] = self.scored[c].astype("category")
        self.peers = PeerBenchmark(self.scored) if self.scored is not None else None

    # ---------- vectorised scoring (batch / portfolio) ----------
    def score_frame(self, raw: pd.DataFrame) -> pd.DataFrame:
        X = prepare(raw)
        feat = add_derived(raw)
        out = pd.DataFrame(index=raw.index)
        out["pred_pd"] = _sigmoid(self.pd_model.predict(X))
        out["score_overall"] = np.clip(self.fhs_model.predict(X), 0, 100)
        for col, name in PILLARS.items():
            out["score_" + col.replace("_Score", "")] = np.clip(self.pillar_models[col].predict(X), 0, 100)
        out["risk_band"] = [risk_band(p) for p in out["pred_pd"]]
        a = self.anomaly.score(feat)
        out["anomaly_percentile"] = a["anomaly_percentile"]
        out["is_outlier"] = a["is_outlier"]
        out["rule_flag_count"] = feat.apply(lambda r: len(rule_flags(r)), axis=1)
        return out

    # ---------- single business: full card ----------
    def _predict_one(self, raw_row: dict) -> tuple[pd.DataFrame, float, float]:
        X = prepare(pd.DataFrame([raw_row]))
        return X, float(_sigmoid(self.pd_model.predict(X))[0]), float(np.clip(self.fhs_model.predict(X)[0], 0, 100))

    def quick_score(self, raw_row: dict) -> dict:
        """Fast path for the what-if simulator."""
        X, p, s = self._predict_one(raw_row)
        pillars = {name: float(np.clip(self.pillar_models[c].predict(X)[0], 0, 100)) for c, name in PILLARS.items()}
        return {"score": s, "pd": p, "risk_band": risk_band(p), "pillars": pillars}

    def _drivers(self, model, X, feat_row, top=6, sign=1):
        """Exact TreeSHAP values computed by LightGBM itself (no shap library needed at serving time)."""
        contrib = np.asarray(model.predict(X, pred_contrib=True))[0]
        sv = contrib[:-1] * sign  # last column is the expected (average) value
        df = pd.DataFrame({"feature": MODEL_FEATURES, "impact": sv})
        df["label"] = df.feature.map(LABELS).fillna(df.feature)
        df["pillar"] = df.feature.map(FEATURE_PILLAR).fillna("Other")
        df["source"] = df.feature.map(SOURCE_OF).fillna("Derived / profile")
        df["value"] = [_clean(feat_row.get(f)) for f in df.feature]
        base = float(contrib[-1]) * sign
        return base, df.sort_values("impact", key=abs, ascending=False).head(top), df

    def health_card(self, raw_row: dict) -> dict:
        raw_row = {k: _clean(v) for k, v in raw_row.items()}
        feat_row = add_derived(pd.DataFrame([raw_row])).iloc[0].to_dict()
        feat_row = {k: _clean(v) for k, v in feat_row.items()}
        X, p, s = self._predict_one(raw_row)

        # Pillars with their own top drivers
        pillars = []
        for col, name in PILLARS.items():
            val = float(np.clip(self.pillar_models[col].predict(X)[0], 0, 100))
            _, top, _ = self._drivers(self.pillar_models[col], X, feat_row, top=2)
            pct = self._population_pct(col, val)
            pillars.append({
                "key": col, "name": name, "score": round(val, 1), "grade": grade_from_pct(pct) if pct is not None else grade(val),
                "population_percentile": pct,
                "drivers": [{"label": r.label, "impact": round(r.impact, 2)} for r in top.itertuples()],
            })

        # Why this overall score? SHAP in score points.
        base_s, top_s, all_s = self._drivers(self.fhs_model, X, feat_row, top=8)
        # Why this risk? SHAP on log-odds of default (positive = riskier).
        _, top_p, _ = self._drivers(self.pd_model, X, feat_row, top=8)
        pillar_contrib = all_s.groupby("pillar")["impact"].sum().round(2).to_dict()

        a = self.anomaly.score(pd.DataFrame([feat_row]))
        flags = rule_flags(feat_row)
        is_out = bool(a["is_outlier"].iloc[0])
        rec = recommend(feat_row, p, flags, is_out)

        peer = self.peers.compare(feat_row) if self.peers else None
        risk_indicators = self._risk_indicators(feat_row, flags, is_out, float(a["anomaly_percentile"].iloc[0]))
        insights = self._insights(raw_row, feat_row, pillars, top_s, peer, p, s)

        return {
            "msme_id": raw_row.get(ID_COL),
            "profile": {k: raw_row.get(k) for k in CATEGORICAL + ["Years_in_Operation", "Employee_Count", "Annual_Turnover_INR", "Credit_History_Months"]},
            "financial_health_score": round(s, 1),
            "grade": grade(s),
            "probability_of_default": round(p, 4),
            "risk_band": rec["risk_band"],
            "pillars": pillars,
            "explanation": {
                "base_score": round(base_s, 1),
                "top_drivers": [
                    {"feature": r.feature, "label": r.label, "value": r.value, "impact_points": round(r.impact, 2),
                     "pillar": r.pillar, "source": r.source}
                    for r in top_s.itertuples()
                ],
                "pillar_contribution_points": pillar_contrib,
                "risk_reason_codes": [
                    {"feature": r.feature, "label": r.label, "value": r.value, "impact_log_odds": round(r.impact, 3)}
                    for r in top_p.itertuples() if r.impact > 0
                ][:4],
            },
            "risk_indicators": risk_indicators,
            "anomaly": {"percentile": float(a["anomaly_percentile"].iloc[0]), "is_outlier": is_out, "flags": flags},
            "credit_recommendation": rec,
            "seasonality": {
                "seasonality_index": feat_row["Seasonality_Index"],
                "avg_month_revenue_inr": feat_row["Monthly_GST_Sales_INR"],
                "lean_month_revenue_inr": round(feat_row["Lean_Month_Revenue_INR"]),
                "peak_month_revenue_inr": round(feat_row["Peak_Month_Revenue_INR"]),
                "avg_month_surplus_inr": round(feat_row["Monthly_Bank_Credits_INR"] - feat_row["Monthly_Bank_Debits_INR"]),
                "lean_month_surplus_inr": round(feat_row["Lean_Month_Surplus_INR"]),
            },
            "peer_benchmark": peer,
            "insights": insights,
            "data_coverage": self._coverage(raw_row),
        }

    # ---------- helpers ----------
    def _population_pct(self, pillar_col: str, value: float):
        """Where this pillar score sits among all 50k businesses (0-100)."""
        if self.scored is None:
            return None
        if not hasattr(self, "_sorted_pillars"):
            self._sorted_pillars = {
                c: np.sort(self.scored["score_" + c.replace("_Score", "")].to_numpy()) for c in PILLARS
            }
        v = self._sorted_pillars[pillar_col]
        return round(float(np.searchsorted(v, value) / len(v) * 100), 1)

    @staticmethod
    def _risk_indicators(f, flags, is_out, anomaly_pct):
        credits = f["Monthly_Bank_Credits_INR"] or 0
        # (label, feature, red if, amber if, format)
        checks = [
            ("Overdraft usage", "Overdraft_Usage_Ratio", lambda v: v > 0.40, lambda v: v > 0.30, "{:.0%}"),
            ("Transaction volatility", "Transaction_Volatility_Index", lambda v: v > 0.45, lambda v: v > 0.35, "{:.2f}"),
            ("Customer concentration", "Customer_Concentration_Ratio", lambda v: v > 0.70, lambda v: v > 0.55, "{:.0%}"),
            ("Days customers take to pay", "Avg_Invoice_Payment_Delay_Days", lambda v: v > 30, lambda v: v > 20, "{:.0f} days"),
            ("EMI share of income", "EMI_Burden", lambda v: v > 0.30, lambda v: v > 0.20, "{:.0%}"),
            ("Lean-month cash surplus", "Lean_Month_Surplus_INR", lambda v: v < 0, lambda v: v < 0.05 * credits, "₹{:,.0f}"),
            ("Revenue growth", "Revenue_Growth_Rate_Pct", lambda v: v < -10, lambda v: v < 0, "{:.1f}%"),
            ("Employee attrition", "Employee_Attrition_Rate_Pct", lambda v: v > 30, lambda v: v > 22, "{:.1f}%"),
        ]
        out = []
        for name, key, red, amber, fmt in checks:
            val = f.get(key)
            if val is None or (isinstance(val, float) and not np.isfinite(val)):
                # Can't be computed (e.g. no bank credits): flag it rather than fail.
                out.append({"indicator": name, "value": "n/a", "level": "Watch"})
                continue
            level = "High" if red(val) else ("Watch" if amber(val) else "OK")
            out.append({"indicator": name, "value": fmt.format(val), "level": level})
        out.append({"indicator": "Data consistency", "value": f"{len(flags)} flag(s) · unusualness {anomaly_pct:.0f}/100",
                    "level": "High" if flags else ("Watch" if is_out else "OK")})
        return out

    def _insights(self, raw_row, f, pillars, top_s, peer, p, s):
        strengths, concerns = [], []
        for r in top_s.itertuples():
            (strengths if r.impact > 0 else concerns).append(f"{r.label} ({'+' if r.impact > 0 else ''}{r.impact:.1f} pts)")
        best = max(pillars, key=lambda x: x["score"])
        worst = min(pillars, key=lambda x: x["score"])

        # What-if: move each actionable metric to the peer 75th-percentile level and re-score.
        actions = []
        peer_lookup = {m["metric"]: m for m in (peer or {}).get("metrics", [])}
        for feat, direction in ACTIONABLE.items():
            cur = raw_row.get(feat)
            if cur is None:
                continue
            target = None
            if feat in peer_lookup:
                target = peer_lookup[feat]["peer_p75"] if direction > 0 else peer_lookup[feat]["peer_p25"]
            elif feat == "EMI_On_Time_Rate_Pct":
                target = 98.0
            elif feat == "Customer_Concentration_Ratio":
                target = 0.30
            if target is None or (direction > 0 and target <= cur) or (direction < 0 and target >= cur):
                continue
            new = dict(raw_row)
            new[feat] = target
            _, p2, s2 = self._predict_one(new)
            if s2 - s > 0.3 or p - p2 > 0.003:
                actions.append({
                    "action": ADVICE[feat], "feature": feat, "from": round(cur, 2), "to": round(target, 2),
                    "score_gain": round(s2 - s, 1), "pd_drop_pct_points": round((p - p2) * 100, 2),
                })
        actions.sort(key=lambda a: a["score_gain"], reverse=True)
        return {
            "strengths": strengths[:3],
            "concerns": concerns[:3],
            "strongest_pillar": best["name"],
            "weakest_pillar": worst["name"],
            "improvement_actions": actions[:4],
        }

    @staticmethod
    def _coverage(raw_row):
        sources = {
            "GSTN": raw_row.get("Monthly_GST_Sales_INR") is not None,
            "UPI": raw_row.get("Monthly_UPI_Inflow_INR") is not None,
            "Account Aggregator": raw_row.get("Monthly_Bank_Credits_INR") is not None,
            "EPFO": raw_row.get("Monthly_Payroll_INR") is not None,
            "Invoices": raw_row.get("Avg_Invoice_Payment_Delay_Days") is not None,
            "Credit Bureau": (raw_row.get("Credit_History_Months") or 0) > 0,
        }
        n = sum(sources.values())
        conf = "High" if n >= 6 else ("Medium" if n >= 5 else "Low")
        return {"sources": sources, "sources_available": n, "confidence": conf,
                "note": None if sources["Credit Bureau"] else "No bureau history: score relies fully on alternate data (NTC)."}


@lru_cache(maxsize=1)
def get_engine() -> HealthCardEngine:
    return HealthCardEngine()


def batch_score():
    """Score all 50k businesses once; used for peer benchmarks and the portfolio view."""
    from .config import DATA_PATH
    from .peers import PEER_KEYS
    raw = pd.read_csv(DATA_PATH)
    eng = HealthCardEngine()
    scores = eng.score_frame(raw)
    feat = add_derived(raw)
    keep = [ID_COL] + CATEGORICAL + NUMERIC_RAW + [c for c in PEER_METRICS if c not in NUMERIC_RAW] + [
        "Lean_Month_Surplus_INR", "EMI_Burden", "GST_Bank_Gap", "Financial_Health_Score",
        "Probability_of_Default", "Credit_Risk_Category", "Credit_Eligible", "Recommended_Credit_Limit_INR",
    ] + list(PILLARS)
    keep = list(dict.fromkeys(keep))
    out = pd.concat([feat[keep], scores], axis=1)
    recs = [
        recommend(r, r["pred_pd"], ["flag"] * int(r["rule_flag_count"]), bool(r["is_outlier"]))
        for r in out[["Annual_Turnover_INR", "Monthly_Bank_Credits_INR", "Monthly_Bank_Debits_INR",
                      "Lean_Month_Surplus_INR", "Customer_Concentration_Ratio", "Credit_History_Months",
                      "Industry", "pred_pd", "rule_flag_count", "is_outlier"]].to_dict("records")
    ]
    out["decision"] = [r["decision"] for r in recs]
    out["rec_limit"] = [r["recommended_limit_inr"] for r in recs]
    out["policy_limit"] = [r["policy_limit_inr"] for r in recs]
    out["binding_constraint"] = [r["binding_constraint"] for r in recs]
    out.to_parquet(ARTIFACTS / "scored.parquet", index=False)
    print("scored", out.shape, "peer keys", PEER_KEYS)


if __name__ == "__main__":
    batch_score()
