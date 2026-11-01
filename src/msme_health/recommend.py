"""Credit decision: risk band, eligibility, limit, pricing and conditions.

Two limits are computed and the lower one wins:
  * Policy limit       - % of annual turnover by risk band (matches the company's data)
  * Affordability limit - the loan whose EMI fits in half of the monthly cash surplus
The seasonality stress test then checks the lean month separately.
"""
from __future__ import annotations

import math

from .config import POLICY, RISK_BANDS

PRODUCT_BY_INDUSTRY = {
    "Logistics": "Commercial vehicle / fleet finance (Cars24 used CVs)",
    "Trading": "Inventory & working-capital line",
    "Retail": "Inventory & working-capital line",
    "Manufacturing": "Term loan for equipment",
    "Construction": "Equipment & vehicle finance",
    "Food & Beverage": "Working-capital line",
    "Textiles": "Working-capital line",
    "Services": "Business term loan",
    "IT & Digital": "Business term loan",
    "Healthcare": "Equipment finance",
}


def risk_band(pd_value: float) -> str:
    return next(name for cut, name in RISK_BANDS if pd_value < cut)


def emi_to_principal(emi: float, annual_rate_pct: float, months: int) -> float:
    r = annual_rate_pct / 1200
    if emi <= 0 or months <= 0:
        return 0.0
    return emi * (1 - (1 + r) ** -months) / r


def principal_to_emi(principal: float, annual_rate_pct: float, months: int) -> float:
    r = annual_rate_pct / 1200
    if principal <= 0 or months <= 0:
        return 0.0
    return principal * r / (1 - (1 + r) ** -months)


def recommend(row: dict, pd_value: float, anomaly_flags: list[str], is_outlier: bool) -> dict:
    band = risk_band(pd_value)
    step = POLICY["round_to"]
    turnover = float(row["Annual_Turnover_INR"])
    surplus = float(row["Monthly_Bank_Credits_INR"]) - float(row["Monthly_Bank_Debits_INR"])
    lean_surplus = float(row.get("Lean_Month_Surplus_INR", surplus))

    policy_limit = turnover * POLICY["limit_pct_of_turnover"][band]
    tenure = POLICY["tenure_months"][band]
    rate = POLICY["annual_rate_pct"][band]
    affordable_emi = max(surplus, 0) * POLICY["max_new_emi_share_of_surplus"]
    afford_limit = emi_to_principal(affordable_emi, rate, tenure) if rate else 0.0

    limit = math.floor(min(policy_limit, afford_limit) / step) * step
    binding = "policy (% of turnover)" if policy_limit <= afford_limit else "affordability (cash surplus)"
    emi = principal_to_emi(limit, rate, tenure) if rate else 0.0

    conditions, reasons = [], []
    if band == "High":
        decision = "Decline"
        reasons.append(f"Predicted default probability {pd_value:.1%} is above the 15% cut-off")
    elif limit <= 0:
        decision = "Decline"
        reasons.append("No monthly cash surplus to service a new EMI")
    elif anomaly_flags or is_outlier:
        decision = "Refer to underwriter"
        reasons.append("Data consistency checks raised flags that need manual review")
    elif band == "Medium":
        decision = "Approve with conditions"
    else:
        decision = "Approve"

    if decision not in ("Decline",):
        if lean_surplus < emi:
            conditions.append("Seasonal stress: lean-month cash may not cover the EMI. Offer a seasonal or step-down EMI.")
        if band == "Medium":
            conditions.append("Monthly monitoring of bank flows through the Account Aggregator")
        if float(row.get("Customer_Concentration_Ratio", 0)) > 0.6:
            conditions.append("High dependence on few customers: consider invoice-backed (TReDS) financing")
        if int(row.get("Thin_File", 0)) == 1:
            conditions.append("New-to-credit: start with a smaller ticket and step up after 6 clean EMIs")

    return {
        "risk_band": band,
        "probability_of_default": round(pd_value, 4),
        "eligible": decision != "Decline",
        "decision": decision,
        "recommended_limit_inr": int(limit),
        "policy_limit_inr": int(policy_limit),
        "affordability_limit_inr": int(afford_limit),
        "binding_constraint": binding if limit > 0 else None,
        "tenure_months": tenure,
        "indicative_rate_pct": rate,
        "estimated_emi_inr": round(emi),
        "monthly_surplus_inr": round(surplus),
        "lean_month_surplus_inr": round(lean_surplus),
        "suggested_product": PRODUCT_BY_INDUSTRY.get(row.get("Industry"), "Business term loan") if decision != "Decline" else None,
        "conditions": conditions,
        "reasons": reasons,
    }
