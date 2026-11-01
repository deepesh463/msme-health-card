"""Behavioural tests: the model must respond to inputs the way a credit officer expects."""
import pandas as pd
import pytest

from msme_health.config import DATA_PATH
from msme_health.engine import get_engine
from msme_health.features import add_derived
from msme_health.recommend import emi_to_principal, principal_to_emi, risk_band


@pytest.fixture(scope="module")
def eng():
    return get_engine()


@pytest.fixture(scope="module")
def row():
    return pd.read_csv(DATA_PATH, nrows=1).iloc[0].to_dict()


def test_risk_band_cutoffs():
    assert risk_band(0.03) == "Low" and risk_band(0.10) == "Medium" and risk_band(0.40) == "High"


def test_emi_round_trip():
    p = emi_to_principal(10000, 18, 24)
    assert abs(principal_to_emi(p, 18, 24) - 10000) < 1


@pytest.mark.parametrize("feature,better,worse", [
    ("GST_Filing_Timeliness_Pct", 99, 60),
    ("Overdraft_Usage_Ratio", 0.05, 0.55),
    ("Avg_Invoice_Payment_Delay_Days", 2, 60),
    ("Salary_Consistency_Pct", 99, 60),
])
def test_monotonic_behaviour(eng, row, feature, better, worse):
    good = eng.quick_score({**row, feature: better})
    bad = eng.quick_score({**row, feature: worse})
    assert good["pd"] <= bad["pd"]
    assert good["score"] >= bad["score"]


def test_derived_features_single_row(row):
    f = add_derived(pd.DataFrame([row])).iloc[0]
    assert -1 < f["Net_Surplus_Ratio"] < 1
    assert f["Lean_Month_Revenue_INR"] <= f["Monthly_GST_Sales_INR"] <= f["Peak_Month_Revenue_INR"]


def test_missing_emi_rate_is_handled(eng, row):
    card = eng.health_card({**row, "EMI_On_Time_Rate_Pct": None})
    assert 0 <= card["financial_health_score"] <= 100


def test_circular_transactions_flagged(eng, row):
    fraud = {**row, "Monthly_Bank_Credits_INR": row["Monthly_GST_Sales_INR"] * 2.5}
    card = eng.health_card(fraud)
    assert card["anomaly"]["flags"]
    assert card["credit_recommendation"]["decision"] in {"Refer to underwriter", "Decline"}
