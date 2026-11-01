"""Anomaly detection: an Isolation Forest plus plain consistency rules.

There are no fraud labels in the data, so we use two unsupervised layers:
1. Rules that check whether the data sources agree with each other.
2. An Isolation Forest that flags businesses whose overall profile is unusual.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .config import RANDOM_STATE

IF_FEATURES = [
    "GST_Bank_Gap", "GST_Turnover_Gap", "UPI_Share_of_Credits", "Net_Surplus_Ratio",
    "Balance_Cover_Months", "EMI_Burden", "Gross_Margin_GST", "Payroll_to_Sales",
    "Overdraft_Usage_Ratio", "Transaction_Volatility_Index", "Cashflow_Stability_Index",
    "UPI_Daily_Txn_Count", "Log_UPI_Ticket", "Log_Revenue_per_Employee",
]

# Rule thresholds sit just outside the 1st/99th percentile of the clean data.
RULES = [
    ("GST_Bank_Gap", ">", 0.25, "Bank credits are much higher than GST sales: possible circular or unreported transactions"),
    ("GST_Bank_Gap", "<", -0.20, "Bank credits are much lower than GST sales: possible inflated GST sales"),
    ("GST_Turnover_Gap", ">", 0.25, "GST sales are much higher than declared turnover"),
    ("GST_Turnover_Gap", "<", -0.30, "Declared turnover is much higher than GST sales"),
    ("UPI_Share_of_Credits", ">", 1.0, "UPI inflow exceeds total bank credits: sources disagree"),
    ("Payroll_to_Sales", ">", 1.0, "Payroll exceeds monthly sales: possible ghost payroll or loss-making"),
    ("Net_Surplus_Ratio", "<", -0.10, "Spending more than earning each month"),
    ("EMI_Burden", ">", 0.45, "Existing EMIs take more than 45% of income"),
]


def _if_frame(feat: pd.DataFrame) -> pd.DataFrame:
    x = feat.copy()
    x["Log_UPI_Ticket"] = np.log1p(x["UPI_Avg_Ticket_Size_INR"])
    x["Log_Revenue_per_Employee"] = np.log1p(x["Revenue_per_Employee"])
    return x[IF_FEATURES].astype(float).fillna(0.0)


def rule_flags(row: pd.Series | dict) -> list[str]:
    flags = []
    for col, op, thr, msg in RULES:
        v = row.get(col)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        if (op == ">" and v > thr) or (op == "<" and v < thr):
            flags.append(msg)
    return flags


class AnomalyDetector:
    def __init__(self, contamination: float = 0.02):
        self.model = IsolationForest(
            n_estimators=300, contamination=contamination, random_state=RANDOM_STATE, n_jobs=-1
        )
        self.ref_scores: np.ndarray | None = None

    def fit(self, feat: pd.DataFrame) -> "AnomalyDetector":
        x = _if_frame(feat)
        self.model.fit(x)
        # Sorted reference scores: lets us express any new score as a 0-100 "unusualness" percentile.
        self.ref_scores = np.sort(-self.model.score_samples(x))
        return self

    def score(self, feat: pd.DataFrame) -> pd.DataFrame:
        x = _if_frame(feat)
        raw = -self.model.score_samples(x)  # higher = more unusual
        pct = np.searchsorted(self.ref_scores, raw, side="right") / len(self.ref_scores) * 100
        is_outlier = self.model.predict(x) == -1
        return pd.DataFrame({"anomaly_percentile": pct.round(1), "is_outlier": is_outlier}, index=feat.index)
