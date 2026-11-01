"""Peer benchmarking: compare a business with others in the same industry and city tier."""
from __future__ import annotations

import numpy as np
import pandas as pd

PEER_KEYS = ["Industry", "Location_Category"]
MIN_PEERS = 200  # fall back to industry-only peers below this size

# metric -> True if higher is better
PEER_METRICS = {
    "Monthly_GST_Sales_INR": True,
    "Monthly_UPI_Inflow_INR": True,
    "UPI_Daily_Txn_Count": True,
    "Monthly_Bank_Credits_INR": True,
    "Average_Bank_Balance_INR": True,
    "Net_Surplus_Ratio": True,
    "Lean_Month_Surplus_Ratio": True,
    "Cashflow_Stability_Index": True,
    "Overdraft_Usage_Ratio": False,
    "Transaction_Volatility_Index": False,
    "Seasonality_Index": False,
    "Revenue_Growth_Rate_Pct": True,
    "GST_Filing_Timeliness_Pct": True,
    "Salary_Consistency_Pct": True,
    "Employee_Attrition_Rate_Pct": False,
    "Avg_Invoice_Payment_Delay_Days": False,
    "Vendor_Payment_Timeliness_Pct": True,
    "Monthly_Payroll_INR": True,
}


class PeerBenchmark:
    """Holds sorted values per peer group so percentiles are a fast lookup."""

    def __init__(self, frame: pd.DataFrame):
        cols = PEER_KEYS + [m for m in PEER_METRICS if m in frame.columns] + [
            c for c in frame.columns if c.startswith("score_")
        ]
        self.frame = frame[cols].copy()
        self._groups = {k: g for k, g in self.frame.groupby(PEER_KEYS, observed=True)}
        self._industry = {k: g for k, g in self.frame.groupby("Industry", observed=True)}

    def peers_for(self, industry: str, location: str) -> tuple[pd.DataFrame, str]:
        g = self._groups.get((industry, location))
        if g is not None and len(g) >= MIN_PEERS:
            return g, f"{industry} businesses in {location}"
        g = self._industry.get(industry)
        if g is not None and len(g) >= MIN_PEERS:
            return g, f"{industry} businesses (all locations)"
        return self.frame, "all businesses"

    @staticmethod
    def percentile(values: pd.Series, x: float) -> float | None:
        v = np.sort(values.dropna().to_numpy())
        if len(v) == 0 or x is None or (isinstance(x, float) and np.isnan(x)):
            return None
        return float(np.searchsorted(v, x, side="right") / len(v) * 100)

    def compare(self, row: dict, metrics: dict[str, bool] | None = None) -> dict:
        metrics = metrics or PEER_METRICS
        peers, label = self.peers_for(row["Industry"], row["Location_Category"])
        out = []
        for m, higher_better in metrics.items():
            if m not in peers.columns or m not in row:
                continue
            x = row[m]
            pct = self.percentile(peers[m], x)
            if pct is None:
                continue
            q = peers[m].quantile([0.25, 0.5, 0.75])
            # "standing": 0-100 where higher always means better than peers
            standing = pct if higher_better else 100 - pct
            out.append({
                "metric": m,
                "value": float(x),
                "peer_p25": float(q.loc[0.25]),
                "peer_median": float(q.loc[0.5]),
                "peer_p75": float(q.loc[0.75]),
                "percentile": round(pct, 1),
                "standing": round(standing, 1),
                "higher_is_better": higher_better,
            })
        return {"peer_group": label, "peer_count": int(len(peers)), "metrics": out}
