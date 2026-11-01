"""Data-source connectors.

Each external source sits behind the same small interface. Today they are mocks that
read from the provided dataset; swapping in a real Account Aggregator / GSTN / EPFO client
means writing one class, with no change to scoring.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

# Which dataset fields each source provides.
SOURCE_FIELDS = {
    "gstn": ["Annual_Turnover_INR", "Monthly_GST_Sales_INR", "Monthly_GST_Purchases_INR",
             "GST_Filing_Timeliness_Pct", "GST_Return_Frequency", "Seasonality_Index", "Revenue_Growth_Rate_Pct"],
    "upi": ["Monthly_UPI_Inflow_INR", "Monthly_UPI_Outflow_INR", "UPI_Avg_Ticket_Size_INR", "UPI_Daily_Txn_Count"],
    "account_aggregator": ["Monthly_Bank_Credits_INR", "Monthly_Bank_Debits_INR", "Average_Bank_Balance_INR",
                           "Cashflow_Stability_Index", "Overdraft_Usage_Ratio", "Transaction_Volatility_Index"],
    "credit_bureau": ["Has_Existing_Loan", "Monthly_Loan_EMI_INR", "EMI_On_Time_Rate_Pct", "Credit_History_Months"],
    "epfo": ["Employee_Count", "Monthly_Payroll_INR", "Salary_Consistency_Pct", "Employee_Attrition_Rate_Pct"],
    "invoices": ["Avg_Invoice_Payment_Delay_Days", "Customer_Concentration_Ratio", "Vendor_Payment_Timeliness_Pct"],
    "profile": ["Customer_Segment", "Business_Type", "Industry", "Location_Category", "Years_in_Operation"],
}

REAL_INTEGRATION = {
    "gstn": "GSTN via a GST Suvidha Provider (GSP) API, taxpayer OTP consent; GSTR-1/3B returns",
    "upi": "Bank / PSP merchant settlement data, or UPI flows inside AA bank statements",
    "account_aggregator": "Sahamati AA network as an FIU: consent request -> FI fetch -> decrypt ReBIT JSON",
    "credit_bureau": "CIBIL / Experian / CRIF commercial bureau API (empty for NTC)",
    "epfo": "EPFO establishment data via AA (EPF FI type) or ULI",
    "invoices": "GST e-invoice (IRP) and TReDS platforms",
    "profile": "Udyam registration + MCA21 for companies",
}


class Connector(ABC):
    source: str

    @abstractmethod
    def fetch(self, msme_id: str, consent_id: str | None) -> dict:
        ...


class MockConnector(Connector):
    """Serves one source's fields for an MSME from the provided dataset."""

    def __init__(self, source: str, table: pd.DataFrame):
        self.source = source
        self.table = table

    def fetch(self, msme_id: str, consent_id: str | None) -> dict:
        if msme_id not in self.table.index:
            raise KeyError(msme_id)
        row = self.table.loc[msme_id, SOURCE_FIELDS[self.source]]
        return {k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v)) for k, v in row.items()}


def build_connectors(table: pd.DataFrame) -> dict[str, Connector]:
    return {s: MockConnector(s, table) for s in SOURCE_FIELDS}
