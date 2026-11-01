"""API request/response models with validation."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

# A real business from the dataset, so "Try it out" in /docs works without editing.
EXAMPLE_MSME = {
    "MSME_ID": "DEMO0000001", "Customer_Segment": "NTC", "Business_Type": "LLP",
    "Industry": "Logistics", "Location_Category": "Tier 2", "Years_in_Operation": 8.0,
    "Employee_Count": 2, "Annual_Turnover_INR": 5854430,
    "Monthly_GST_Sales_INR": 476042, "Monthly_GST_Purchases_INR": 327244,
    "GST_Filing_Timeliness_Pct": 85.7, "GST_Return_Frequency": "Monthly",
    "Monthly_UPI_Inflow_INR": 181309, "Monthly_UPI_Outflow_INR": 131501,
    "UPI_Avg_Ticket_Size_INR": 611, "UPI_Daily_Txn_Count": 11.4,
    "Transaction_Volatility_Index": 0.251, "Monthly_Bank_Credits_INR": 514240,
    "Monthly_Bank_Debits_INR": 483729, "Average_Bank_Balance_INR": 127948,
    "Cashflow_Stability_Index": 0.706, "Overdraft_Usage_Ratio": 0.231,
    "Has_Existing_Loan": "No", "Monthly_Loan_EMI_INR": 0, "EMI_On_Time_Rate_Pct": None,
    "Credit_History_Months": 0, "Monthly_Payroll_INR": 40508, "Salary_Consistency_Pct": 87.0,
    "Employee_Attrition_Rate_Pct": 23.1, "Avg_Invoice_Payment_Delay_Days": 34.0,
    "Customer_Concentration_Ratio": 0.45, "Vendor_Payment_Timeliness_Pct": 90.4,
    "Seasonality_Index": 0.243, "Revenue_Growth_Rate_Pct": 31.1,
}

Segment = Literal["NTC", "NTB", "Existing-to-Credit"]
Industry = Literal["Retail", "Services", "Manufacturing", "Trading", "Food & Beverage",
                   "Logistics", "Construction", "Textiles", "IT & Digital", "Healthcare"]


class MSMEData(BaseModel):
    """One business's consolidated data (all sources). Monetary values are monthly INR unless noted."""
    model_config = ConfigDict(json_schema_extra={"examples": [EXAMPLE_MSME]})
    MSME_ID: str = Field(..., examples=["MSME0000001"])
    Customer_Segment: Segment
    Business_Type: Literal["Proprietorship", "Partnership", "LLP", "Private Limited"]
    Industry: Industry
    Location_Category: Literal["Metro", "Tier 1", "Tier 2", "Tier 3", "Rural"]
    Years_in_Operation: float = Field(..., ge=0, le=100)
    Employee_Count: int = Field(..., ge=0)
    Annual_Turnover_INR: float = Field(..., gt=0)
    # GSTN
    Monthly_GST_Sales_INR: float = Field(..., ge=0)
    Monthly_GST_Purchases_INR: float = Field(..., ge=0)
    GST_Filing_Timeliness_Pct: float = Field(..., ge=0, le=100)
    GST_Return_Frequency: Literal["Monthly", "Quarterly"]
    # UPI
    Monthly_UPI_Inflow_INR: float = Field(..., ge=0)
    Monthly_UPI_Outflow_INR: float = Field(..., ge=0)
    UPI_Avg_Ticket_Size_INR: float = Field(..., ge=0)
    UPI_Daily_Txn_Count: float = Field(..., ge=0)
    # Account Aggregator (bank statements)
    Transaction_Volatility_Index: float = Field(..., ge=0, le=1)
    Monthly_Bank_Credits_INR: float = Field(..., ge=0)
    Monthly_Bank_Debits_INR: float = Field(..., ge=0)
    Average_Bank_Balance_INR: float = Field(..., ge=0)
    Cashflow_Stability_Index: float = Field(..., ge=0, le=1)
    Overdraft_Usage_Ratio: float = Field(..., ge=0, le=1)
    # Credit
    Has_Existing_Loan: Literal["Yes", "No"]
    Monthly_Loan_EMI_INR: float = Field(0, ge=0)
    EMI_On_Time_Rate_Pct: Optional[float] = Field(None, ge=0, le=100)
    Credit_History_Months: float = Field(0, ge=0)
    # EPFO
    Monthly_Payroll_INR: float = Field(..., ge=0)
    Salary_Consistency_Pct: float = Field(..., ge=0, le=100)
    Employee_Attrition_Rate_Pct: float = Field(..., ge=0, le=100)
    # Invoices
    Avg_Invoice_Payment_Delay_Days: float = Field(..., ge=0)
    Customer_Concentration_Ratio: float = Field(..., ge=0, le=1)
    Vendor_Payment_Timeliness_Pct: float = Field(..., ge=0, le=100)
    # Derived from GST history
    Seasonality_Index: float = Field(..., ge=0, le=1)
    Revenue_Growth_Rate_Pct: float


class ConsentRequest(BaseModel):
    msme_id: str = Field(..., examples=["MSME0000001"])
    purpose: str = "MSME credit underwriting"
    fi_types: list[str] = Field(default=["DEPOSIT", "GSTR1_3B", "EPF"], description="Data types requested (ReBIT FI types)")
    data_range_months: int = Field(12, ge=1, le=36)
    consent_validity_days: int = Field(90, ge=1, le=365)


class SourcePayload(BaseModel):
    """Partial data from one source, merged into the business's record."""
    consent_id: Optional[str] = None
    fields: dict[str, float | str | None] = Field(
        ..., examples=[{"GST_Filing_Timeliness_Pct": 95.0, "Avg_Invoice_Payment_Delay_Days": 20}])


class ScoreSummary(BaseModel):
    msme_id: str
    financial_health_score: float
    grade: str
    probability_of_default: float
    risk_band: str
    decision: str
    recommended_limit_inr: int
    scored_at: str
