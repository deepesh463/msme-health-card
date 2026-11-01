"""API request/response models with validation."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

Segment = Literal["NTC", "NTB", "Existing-to-Credit"]
Industry = Literal["Retail", "Services", "Manufacturing", "Trading", "Food & Beverage",
                   "Logistics", "Construction", "Textiles", "IT & Digital", "Healthcare"]


class MSMEData(BaseModel):
    """One business's consolidated data (all sources). Monetary values are monthly INR unless noted."""
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
    msme_id: str
    purpose: str = "MSME credit underwriting"
    fi_types: list[str] = Field(default=["DEPOSIT", "GSTR1_3B", "EPF"], description="Data types requested (ReBIT FI types)")
    data_range_months: int = Field(12, ge=1, le=36)
    consent_validity_days: int = Field(90, ge=1, le=365)


class SourcePayload(BaseModel):
    """Partial data from one source, merged into the business's record."""
    consent_id: Optional[str] = None
    fields: dict[str, float | str | None]


class ScoreSummary(BaseModel):
    msme_id: str
    financial_health_score: float
    grade: str
    probability_of_default: float
    risk_band: str
    decision: str
    recommended_limit_inr: int
    scored_at: str
