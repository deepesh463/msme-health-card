"""Feature engineering.

Turns the raw data into the ratios a credit analyst looks at. Every derived
feature has a plain-English meaning so it can appear in explanations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CATEGORICAL, NUMERIC_RAW

# Fixed category lists so training and live scoring encode categories the same way.
CATEGORY_LEVELS = {
    "Customer_Segment": ["NTC", "NTB", "Existing-to-Credit"],
    "Business_Type": ["Proprietorship", "Partnership", "LLP", "Private Limited"],
    "Industry": [
        "Retail", "Services", "Manufacturing", "Trading", "Food & Beverage",
        "Logistics", "Construction", "Textiles", "IT & Digital", "Healthcare",
    ],
    "Location_Category": ["Metro", "Tier 1", "Tier 2", "Tier 3", "Rural"],
    "GST_Return_Frequency": ["Monthly", "Quarterly"],
    "Has_Existing_Loan": ["No", "Yes"],
}


def _safe_div(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    out = np.full_like(a, np.nan)
    np.divide(a, b, out=out, where=b != 0)
    return out


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Add analyst-style ratios. Works on one row or on 50k rows."""
    d = df.copy()
    credits = d["Monthly_Bank_Credits_INR"]
    debits = d["Monthly_Bank_Debits_INR"]
    gst_sales = d["Monthly_GST_Sales_INR"]
    emi = d["Monthly_Loan_EMI_INR"].fillna(0)
    payroll = d["Monthly_Payroll_INR"]
    seas = d["Seasonality_Index"].clip(0, 1)

    # Do the data sources agree with each other?
    d["GST_Bank_Gap"] = _safe_div(credits - gst_sales, gst_sales)          # bank credits vs GST sales
    d["GST_Turnover_Gap"] = _safe_div(gst_sales * 12, d["Annual_Turnover_INR"]) - 1
    d["UPI_Share_of_Credits"] = _safe_div(d["Monthly_UPI_Inflow_INR"], credits)

    # Cash flow
    d["Net_Surplus_Ratio"] = _safe_div(credits - debits, credits)          # profit-like margin on bank flows
    d["Balance_Cover_Months"] = _safe_div(d["Average_Bank_Balance_INR"], debits)
    d["EMI_Burden"] = _safe_div(emi, credits)                              # FOIR-style debt load
    d["Gross_Margin_GST"] = _safe_div(gst_sales - d["Monthly_GST_Purchases_INR"], gst_sales)
    d["Payroll_to_Sales"] = _safe_div(payroll, gst_sales)
    d["Revenue_per_Employee"] = _safe_div(d["Annual_Turnover_INR"], d["Employee_Count"])

    # Seasonality stress test: in the lean month, sales-linked flows shrink by
    # the seasonality index while fixed costs (payroll, EMI) stay the same.
    variable_costs = (debits - payroll - emi).clip(lower=0)
    lean_surplus = (credits - variable_costs) * (1 - seas) - payroll - emi
    d["Lean_Month_Surplus_INR"] = lean_surplus
    d["Lean_Month_Surplus_Ratio"] = _safe_div(lean_surplus, credits)
    d["Lean_Month_Revenue_INR"] = gst_sales * (1 - seas)
    d["Peak_Month_Revenue_INR"] = gst_sales * (1 + seas)

    d["Thin_File"] = (d["Credit_History_Months"].fillna(0) == 0).astype(int)
    return d


DERIVED = [
    "GST_Bank_Gap", "GST_Turnover_Gap", "UPI_Share_of_Credits", "Net_Surplus_Ratio",
    "Balance_Cover_Months", "EMI_Burden", "Gross_Margin_GST", "Payroll_to_Sales",
    "Revenue_per_Employee", "Lean_Month_Surplus_Ratio", "Thin_File",
]

MODEL_FEATURES = CATEGORICAL + NUMERIC_RAW + DERIVED

# +1: higher value can only raise default risk. -1: can only lower it.
# Keeps the model's behaviour in line with credit common sense.
PD_MONOTONE = {
    "Transaction_Volatility_Index": 1,
    "Overdraft_Usage_Ratio": 1,
    "Employee_Attrition_Rate_Pct": 1,
    "Avg_Invoice_Payment_Delay_Days": 1,
    "Customer_Concentration_Ratio": 1,
    "EMI_Burden": 1,
    "GST_Filing_Timeliness_Pct": -1,
    "Cashflow_Stability_Index": -1,
    "EMI_On_Time_Rate_Pct": -1,
    "Salary_Consistency_Pct": -1,
    "Vendor_Payment_Timeliness_Pct": -1,
    "Years_in_Operation": -1,
    "Balance_Cover_Months": -1,
    "Lean_Month_Surplus_Ratio": -1,
}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Raw rows -> model-ready frame (derived features + fixed categories)."""
    d = add_derived(df)
    for col in CATEGORICAL:
        d[col] = pd.Categorical(d[col], categories=CATEGORY_LEVELS[col])
    for col in NUMERIC_RAW + DERIVED:
        d[col] = pd.to_numeric(d[col], errors="coerce").astype(float)
    return d[MODEL_FEATURES]


def monotone_vector(features: list[str], directions: dict[str, int]) -> list[int]:
    return [directions.get(f, 0) for f in features]


# Plain-English names used in explanations and the dashboard.
LABELS = {
    "Customer_Segment": "Customer segment",
    "Business_Type": "Business type",
    "Industry": "Industry",
    "Location_Category": "Location",
    "GST_Return_Frequency": "GST return frequency",
    "Has_Existing_Loan": "Has an existing loan",
    "Years_in_Operation": "Years in operation",
    "Employee_Count": "Employees",
    "Annual_Turnover_INR": "Annual turnover",
    "Monthly_GST_Sales_INR": "Monthly GST sales",
    "Monthly_GST_Purchases_INR": "Monthly GST purchases",
    "GST_Filing_Timeliness_Pct": "GST filed on time",
    "Monthly_UPI_Inflow_INR": "Monthly UPI inflow",
    "Monthly_UPI_Outflow_INR": "Monthly UPI outflow",
    "UPI_Avg_Ticket_Size_INR": "UPI average ticket size",
    "UPI_Daily_Txn_Count": "UPI transactions per day",
    "Transaction_Volatility_Index": "Transaction volatility",
    "Monthly_Bank_Credits_INR": "Monthly bank credits",
    "Monthly_Bank_Debits_INR": "Monthly bank debits",
    "Average_Bank_Balance_INR": "Average bank balance",
    "Cashflow_Stability_Index": "Cash-flow stability",
    "Monthly_Loan_EMI_INR": "Existing monthly EMI",
    "EMI_On_Time_Rate_Pct": "EMIs paid on time",
    "Overdraft_Usage_Ratio": "Overdraft usage",
    "Monthly_Payroll_INR": "Monthly payroll",
    "Salary_Consistency_Pct": "Salary paid consistently",
    "Employee_Attrition_Rate_Pct": "Employee attrition",
    "Avg_Invoice_Payment_Delay_Days": "Days customers take to pay",
    "Customer_Concentration_Ratio": "Dependence on top customers",
    "Vendor_Payment_Timeliness_Pct": "Vendors paid on time",
    "Seasonality_Index": "Seasonality of sales",
    "Revenue_Growth_Rate_Pct": "Revenue growth",
    "Credit_History_Months": "Credit history length",
    "GST_Bank_Gap": "Bank credits vs GST sales gap",
    "GST_Turnover_Gap": "GST sales vs declared turnover gap",
    "UPI_Share_of_Credits": "Share of income via UPI",
    "Net_Surplus_Ratio": "Monthly cash surplus margin",
    "Balance_Cover_Months": "Months of expenses covered by balance",
    "EMI_Burden": "EMI as share of income",
    "Gross_Margin_GST": "Gross margin (GST)",
    "Payroll_to_Sales": "Payroll as share of sales",
    "Revenue_per_Employee": "Revenue per employee",
    "Lean_Month_Surplus_Ratio": "Lean-month cash surplus",
    "Thin_File": "No credit history",
}

# Which pillar each input mainly belongs to (for grouping explanations).
FEATURE_PILLAR = {
    "Years_in_Operation": "Business Stability", "Employee_Count": "Business Stability",
    "Employee_Attrition_Rate_Pct": "Business Stability", "Customer_Concentration_Ratio": "Business Stability",
    "Business_Type": "Business Stability", "Revenue_per_Employee": "Business Stability",
    "Cashflow_Stability_Index": "Cash Flow", "Average_Bank_Balance_INR": "Cash Flow",
    "Monthly_Bank_Credits_INR": "Cash Flow", "Monthly_Bank_Debits_INR": "Cash Flow",
    "Overdraft_Usage_Ratio": "Cash Flow", "Net_Surplus_Ratio": "Cash Flow",
    "Balance_Cover_Months": "Cash Flow", "EMI_Burden": "Cash Flow", "Lean_Month_Surplus_Ratio": "Cash Flow",
    "Monthly_Loan_EMI_INR": "Cash Flow",
    "Transaction_Volatility_Index": "Revenue Consistency", "Seasonality_Index": "Revenue Consistency",
    "Monthly_GST_Sales_INR": "Revenue Consistency", "Monthly_UPI_Inflow_INR": "Revenue Consistency",
    "UPI_Daily_Txn_Count": "Revenue Consistency", "UPI_Avg_Ticket_Size_INR": "Revenue Consistency",
    "Monthly_UPI_Outflow_INR": "Revenue Consistency", "UPI_Share_of_Credits": "Revenue Consistency",
    "Annual_Turnover_INR": "Revenue Consistency", "Gross_Margin_GST": "Revenue Consistency",
    "Monthly_GST_Purchases_INR": "Revenue Consistency",
    "EMI_On_Time_Rate_Pct": "Payment Behaviour", "Vendor_Payment_Timeliness_Pct": "Payment Behaviour",
    "Avg_Invoice_Payment_Delay_Days": "Payment Behaviour", "Credit_History_Months": "Payment Behaviour",
    "Has_Existing_Loan": "Payment Behaviour", "Thin_File": "Payment Behaviour", "Customer_Segment": "Payment Behaviour",
    "Revenue_Growth_Rate_Pct": "Business Growth", "Industry": "Business Growth", "Location_Category": "Business Growth",
    "GST_Filing_Timeliness_Pct": "Compliance", "GST_Return_Frequency": "Compliance",
    "Salary_Consistency_Pct": "Compliance", "Monthly_Payroll_INR": "Compliance", "Payroll_to_Sales": "Compliance",
    "GST_Bank_Gap": "Compliance", "GST_Turnover_Gap": "Compliance",
}
