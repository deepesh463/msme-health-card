"""Central configuration: paths, column groups, pillars and credit policy.

Everything that a risk team would want to tune lives here, so policy changes
do not need code changes.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "msme_synthetic_50k.csv"
ARTIFACTS = ROOT / "artifacts"

ID_COL = "MSME_ID"

CATEGORICAL = [
    "Customer_Segment",
    "Business_Type",
    "Industry",
    "Location_Category",
    "GST_Return_Frequency",
    "Has_Existing_Loan",
]

NUMERIC_RAW = [
    "Years_in_Operation",
    "Employee_Count",
    "Annual_Turnover_INR",
    "Monthly_GST_Sales_INR",
    "Monthly_GST_Purchases_INR",
    "GST_Filing_Timeliness_Pct",
    "Monthly_UPI_Inflow_INR",
    "Monthly_UPI_Outflow_INR",
    "UPI_Avg_Ticket_Size_INR",
    "UPI_Daily_Txn_Count",
    "Transaction_Volatility_Index",
    "Monthly_Bank_Credits_INR",
    "Monthly_Bank_Debits_INR",
    "Average_Bank_Balance_INR",
    "Cashflow_Stability_Index",
    "Monthly_Loan_EMI_INR",
    "EMI_On_Time_Rate_Pct",  # blank when the business has no loan
    "Overdraft_Usage_Ratio",
    "Monthly_Payroll_INR",
    "Salary_Consistency_Pct",
    "Employee_Attrition_Rate_Pct",
    "Avg_Invoice_Payment_Delay_Days",
    "Customer_Concentration_Ratio",
    "Vendor_Payment_Timeliness_Pct",
    "Seasonality_Index",
    "Revenue_Growth_Rate_Pct",
    "Credit_History_Months",
]

# Columns the company supplied as answers. Never used as model inputs.
PILLARS = {
    "Business_Stability_Score": "Business Stability",
    "Cashflow_Score": "Cash Flow",
    "Revenue_Consistency_Score": "Revenue Consistency",
    "Payment_Behaviour_Score": "Payment Behaviour",
    "Business_Growth_Score": "Business Growth",
    "Compliance_Score": "Compliance",
}
TARGET_SCORE = "Financial_Health_Score"
TARGET_PD = "Probability_of_Default"
DECISION_COLS = ["Credit_Risk_Category", "Credit_Eligible", "Recommended_Credit_Limit_INR"]
ALL_TARGETS = list(PILLARS) + [TARGET_SCORE, TARGET_PD] + DECISION_COLS

# Risk bands found in the data: PD < 6% Low, 6-15% Medium, > 15% High.
RISK_BANDS = [(0.06, "Low"), (0.15, "Medium"), (1.01, "High")]

# Credit policy. Limit-to-turnover ratios are the medians found in the data
# (Low ~16%, Medium ~8%, High not eligible). The affordability cap and pricing
# are our addition on top of the company's policy.
POLICY = {
    "limit_pct_of_turnover": {"Low": 0.16, "Medium": 0.08, "High": 0.0},
    "max_new_emi_share_of_surplus": 0.50,  # at most half the average monthly surplus goes to the new EMI
    "tenure_months": {"Low": 36, "Medium": 24, "High": 0},
    "annual_rate_pct": {"Low": 14.0, "Medium": 18.0, "High": None},
    "round_to": 1000,
}

# Which business data source each input comes from (used for the card and docs).
SOURCE_OF = {
    "Monthly_GST_Sales_INR": "GSTN",
    "Monthly_GST_Purchases_INR": "GSTN",
    "GST_Filing_Timeliness_Pct": "GSTN",
    "GST_Return_Frequency": "GSTN",
    "Annual_Turnover_INR": "GSTN",
    "Revenue_Growth_Rate_Pct": "GSTN",
    "Seasonality_Index": "GSTN",
    "Years_in_Operation": "GST registration / Udyam",
    "GST_Bank_Gap": "Cross-source check (GSTN vs AA)",
    "GST_Turnover_Gap": "Cross-source check (GSTN)",
    "UPI_Share_of_Credits": "Cross-source check (UPI vs AA)",
    "Net_Surplus_Ratio": "Account Aggregator",
    "Balance_Cover_Months": "Account Aggregator",
    "EMI_Burden": "Account Aggregator",
    "Lean_Month_Surplus_Ratio": "AA + GSTN (stress test)",
    "Gross_Margin_GST": "GSTN",
    "Payroll_to_Sales": "EPFO + GSTN",
    "Monthly_UPI_Inflow_INR": "UPI",
    "Monthly_UPI_Outflow_INR": "UPI",
    "UPI_Avg_Ticket_Size_INR": "UPI",
    "UPI_Daily_Txn_Count": "UPI",
    "Monthly_Bank_Credits_INR": "Account Aggregator",
    "Monthly_Bank_Debits_INR": "Account Aggregator",
    "Average_Bank_Balance_INR": "Account Aggregator",
    "Cashflow_Stability_Index": "Account Aggregator",
    "Overdraft_Usage_Ratio": "Account Aggregator",
    "Transaction_Volatility_Index": "Account Aggregator",
    "Monthly_Loan_EMI_INR": "Credit Bureau / AA",
    "EMI_On_Time_Rate_Pct": "Credit Bureau / AA",
    "Credit_History_Months": "Credit Bureau",
    "Monthly_Payroll_INR": "EPFO",
    "Salary_Consistency_Pct": "EPFO",
    "Employee_Attrition_Rate_Pct": "EPFO",
    "Employee_Count": "EPFO",
    "Avg_Invoice_Payment_Delay_Days": "Invoices (TReDS / e-invoice)",
    "Customer_Concentration_Ratio": "Invoices (TReDS / e-invoice)",
    "Vendor_Payment_Timeliness_Pct": "Invoices (TReDS / e-invoice)",
}

RANDOM_STATE = 42
