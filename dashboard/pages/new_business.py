import streamlit as st

import card_view
from ui import engine, raw_data, setup

setup("➕ Score a new business",
      "Enter a business's consented data (as it would arrive from GSTN, AA, UPI and EPFO) and get its Health Card instantly.")

ref = raw_data()
med = ref.median(numeric_only=True)

with st.form("new"):
    st.markdown("##### Business profile")
    c = st.columns(5)
    seg = c[0].selectbox("Segment", ["NTC", "NTB", "Existing-to-Credit"])
    btype = c[1].selectbox("Business type", ["Proprietorship", "Partnership", "LLP", "Private Limited"])
    ind = c[2].selectbox("Industry", sorted(ref.Industry.unique()), index=sorted(ref.Industry.unique()).index("Logistics"))
    loc = c[3].selectbox("Location", ["Metro", "Tier 1", "Tier 2", "Tier 3", "Rural"], index=2)
    years = c[4].number_input("Years in operation", 0.0, 60.0, float(med.Years_in_Operation), 0.5)

    st.markdown("##### GSTN")
    c = st.columns(5)
    gst_sales = c[0].number_input("Monthly GST sales ₹", 0.0, value=float(med.Monthly_GST_Sales_INR), step=10000.0)
    gst_purch = c[1].number_input("Monthly GST purchases ₹", 0.0, value=float(med.Monthly_GST_Purchases_INR), step=10000.0)
    gst_time = c[2].slider("GST filed on time %", 50.0, 100.0, float(med.GST_Filing_Timeliness_Pct))
    gst_freq = c[3].selectbox("Return frequency", ["Monthly", "Quarterly"])
    growth = c[4].number_input("Revenue growth % (YoY)", -60.0, 100.0, float(med.Revenue_Growth_Rate_Pct))

    st.markdown("##### Bank (Account Aggregator) & UPI")
    c = st.columns(5)
    credits = c[0].number_input("Monthly bank credits ₹", 0.0, value=float(med.Monthly_Bank_Credits_INR), step=10000.0)
    debits = c[1].number_input("Monthly bank debits ₹", 0.0, value=float(med.Monthly_Bank_Debits_INR), step=10000.0)
    bal = c[2].number_input("Average balance ₹", 0.0, value=float(med.Average_Bank_Balance_INR), step=10000.0)
    cfs = c[3].slider("Cash-flow stability", 0.25, 1.0, float(med.Cashflow_Stability_Index))
    od = c[4].slider("Overdraft usage", 0.0, 0.6, float(med.Overdraft_Usage_Ratio))
    c = st.columns(5)
    upi_in = c[0].number_input("Monthly UPI inflow ₹", 0.0, value=float(med.Monthly_UPI_Inflow_INR), step=5000.0)
    upi_out = c[1].number_input("Monthly UPI outflow ₹", 0.0, value=float(med.Monthly_UPI_Outflow_INR), step=5000.0)
    ticket = c[2].number_input("UPI avg ticket ₹", 0.0, value=float(med.UPI_Avg_Ticket_Size_INR))
    txn = c[3].number_input("UPI txns / day", 0.0, value=float(med.UPI_Daily_Txn_Count))
    vol = c[4].slider("Transaction volatility", 0.0, 0.8, float(med.Transaction_Volatility_Index))

    st.markdown("##### Credit, payroll (EPFO) and invoices")
    c = st.columns(5)
    has_loan = c[0].selectbox("Existing loan?", ["No", "Yes"])
    emi = c[1].number_input("Existing EMI ₹", 0.0, value=0.0, step=1000.0)
    emi_ot = c[2].slider("EMIs on time % (if loan)", 60.0, 100.0, 92.0)
    hist = c[3].number_input("Credit history (months)", 0, 240, 0)
    emp = c[4].number_input("Employees", 1, 1000, int(med.Employee_Count))
    c = st.columns(5)
    payroll = c[0].number_input("Monthly payroll ₹", 0.0, value=float(med.Monthly_Payroll_INR), step=5000.0)
    sal = c[1].slider("Salary consistency %", 50.0, 100.0, float(med.Salary_Consistency_Pct))
    attr = c[2].slider("Attrition %", 0.0, 50.0, float(med.Employee_Attrition_Rate_Pct))
    delay = c[3].number_input("Invoice payment delay (days)", 0.0, 120.0, float(med.Avg_Invoice_Payment_Delay_Days))
    conc = c[4].slider("Customer concentration", 0.05, 0.95, float(med.Customer_Concentration_Ratio))
    c = st.columns(5)
    vendor = c[0].slider("Vendors paid on time %", 50.0, 100.0, float(med.Vendor_Payment_Timeliness_Pct))
    seas = c[1].slider("Seasonality index", 0.0, 1.0, float(med.Seasonality_Index))
    submitted = st.form_submit_button("Generate Health Card", type="primary", use_container_width=True)

if submitted:
    row = {
        "MSME_ID": "NEW-APPLICANT", "Customer_Segment": seg, "Business_Type": btype, "Industry": ind, "Location_Category": loc,
        "Years_in_Operation": years, "Employee_Count": emp, "Annual_Turnover_INR": gst_sales * 12,
        "Monthly_GST_Sales_INR": gst_sales, "Monthly_GST_Purchases_INR": gst_purch, "GST_Filing_Timeliness_Pct": gst_time,
        "GST_Return_Frequency": gst_freq, "Monthly_UPI_Inflow_INR": upi_in, "Monthly_UPI_Outflow_INR": upi_out,
        "UPI_Avg_Ticket_Size_INR": ticket, "UPI_Daily_Txn_Count": txn, "Transaction_Volatility_Index": vol,
        "Monthly_Bank_Credits_INR": credits, "Monthly_Bank_Debits_INR": debits, "Average_Bank_Balance_INR": bal,
        "Cashflow_Stability_Index": cfs, "Has_Existing_Loan": has_loan, "Monthly_Loan_EMI_INR": emi if has_loan == "Yes" else 0,
        "EMI_On_Time_Rate_Pct": emi_ot if has_loan == "Yes" else None, "Overdraft_Usage_Ratio": od,
        "Monthly_Payroll_INR": payroll, "Salary_Consistency_Pct": sal, "Employee_Attrition_Rate_Pct": attr,
        "Avg_Invoice_Payment_Delay_Days": delay, "Customer_Concentration_Ratio": conc, "Vendor_Payment_Timeliness_Pct": vendor,
        "Seasonality_Index": seas, "Revenue_Growth_Rate_Pct": growth, "Credit_History_Months": hist if seg != "NTC" else 0,
    }
    with st.spinner("Scoring…"):
        card = engine().health_card(row)
    st.divider()
    card_view.render(card, row)
    st.caption("The same scoring is available as an API: `POST /api/v1/score`.")
