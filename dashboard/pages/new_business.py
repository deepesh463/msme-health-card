import math

import streamlit as st

import card_view
from ui import chart_card, engine, raw_data, setup

setup("➕ New applicant assessment",
      "Enter a business's consented data (as it would arrive from GSTN, AA, UPI and EPFO) and get its Health Card instantly.")

ref = raw_data()


def _example(mask, target_score):
    """A real business from the data: the one whose health score is closest to the target."""
    pool = ref[mask]
    return pool.iloc[(pool.score_overall - target_score).abs().argsort().iloc[0]]


PRESETS = {
    "Typical business": None,  # median of every input
    "Strong business": lambda: _example((ref.decision == "Approve") & (ref.Customer_Segment == "NTC"), 85),
    "Risky business": lambda: _example(ref.decision == "Decline", 62),
}

top = st.columns([1.3, 2])
with top[0]:
    preset = st.selectbox("Start from", list(PRESETS), help="Pre-fills the form. Change any value before scoring.")
with top[1]:
    st.markdown(
        '<div style="display:flex;gap:8px;align-items:center;height:100%;padding-top:26px;font-size:.8rem;color:#475569">'
        '<span style="background:#EEF2FF;color:#4338CA;font-weight:700;border-radius:999px;padding:3px 10px">1 · Fill in the data</span>→'
        '<span style="background:#EEF2FF;color:#4338CA;font-weight:700;border-radius:999px;padding:3px 10px">2 · Generate</span>→'
        '<span style="background:#EEF2FF;color:#4338CA;font-weight:700;border-radius:999px;padding:3px 10px">3 · Health Card + decision</span></div>',
        unsafe_allow_html=True,
    )

base = ref.median(numeric_only=True) if PRESETS[preset] is None else PRESETS[preset]()
k = preset.split()[0].lower()  # widget keys per preset, so switching preset refills the form


def v(col, default=0.0):
    x = base.get(col, default) if hasattr(base, "get") else default
    try:
        return default if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)
    except (TypeError, ValueError):
        return default


def clip(x, lo, hi):
    return type(lo)(min(max(x, lo), hi))


def cat(col, options, default):
    x = base.get(col) if PRESETS[preset] is not None else None
    return options.index(x) if x in options else options.index(default)


def money(col_box, label, col, step=10000):
    return col_box.number_input(label, 0, value=int(round(v(col))), step=step, format="%d", key=f"{k}_{col}")


def src(*names):
    return " ".join(f'<span style="font-size:.66rem;font-weight:600;background:#EEF2FF;color:#4338CA;padding:2px 8px;'
                    f'border-radius:999px;margin-left:4px">{n}</span>' for n in names)


industries = sorted(ref.Industry.unique())
with st.form("new", border=False):
    with chart_card(f"🏢 Business profile {src('Udyam registration')}", "Who the business is"):
        c = st.columns(5)
        segs = ["NTC", "NTB", "Existing-to-Credit"]
        seg = c[0].selectbox("Segment", segs, index=cat("Customer_Segment", segs, "NTC"), key=f"{k}_seg",
                             help="NTC = never had a loan · NTB = new to this bank")
        types = ["Proprietorship", "Partnership", "LLP", "Private Limited"]
        btype = c[1].selectbox("Business type", types, index=cat("Business_Type", types, "Proprietorship"), key=f"{k}_bt")
        ind = c[2].selectbox("Industry", industries, index=cat("Industry", industries, "Logistics"), key=f"{k}_ind")
        locs = ["Metro", "Tier 1", "Tier 2", "Tier 3", "Rural"]
        loc = c[3].selectbox("Location", locs, index=cat("Location_Category", locs, "Tier 2"), key=f"{k}_loc")
        years = c[4].number_input("Years in operation", 0.0, 60.0, clip(round(v("Years_in_Operation"), 1), 0.0, 60.0), 0.5, key=f"{k}_yrs")

    a, b = st.columns(2)
    with a:
        with chart_card(f"🧾 GST returns {src('GSTN')}", "Sales, purchases and filing discipline"):
            c = st.columns(2)
            gst_sales = money(c[0], "Monthly GST sales (₹)", "Monthly_GST_Sales_INR")
            gst_purch = money(c[1], "Monthly GST purchases (₹)", "Monthly_GST_Purchases_INR")
            c = st.columns(3)
            gst_time = c[0].slider("Filed on time (%)", 50.0, 100.0, clip(v("GST_Filing_Timeliness_Pct", 85.0), 50.0, 100.0), key=f"{k}_gstt")
            freqs = ["Monthly", "Quarterly"]
            gst_freq = c[1].selectbox("Return frequency", freqs, index=cat("GST_Return_Frequency", freqs, "Monthly"), key=f"{k}_gstf")
            growth = c[2].number_input("Revenue growth (% YoY)", -60.0, 100.0, clip(round(v("Revenue_Growth_Rate_Pct"), 1), -60.0, 100.0), key=f"{k}_gr")
    with b:
        with chart_card(f"👥 Payroll {src('EPFO')}", "Team size and how reliably salaries are paid"):
            c = st.columns(2)
            emp = c[0].number_input("Employees", 1, 1000, clip(int(v("Employee_Count", 1)), 1, 1000), key=f"{k}_emp")
            payroll = money(c[1], "Monthly payroll (₹)", "Monthly_Payroll_INR", 5000)
            c = st.columns(2)
            sal = c[0].slider("Salaries paid on time (%)", 50.0, 100.0, clip(v("Salary_Consistency_Pct", 90.0), 50.0, 100.0), key=f"{k}_sal")
            attr = c[1].slider("Staff leaving per year (%)", 0.0, 50.0, clip(v("Employee_Attrition_Rate_Pct", 15.0), 0.0, 50.0), key=f"{k}_attr")

    with chart_card(f"🏦 Bank account & UPI {src('Account Aggregator', 'UPI')}", "Money in and out, balance and stability"):
        c = st.columns(5)
        credits = money(c[0], "Monthly bank credits (₹)", "Monthly_Bank_Credits_INR")
        debits = money(c[1], "Monthly bank debits (₹)", "Monthly_Bank_Debits_INR")
        bal = money(c[2], "Average balance (₹)", "Average_Bank_Balance_INR")
        cfs = c[3].slider("Cash-flow stability (0–1)", 0.25, 1.0, clip(v("Cashflow_Stability_Index", 0.7), 0.25, 1.0), key=f"{k}_cfs")
        od = c[4].slider("Overdraft used (share of limit)", 0.0, 0.6, clip(v("Overdraft_Usage_Ratio", 0.2), 0.0, 0.6), key=f"{k}_od")
        c = st.columns(5)
        upi_in = money(c[0], "Monthly UPI inflow (₹)", "Monthly_UPI_Inflow_INR", 5000)
        upi_out = money(c[1], "Monthly UPI outflow (₹)", "Monthly_UPI_Outflow_INR", 5000)
        ticket = money(c[2], "UPI average ticket (₹)", "UPI_Avg_Ticket_Size_INR", 50)
        txn = c[3].number_input("UPI payments per day", 0.0, value=round(v("UPI_Daily_Txn_Count"), 1), key=f"{k}_txn")
        vol = c[4].slider("Transaction volatility (0–1)", 0.0, 0.8, clip(v("Transaction_Volatility_Index", 0.3), 0.0, 0.8), key=f"{k}_vol")

    a, b = st.columns(2)
    with a:
        with chart_card(f"💳 Existing loans {src('Credit bureau')}", "Leave as “No” for a new-to-credit business"):
            c = st.columns(2)
            yn = ["No", "Yes"]
            has_loan = c[0].selectbox("Existing loan?", yn, index=cat("Has_Existing_Loan", yn, "No"), key=f"{k}_loan")
            emi = money(c[1], "Existing EMI (₹ / month)", "Monthly_Loan_EMI_INR", 1000)
            c = st.columns(2)
            emi_ot = c[0].slider("EMIs paid on time (%)", 60.0, 100.0, clip(v("EMI_On_Time_Rate_Pct", 92.0), 60.0, 100.0), key=f"{k}_emiot")
            hist = c[1].number_input("Credit history (months)", 0, 240, clip(int(v("Credit_History_Months", 0)), 0, 240), key=f"{k}_hist")
    with b:
        with chart_card(f"📄 Customers & suppliers {src('GST e-invoices')}", "How fast customers pay and how the business pays others"):
            c = st.columns(2)
            delay = c[0].number_input("Days customers take to pay", 0.0, 120.0, clip(round(v("Avg_Invoice_Payment_Delay_Days", 15.0), 1), 0.0, 120.0), key=f"{k}_delay")
            conc = c[1].slider("Sales from top customers (share)", 0.05, 0.95, clip(v("Customer_Concentration_Ratio", 0.4), 0.05, 0.95), key=f"{k}_conc")
            c = st.columns(2)
            vendor = c[0].slider("Suppliers paid on time (%)", 50.0, 100.0, clip(v("Vendor_Payment_Timeliness_Pct", 85.0), 50.0, 100.0), key=f"{k}_vend")
            seas = c[1].slider("Seasonality (0 = steady, 1 = very seasonal)", 0.0, 1.0, clip(v("Seasonality_Index", 0.3), 0.0, 1.0), key=f"{k}_seas")

    st.markdown("")
    submitted = st.form_submit_button("⚡  Generate Health Card", type="primary", use_container_width=True)
    st.caption("Uses the same models as the API endpoint `POST /api/v1/score`. Takes under a second.")

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
