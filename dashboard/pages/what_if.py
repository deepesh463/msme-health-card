import plotly.graph_objects as go
import streamlit as st

from ui import BAND_COLOR, GREEN, INDIGO, RED, badge, engine, pick_business, raw_data, setup, style_fig

setup("🧪 What-if simulator",
      "Move the levers a business owner controls and watch the score and default risk respond in real time.")

biz = pick_business("wi")
base = raw_data().loc[biz].to_dict()
base["MSME_ID"] = biz
eng = engine()
before = eng.quick_score(base)

# (column, label, min, max, step)
LEVERS = [
    ("GST_Filing_Timeliness_Pct", "GST returns filed on time (%)", 50.0, 100.0, 0.5),
    ("Overdraft_Usage_Ratio", "Overdraft usage (share of limit)", 0.0, 0.6, 0.01),
    ("Avg_Invoice_Payment_Delay_Days", "Days customers take to pay", 0.0, 90.0, 1.0),
    ("Vendor_Payment_Timeliness_Pct", "Vendors paid on time (%)", 50.0, 100.0, 0.5),
    ("Salary_Consistency_Pct", "Salaries paid on schedule (%)", 50.0, 100.0, 0.5),
    ("Customer_Concentration_Ratio", "Share of sales from top customers", 0.05, 0.95, 0.01),
    ("Cashflow_Stability_Index", "Cash-flow stability index", 0.25, 1.0, 0.01),
    ("Seasonality_Index", "Seasonality of sales", 0.0, 1.0, 0.01),
]
if base["Has_Existing_Loan"] == "Yes" and base["EMI_On_Time_Rate_Pct"] == base["EMI_On_Time_Rate_Pct"]:
    LEVERS.append(("EMI_On_Time_Rate_Pct", "EMIs paid on time (%)", 60.0, 100.0, 0.5))


def key(col):
    return f"wi_{biz}_{col}"


st.markdown(f"#### {biz} · {base['Industry']} · {base['Location_Category']} · {base['Customer_Segment']}")
left, right = st.columns([1, 1.2])
with left:
    h1, h2 = st.columns([2, 1])
    h1.markdown("**Adjust the levers**")
    if h2.button("↺ Reset", use_container_width=True):
        for col, *_ in LEVERS:
            st.session_state.pop(key(col), None)
        st.rerun()
    new = dict(base)
    for col, label, lo, hi, step in LEVERS:
        start = float(min(max(base[col], lo), hi))
        new[col] = st.slider(label, lo, hi, start, step, key=key(col))

after = eng.quick_score(new)
with right:
    a, b = st.columns(2)
    d_score = after["score"] - before["score"]
    d_pd = (after["pd"] - before["pd"]) * 100
    with a:
        st.metric("Financial Health Score", f"{after['score']:.1f}",
                  f"{d_score:+.1f} pts" if abs(d_score) >= 0.05 else None)
        st.markdown(badge(f"{before['risk_band']} → {after['risk_band']} risk", BAND_COLOR[after["risk_band"]]), unsafe_allow_html=True)
    with b:
        st.metric("Probability of default", f"{after['pd']:.1%}",
                  f"{d_pd:+.2f} pp" if abs(d_pd) >= 0.005 else None, delta_color="inverse")
    names = list(before["pillars"])
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=[before["pillars"][n] for n in names] + [before["pillars"][names[0]]], theta=names + [names[0]],
                                  name="Before", line_color="#94A3B8", fill="toself", fillcolor="rgba(148,163,184,0.15)"))
    fig.add_trace(go.Scatterpolar(r=[after["pillars"][n] for n in names] + [after["pillars"][names[0]]], theta=names + [names[0]],
                                  name="After", line_color=INDIGO, fill="toself", fillcolor="rgba(79,70,229,0.18)"))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100], showticklabels=False), bgcolor="rgba(0,0,0,0)",
                                 domain=dict(x=[0.2, 0.8])), title="Pillar scores: before vs after")
    fig = style_fig(fig, 400)
    fig.update_layout(margin=dict(l=40, r=40, t=80, b=30))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    deltas = {n: after["pillars"][n] - before["pillars"][n] for n in names}
    fig = go.Figure(go.Bar(x=list(deltas.values()), y=list(deltas), orientation="h", cliponaxis=False,
                           marker_color=[GREEN if v >= 0 else RED for v in deltas.values()],
                           text=[f"{v:+.1f}" for v in deltas.values()], textposition="outside"))
    span = max(1.0, max(abs(v) for v in deltas.values()) * 1.4)
    fig.update_xaxes(range=[-span, span])
    fig.update_layout(title="Change per pillar (points)")
    st.plotly_chart(style_fig(fig, 260, legend=False), use_container_width=True, config={"displayModeBar": False})
st.caption("The model has monotonic constraints, so improving a good behaviour can never lower the score. "
           "That makes this simulator safe to show to a business owner.")
