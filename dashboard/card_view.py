"""Renders a full Health Card. Used by the Health Card page and the 'Score a new business' page."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from msme_health.features import LABELS

from ui import (AMBER, BAND_COLOR, DECISION_COLOR, GREEN, INDIGO, LEVEL_COLOR, PILLAR_GRADE_RANGE, RED, SLATE, badge,
                card, chips, gauge, grade_key, inr, pillar_tile, style_fig)

PCT_METRICS = {"Net_Surplus_Ratio", "Lean_Month_Surplus_Ratio", "Overdraft_Usage_Ratio"}


def _fmt(metric: str, v: float) -> str:
    if metric.endswith("_INR"):
        return inr(v)
    if metric.endswith("_Pct"):
        return f"{v:.1f}%"
    if metric in PCT_METRICS:
        return f"{v:.0%}"
    if metric.endswith("_Days"):
        return f"{v:.0f} d"
    return f"{v:,.2f}"


def header(card_d: dict, raw: dict):
    p = card_d["profile"]
    cov = card_d["data_coverage"]
    st.markdown(f"### {card_d['msme_id'] or 'New business'}")
    st.markdown(
        chips([p["Customer_Segment"], p["Industry"], p["Location_Category"], p["Business_Type"]])
        + chips([f"{p['Years_in_Operation']:.1f} yrs in business", f"{p['Employee_Count']} employees",
                 f"Turnover {inr(p['Annual_Turnover_INR'])}/yr",
                 f"Credit history {int(p['Credit_History_Months'] or 0)} mo"], grey=True),
        unsafe_allow_html=True,
    )
    src = " ".join(("✅ " if ok else "⛔ ") + name for name, ok in cov["sources"].items())
    note = f" — {cov['note']}" if cov["note"] else ""
    st.caption(f"Data sources: {src}  ·  Confidence: **{cov['confidence']}**{note}")


def top_row(c: dict):
    rec = c["credit_recommendation"]
    a, b, d = st.columns([1.15, 1, 1])
    with a:
        st.plotly_chart(gauge(c["financial_health_score"], c["grade"]), use_container_width=True, config={"displayModeBar": False})
        st.caption("Overall grade by score: A+ 85+ · A 78–85 · B 70–78 · C 62–70 · D below 62")
    with b:
        col = DECISION_COLOR.get(rec["decision"], INDIGO)
        st.markdown(
            f'<div class="card"><h4>Credit decision</h4>{badge(rec["decision"], col)}'
            f'<div class="kpi" style="margin-top:12px">{inr(rec["recommended_limit_inr"])}</div>'
            f'<div class="kpi-sub">Recommended limit'
            + (f' · {rec["tenure_months"]} mo @ {rec["indicative_rate_pct"]:.0f}% · EMI {inr(rec["estimated_emi_inr"])}' if rec["eligible"] else "")
            + "</div>"
            + (f'<div class="kpi-sub" style="margin-top:8px">🚗 <b>{rec["suggested_product"]}</b></div>' if rec["suggested_product"] else "")
            + (f'<div class="kpi-sub">Limit set by: {rec["binding_constraint"]}</div>' if rec["binding_constraint"] else "")
            + "</div>",
            unsafe_allow_html=True,
        )
    with d:
        band = c["risk_band"]
        st.markdown(
            f'<div class="card"><h4>Default risk</h4>{badge(band + " risk", BAND_COLOR[band])}'
            f'<div class="kpi" style="margin-top:12px">{c["probability_of_default"]:.1%}</div>'
            f'<div class="kpi-sub">Predicted probability of default (12 mo)</div>'
            f'<div class="kpi-sub" style="margin-top:8px">Bands: Low &lt; 6% · Medium 6–15% · High &gt; 15%</div></div>',
            unsafe_allow_html=True,
        )


def pillars_row(c: dict):
    st.markdown("##### The 6 pillars")
    st.markdown("**What the grade means:** how this business ranks against all 50,000 MSMEs on that pillar.")
    st.markdown(grade_key(PILLAR_GRADE_RANGE), unsafe_allow_html=True)
    st.caption("Why a ranking and not the raw score: each pillar has its own natural range. Most businesses score about 83 on "
               "Payment but about 41 on Growth, so 82 on Payment is below average (C) while 55 on Growth is in the top 15% (A).")
    cols = st.columns(6)
    for col, p in zip(cols, c["pillars"]):
        with col:
            pillar_tile(p)


def why_tab(c: dict):
    e = c["explanation"]
    drivers = e["top_drivers"]  # biggest effect first; chart reads top-to-bottom
    left, right = st.columns([1.4, 1])
    with left:
        fig = go.Figure(go.Waterfall(
            orientation="h",
            y=["Average MSME"] + [f"{d['label']}" for d in drivers] + ["All other factors", "This business"],
            x=[e["base_score"]] + [d["impact_points"] for d in drivers]
              + [c["financial_health_score"] - e["base_score"] - sum(d["impact_points"] for d in drivers), 0],
            measure=["absolute"] + ["relative"] * (len(drivers) + 1) + ["total"],
            increasing=dict(marker=dict(color=GREEN)), decreasing=dict(marker=dict(color=RED)),
            totals=dict(marker=dict(color=INDIGO)), connector=dict(line=dict(color="#CBD5E1")),
            text=[f"{e['base_score']:.1f}"] + [f"{d['impact_points']:+.1f}" for d in drivers] + ["", f"{c['financial_health_score']:.1f}"],
            textposition="outside",
        ))
        fig.update_layout(title="How each factor moved the score (SHAP, in score points)")
        lo = min(e["base_score"], c["financial_health_score"]) - 8
        fig.update_xaxes(range=[max(0, lo), min(100, max(e["base_score"], c["financial_health_score"]) + 8)])
        fig.update_yaxes(autorange="reversed")
        fig.update_traces(cliponaxis=False)
        st.plotly_chart(style_fig(fig, 420, legend=False), use_container_width=True, config={"displayModeBar": False})
        st.caption("Read top to bottom: start from the average MSME's score, add or subtract each factor's effect, "
                   "and you arrive at this business's score.")
    with right:
        pc = pd.Series(e["pillar_contribution_points"]).sort_values()
        pad = max(0.5, float(pc.abs().max()) * 0.35)
        fig = go.Figure(go.Bar(x=pc.values, y=pc.index, orientation="h",
                               marker_color=[GREEN if v > 0 else RED for v in pc.values],
                               text=[f"{v:+.1f}" for v in pc.values], textposition="outside", cliponaxis=False))
        fig.update_xaxes(range=[min(0, float(pc.min())) - pad, max(0, float(pc.max())) + pad])
        fig.update_layout(title="Net effect by pillar (points)")
        st.plotly_chart(style_fig(fig, 260, legend=False), use_container_width=True, config={"displayModeBar": False})
        st.markdown("**Top reasons raising default risk**")
        if e["risk_reason_codes"]:
            for i, r in enumerate(e["risk_reason_codes"], 1):
                val = "—" if r["value"] is None else (f"{r['value']:,.2f}" if isinstance(r["value"], float) else r["value"])
                st.markdown(f"{i}. **{r['label']}** (value: {val})")
        else:
            st.markdown("No factor is raising risk above the average MSME.")
        st.caption("Reason codes support adverse-action notices under RBI's Fair Practices Code.")


def risk_tab(c: dict):
    left, right = st.columns([1.2, 1])
    with left:
        rows = "".join(
            f'<div class="ind-row"><span><span class="dot" style="background:{LEVEL_COLOR[r["level"]]}"></span>{r["indicator"]}</span>'
            f'<span><b>{r["value"]}</b> &nbsp; <span class="small">{r["level"]}</span></span></div>'
            for r in c["risk_indicators"]
        )
        st.markdown(f'<div class="card"><h4>Risk indicators</h4>{rows}</div>', unsafe_allow_html=True)
    with right:
        a = c["anomaly"]
        col = RED if a["flags"] else (AMBER if a["is_outlier"] else GREEN)
        status = "Flags raised" if a["flags"] else ("Unusual profile" if a["is_outlier"] else "No anomalies")
        flags = "".join(f"<li>{f}</li>" for f in a["flags"]) or "<li>All cross-source checks passed (GST vs bank, UPI vs bank, payroll vs sales, EMI load).</li>"
        st.markdown(
            f'<div class="card"><h4>Anomaly detection</h4>{badge(status, col)}'
            f'<div class="kpi" style="margin-top:10px">{a["percentile"]:.0f}<span class="small"> /100</span></div>'
            f'<div class="kpi-sub">How unusual this profile is vs all MSMEs (Isolation Forest)</div>'
            f'<ul style="margin-top:10px;font-size:.88rem">{flags}</ul></div>',
            unsafe_allow_html=True,
        )
        rec = c["credit_recommendation"]
        if rec["conditions"] or rec["reasons"]:
            items = "".join(f"<li>{x}</li>" for x in rec["reasons"] + rec["conditions"])
            st.markdown(f'<div class="card" style="margin-top:12px"><h4>Decision notes</h4><ul style="font-size:.88rem">{items}</ul></div>',
                        unsafe_allow_html=True)


def peers_tab(c: dict):
    pb = c["peer_benchmark"]
    if not pb:
        st.info("Peer data not available.")
        return
    st.caption(f"Compared with **{pb['peer_count']:,} {pb['peer_group']}**. Bar = how this business ranks (100 = best among peers).")
    m = pd.DataFrame(pb["metrics"])
    m["label"] = m.metric.map(LABELS).fillna(m.metric)
    m = m.sort_values("standing")
    colors = [RED if s < 25 else (AMBER if s < 50 else (INDIGO if s < 75 else GREEN)) for s in m.standing]
    hover = [f"{lab}<br>This business: {_fmt(mt, v)}<br>Peer median: {_fmt(mt, med)}<br>Peer middle 50%: {_fmt(mt, p25)} – {_fmt(mt, p75)}"
             for lab, mt, v, med, p25, p75 in zip(m.label, m.metric, m.value, m.peer_median, m.peer_p25, m.peer_p75)]
    fig = go.Figure(go.Bar(
        x=m.standing, y=m.label, orientation="h", marker_color=colors,
        text=[f"{_fmt(mt, v)}  (peer median {_fmt(mt, med)})" for mt, v, med in zip(m.metric, m.value, m.peer_median)],
        textposition="auto", hovertext=hover, hoverinfo="text",
    ))
    fig.add_vline(x=50, line_dash="dot", line_color=SLATE, annotation_text="peer median", annotation_position="top")
    fig.update_xaxes(range=[0, 100], title="Peer standing (percentile, higher = better)")
    st.plotly_chart(style_fig(fig, 60 + 30 * len(m), legend=False), use_container_width=True, config={"displayModeBar": False})


def money_tab(c: dict, raw: dict):
    s = c["seasonality"]
    pb = {m["metric"]: m for m in (c["peer_benchmark"] or {}).get("metrics", [])}
    a, b = st.columns(2)
    with a:
        # Monthly revenue profile with seasonality (no monthly history in the data, so lean/avg/peak).
        peer_med = pb.get("Monthly_GST_Sales_INR", {}).get("peer_median")
        fig = go.Figure()
        fig.add_bar(x=["Lean month", "Average month", "Peak month"],
                    y=[s["lean_month_revenue_inr"], s["avg_month_revenue_inr"], s["peak_month_revenue_inr"]],
                    marker_color=["#A5B4FC", INDIGO, "#312E81"], text=[inr(v) for v in [s["lean_month_revenue_inr"], s["avg_month_revenue_inr"], s["peak_month_revenue_inr"]]],
                    textposition="outside", name="This business")
        if peer_med:
            fig.add_hline(y=peer_med, line_dash="dot", line_color=AMBER, annotation_text=f"Peer median {inr(peer_med)}",
                          annotation_position="top left", annotation_font_color=AMBER)
        top = max(s["peak_month_revenue_inr"], peer_med or 0) * 1.2
        fig.update_yaxes(range=[0, top], showticklabels=False)
        fig.update_layout(title=f"Monthly revenue: lean vs average vs peak (seasonality {s['seasonality_index']:.2f})", showlegend=False)
        st.plotly_chart(style_fig(fig, 330, legend=False), use_container_width=True, config={"displayModeBar": False})
    with b:
        emi = c["credit_recommendation"]["estimated_emi_inr"]
        fig = go.Figure()
        vals = [s["avg_month_surplus_inr"], s["lean_month_surplus_inr"]]
        fig.add_bar(x=["Average month", "Lean month"], y=vals, marker_color=[GREEN if v > 0 else RED for v in vals],
                    text=[inr(v) for v in vals], textposition="outside")
        if emi:
            fig.add_hline(y=emi, line_dash="dash", line_color=INDIGO, annotation_text=f"Proposed EMI {inr(emi)}",
                          annotation_position="top left", annotation_font_color=INDIGO)
        hi = max(max(vals), emi or 0) * 1.25
        lo_ = min(0, min(vals) * 1.25)
        fig.update_yaxes(range=[lo_, hi if hi > 0 else 1], showticklabels=False, zeroline=True, zerolinecolor="#94A3B8")
        fig.update_layout(title="Monthly cash left after all costs: can it pay the EMI?", showlegend=False)
        st.plotly_chart(style_fig(fig, 330, legend=False), use_container_width=True, config={"displayModeBar": False})

    # Where the money comes from and goes (monthly)
    credits, debits = raw["Monthly_Bank_Credits_INR"], raw["Monthly_Bank_Debits_INR"]
    upi_in = min(raw["Monthly_UPI_Inflow_INR"], credits)
    emi_now = raw.get("Monthly_Loan_EMI_INR") or 0
    payroll = raw["Monthly_Payroll_INR"]
    other = max(debits - payroll - emi_now, 0)
    surplus = max(credits - debits, 0)
    labels = ["UPI receipts", "Other bank receipts", "Monthly inflow", "Suppliers & other costs", "Payroll", "Existing EMI", "Surplus"]
    fig = go.Figure(go.Sankey(
        node=dict(label=[f"{l}<br>{inr(v)}" for l, v in zip(labels, [upi_in, credits - upi_in, credits, other, payroll, emi_now, surplus])],
                  color=["#818CF8", "#A5B4FC", INDIGO, "#CBD5E1", "#F9A8D4", "#FCA5A5", "#6EE7B7"], pad=18, thickness=16),
        link=dict(source=[0, 1, 2, 2, 2, 2], target=[2, 2, 3, 4, 5, 6],
                  value=[upi_in, credits - upi_in, other, payroll, emi_now, surplus],
                  color="rgba(99,102,241,0.18)"),
    ))
    fig.update_layout(title="Monthly money flow (Account Aggregator + UPI + EPFO)")
    st.plotly_chart(style_fig(fig, 330, legend=False), use_container_width=True, config={"displayModeBar": False})

    a, b, d = st.columns(3)
    with a:
        fig = go.Figure()
        fig.add_bar(x=["GST sales", "GST purchases"], y=[raw["Monthly_GST_Sales_INR"], raw["Monthly_GST_Purchases_INR"]],
                    marker_color=[INDIGO, "#A5B4FC"], text=[inr(raw["Monthly_GST_Sales_INR"]), inr(raw["Monthly_GST_Purchases_INR"])], textposition="outside")
        fig.update_layout(title=f"GST performance · filed on time {raw['GST_Filing_Timeliness_Pct']:.0f}% ({raw['GST_Return_Frequency']})", showlegend=False)
        fig.update_yaxes(showticklabels=False, range=[0, max(raw["Monthly_GST_Sales_INR"], raw["Monthly_GST_Purchases_INR"]) * 1.2])
        st.plotly_chart(style_fig(fig, 300, legend=False), use_container_width=True, config={"displayModeBar": False})
    with b:
        fig = go.Figure()
        fig.add_bar(x=["UPI inflow", "UPI outflow"], y=[raw["Monthly_UPI_Inflow_INR"], raw["Monthly_UPI_Outflow_INR"]],
                    marker_color=[GREEN, "#FCA5A5"], text=[inr(raw["Monthly_UPI_Inflow_INR"]), inr(raw["Monthly_UPI_Outflow_INR"])], textposition="outside")
        fig.update_layout(title=f"UPI · {raw['UPI_Daily_Txn_Count']:.1f} txns/day · avg ticket {inr(raw['UPI_Avg_Ticket_Size_INR'], False)}", showlegend=False)
        fig.update_yaxes(showticklabels=False, range=[0, max(raw["Monthly_UPI_Inflow_INR"], raw["Monthly_UPI_Outflow_INR"]) * 1.2])
        st.plotly_chart(style_fig(fig, 300, legend=False), use_container_width=True, config={"displayModeBar": False})
    with d:
        vals = [raw["Salary_Consistency_Pct"], 100 - raw["Employee_Attrition_Rate_Pct"], raw["GST_Filing_Timeliness_Pct"]]
        fig = go.Figure(go.Bar(x=vals, y=["Salary paid on schedule", "Staff retained", "GST on time"], orientation="h",
                               marker_color=[INDIGO, "#818CF8", "#A5B4FC"], text=[f"{v:.0f}%" for v in vals], textposition="inside"))
        fig.update_xaxes(range=[0, 100])
        fig.update_layout(title=f"Payroll consistency · {inr(raw['Monthly_Payroll_INR'])}/mo")
        st.plotly_chart(style_fig(fig, 300, legend=False), use_container_width=True, config={"displayModeBar": False})


def insights_tab(c: dict):
    ins = c["insights"]
    a, b = st.columns(2)
    with a:
        st.markdown("#### 💪 Strengths")
        for s in ins["strengths"] or ["No standout strengths vs the average MSME."]:
            st.markdown(f'<div class="insight" style="border-color:{GREEN}">{s}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="small">Strongest pillar: <b>{ins["strongest_pillar"]}</b></div>', unsafe_allow_html=True)
    with b:
        st.markdown("#### ⚠️ Concerns")
        for s in ins["concerns"] or ["No major concerns vs the average MSME."]:
            st.markdown(f'<div class="insight" style="border-color:{RED}">{s}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="small">Weakest pillar: <b>{ins["weakest_pillar"]}</b></div>', unsafe_allow_html=True)
    st.markdown("#### 🎯 What would improve this business's score")
    st.caption("Each action moves one metric to the level of the better-performing quarter of peers, then re-scores with the model.")
    acts = ins["improvement_actions"]
    if not acts:
        st.success("Already at or above the peer top-quarter on every actionable metric.")
    for act in acts:
        st.markdown(
            f'<div class="insight" style="border-color:{INDIGO}"><b>{act["action"]}</b><br>'
            f'<span class="small">{LABELS.get(act["feature"], act["feature"])}: {act["from"]:,} → {act["to"]:,}</span>'
            f'<span style="float:right"><b style="color:{GREEN}">+{act["score_gain"]} pts</b> · default risk −{act["pd_drop_pct_points"]} pp</span></div>',
            unsafe_allow_html=True,
        )


def render(c: dict, raw: dict, given: dict | None = None):
    header(c, raw)
    top_row(c)
    st.markdown("")
    pillars_row(c)
    st.markdown("")
    tabs = st.tabs(["🔍 Why this score", "🚦 Risk & anomalies", "👥 Peer benchmark", "💰 Revenue, cash flow, GST, UPI, payroll", "💡 Insights & actions"])
    with tabs[0]:
        why_tab(c)
    with tabs[1]:
        risk_tab(c)
    with tabs[2]:
        peers_tab(c)
    with tabs[3]:
        money_tab(c, raw)
    with tabs[4]:
        insights_tab(c)
    if given:
        with st.expander("Compare with the score supplied in the dataset"):
            g = pd.DataFrame({
                "Supplied": [given["Financial_Health_Score"], f"{given['Probability_of_Default']:.1%}", given["Credit_Risk_Category"], inr(given["Recommended_Credit_Limit_INR"])],
                "Our model": [c["financial_health_score"], f"{c['probability_of_default']:.1%}", c["risk_band"], inr(c["credit_recommendation"]["recommended_limit_inr"])],
            }, index=["Health score", "Default probability", "Risk band", "Credit limit"])
            st.table(g.astype(str))
