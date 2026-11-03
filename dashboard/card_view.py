"""Renders a full Health Card. Used by the Health Card page and the 'New applicant assessment' page."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from msme_health.config import SOURCE_OF
from msme_health.features import LABELS

from ui import (AMBER, BAND_COLOR, DECISION_COLOR, GREEN, INDIGO, LEVEL_COLOR, RED, SLATE, badge,
                card, chips, gauge, grade_key, overall_grade_key, inr, pillar_tile, style_fig)

PCT_METRICS = {"Net_Surplus_Ratio", "Lean_Month_Surplus_Ratio", "Overdraft_Usage_Ratio", "Customer_Concentration_Ratio"}


def _fmt(metric: str, v: float) -> str:
    if metric.endswith("_INR"):
        return inr(v)
    if metric.endswith("_Pct"):
        return f"{v:.1f}%"
    if metric in PCT_METRICS:
        return f"{v:.0%}"
    if metric == "Years_in_Operation":
        return f"{v:.1f} years"
    if metric.endswith("_Days"):
        return f"{v:.0f} days"
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
    pills = "".join(
        f'<span style="display:inline-block;margin:2px 4px 2px 0;padding:3px 10px;border-radius:999px;font-size:.75rem;'
        f'font-weight:600;background:{"#DCFCE7" if ok else "#FEE2E2"};color:{"#166534" if ok else "#991B1B"}">'
        f'{"✓" if ok else "✕"} {name}</span>'
        for name, ok in cov["sources"].items()
    )
    note = f'<div style="font-size:.75rem;color:#64748B;margin-top:6px">ℹ️ {cov["note"]}</div>' if cov["note"] else ""
    st.markdown(
        '<div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:12px;padding:10px 14px;margin:4px 0 10px">'
        '<div style="display:flex;flex-wrap:wrap;align-items:center;gap:6px">'
        '<span style="font-size:.78rem;font-weight:700;color:#0F172A;margin-right:4px">Data sources</span>'
        f'{pills}'
        f'</div>{note}</div>',
        unsafe_allow_html=True,
    )


def top_row(c: dict):
    rec = c["credit_recommendation"]
    a, b, d = st.columns([1.15, 1, 1])
    with a:
        st.plotly_chart(gauge(c["financial_health_score"], c["grade"]), use_container_width=True, config={"displayModeBar": False})
        st.markdown(overall_grade_key(c["grade"]), unsafe_allow_html=True)
    with b:
        col = DECISION_COLOR.get(rec["decision"], INDIGO)
        st.markdown(
            f'<div class="card"><h4>Credit decision</h4>{badge(rec["decision"], col)}'
            f'<div class="kpi" style="margin-top:12px">{inr(rec["recommended_limit_inr"])}</div>'
            f'<div class="kpi-sub">Recommended limit'
            + (f' · {rec["tenure_months"]} mo @ {rec["indicative_rate_pct"]:.0f}% · EMI {inr(rec["estimated_emi_inr"])}' if rec["eligible"] else "")
            + "</div>"
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
    st.markdown(grade_key(), unsafe_allow_html=True)
    cols = st.columns(6)
    for col, p in zip(cols, c["pillars"]):
        with col:
            pillar_tile(p)


def reasons_block(c: dict) -> str:
    """Top reasons raising default risk: value vs peers, data source and relative strength."""
    codes = c["explanation"]["risk_reason_codes"]
    head = ('<div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:14px;padding:12px 14px">'
            '<div style="font-size:.9rem;font-weight:700;color:#0F172A">⚠️ Top reasons raising default risk</div>'
            '<div style="font-size:.74rem;color:#64748B;margin-bottom:8px">Strongest first. Each one pushes this business\'s risk above the average MSME.</div>')
    if not codes:
        return head + '<div style="font-size:.85rem;color:#166534">✅ No factor is raising risk above the average MSME.</div></div>'
    peers = {m["metric"]: m for m in (c.get("peer_benchmark") or {}).get("metrics", [])}
    top = max(r["impact_log_odds"] for r in codes) or 1
    rows = ""
    for i, r in enumerate(codes, 1):
        v = r["value"]
        val = "—" if v is None else (_fmt(r["feature"], v) if isinstance(v, (int, float)) else str(v))
        pm = peers.get(r["feature"])
        peer = f' <span style="color:#64748B">· similar businesses: {_fmt(r["feature"], pm["peer_median"])}</span>' if pm and v is not None else ""
        src = SOURCE_OF.get(r["feature"])
        src_pill = (f'<span style="font-size:.66rem;background:#EEF2FF;color:#4338CA;padding:1px 7px;border-radius:999px;margin-left:6px">{src}</span>'
                    if src else "")
        width = max(8, round(r["impact_log_odds"] / top * 100))
        rows += (
            '<div style="display:flex;gap:10px;align-items:flex-start;padding:8px 0;border-top:1px solid #F1F5F9">'
            f'<div style="flex:none;width:24px;height:24px;border-radius:50%;background:#FEE2E2;color:#991B1B;font-weight:800;'
            f'font-size:.8rem;display:flex;align-items:center;justify-content:center">{i}</div>'
            '<div style="flex:1;min-width:0">'
            f'<div style="font-size:.85rem;font-weight:700;color:#0F172A">{r["label"]}{src_pill}</div>'
            f'<div style="font-size:.78rem;color:#334155">This business: <b>{val}</b>{peer}</div>'
            f'<div style="height:5px;background:#F1F5F9;border-radius:3px;margin-top:5px"><div style="width:{width}%;height:5px;background:#F87171;border-radius:3px"></div></div>'
            '</div></div>'
        )
    foot = ('<div style="font-size:.7rem;color:#64748B;margin-top:6px">These are the <b>reason codes</b> shared with the business '
            'if credit is declined or limited (RBI Fair Practices Code).</div>')
    return head + rows + foot + "</div>"


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
        st.markdown(reasons_block(c), unsafe_allow_html=True)


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


def _val(feature: str, v) -> str:
    return "—" if v is None else (_fmt(feature, v) if isinstance(v, (int, float)) else str(v))


def _factor_card(title: str, icon: str, accent: str, soft: str, drivers: list, empty: str, pillar_label: str, pillar) -> str:
    """Strengths or concerns: each factor with this business's value and its effect in score points."""
    rows = "".join(
        '<div style="display:flex;justify-content:space-between;align-items:center;gap:10px;padding:9px 0;border-top:1px solid #F1F5F9">'
        f'<div style="min-width:0"><div style="font-size:.86rem;font-weight:600;color:#0F172A">{d["label"]}</div>'
        f'<div style="font-size:.74rem;color:#64748B">This business: {_val(d["feature"], d["value"])}</div></div>'
        f'<span style="flex:none;font-size:.78rem;font-weight:700;background:{soft};color:{accent};padding:3px 10px;border-radius:999px">'
        f'{d["impact_points"]:+.1f} pts</span></div>'
        for d in drivers
    ) or f'<div style="font-size:.84rem;color:#64748B;padding:9px 0;border-top:1px solid #F1F5F9">{empty}</div>'
    pill = ""
    if pillar:
        pill = (f'<div style="margin-top:10px;font-size:.78rem;color:#475569">{pillar_label}: '
                f'<b style="color:#0F172A">{pillar["name"]}</b> · grade {pillar["grade"]}'
                + (f' · better than {pillar["population_percentile"]:.0f}% of MSMEs' if pillar.get("population_percentile") is not None else "")
                + '</div>')
    return (
        f'<div style="background:#FFFFFF;border:1px solid #E2E8F0;border-top:4px solid {accent};border-radius:14px;padding:14px 16px;height:100%">'
        f'<div style="font-size:.95rem;font-weight:700;color:#0F172A;margin-bottom:4px">{icon} {title}</div>'
        f'{rows}{pill}</div>'
    )


def insights_tab(c: dict):
    ins = c["insights"]
    drivers = c["explanation"]["top_drivers"]
    pillars = {p["name"]: p for p in c["pillars"]}
    ups = [d for d in drivers if d["impact_points"] > 0][:3]
    downs = [d for d in drivers if d["impact_points"] < 0][:3]
    a, b = st.columns(2)
    with a:
        st.markdown(_factor_card("Strengths", "💪", "#15803D", "#DCFCE7", ups, "No standout strengths vs the average MSME.",
                                 "Strongest pillar", pillars.get(ins["strongest_pillar"])), unsafe_allow_html=True)
    with b:
        st.markdown(_factor_card("Concerns", "⚠️", "#B91C1C", "#FEE2E2", downs, "No major concerns vs the average MSME.",
                                 "Weakest pillar", pillars.get(ins["weakest_pillar"])), unsafe_allow_html=True)
    st.caption("Points = how much each factor moved this business's score compared with the average MSME (SHAP).")

    st.markdown("#### 🎯 How this business can improve its score")
    st.caption("Each action moves one number to the level of the best 25% of similar businesses, then re-scores with the model.")
    acts = ins["improvement_actions"]
    if not acts:
        st.success("Already at or above the best 25% of similar businesses on every number it can change.")
        return
    pd_now = c["probability_of_default"]
    cols = st.columns(2)
    for i, act in enumerate(acts):
        f = act["feature"]
        with cols[i % 2]:
            st.markdown(
                '<div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:14px;padding:14px 16px;margin-bottom:12px">'
                '<div style="display:flex;gap:10px;align-items:flex-start">'
                f'<div style="flex:none;width:26px;height:26px;border-radius:50%;background:#EEF2FF;color:#4338CA;font-weight:800;'
                f'font-size:.82rem;display:flex;align-items:center;justify-content:center">{i + 1}</div>'
                '<div style="flex:1;min-width:0">'
                f'<div style="font-size:.9rem;font-weight:700;color:#0F172A">{act["action"]}</div>'
                f'<div style="font-size:.78rem;color:#475569;margin-top:3px">{LABELS.get(f, f)}: '
                f'<b>{_val(f, act["from"])}</b> → <b style="color:#15803D">{_val(f, act["to"])}</b></div>'
                '<div style="margin-top:8px;display:flex;gap:6px;flex-wrap:wrap">'
                f'<span style="font-size:.76rem;font-weight:700;background:#DCFCE7;color:#166534;padding:3px 10px;border-radius:999px">Score +{act["score_gain"]} pts</span>'
                f'<span style="font-size:.76rem;font-weight:700;background:#EEF2FF;color:#4338CA;padding:3px 10px;border-radius:999px">'
                f'Default risk {pd_now:.1%} → {max(pd_now - act["pd_drop_pct_points"] / 100, 0):.1%}</span>'
                '</div></div></div></div>',
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
