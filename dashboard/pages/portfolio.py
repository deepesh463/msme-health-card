import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from msme_health.config import PILLARS
from ui import (AMBER, BAND_COLOR, DECISION_COLOR, GREEN, INDIGO, RED, SLATE, card, chart_card, inr, scored, setup,
                show_fig, style_fig)

setup("📊 Portfolio overview",
      "50,000 MSMEs scored on alternate data: how many we can lend to, how risky they are, and where the risk sits.")

s = scored()
st.sidebar.markdown("### Filters")
seg = st.sidebar.multiselect("Segment", ["NTC", "NTB", "Existing-to-Credit"], default=["NTC", "NTB", "Existing-to-Credit"])
ind = st.sidebar.multiselect("Industry", sorted(s.Industry.unique()))
loc = st.sidebar.multiselect("Location", ["Metro", "Tier 1", "Tier 2", "Tier 3", "Rural"])
f = s[s.Customer_Segment.isin(seg)]
if ind:
    f = f[f.Industry.isin(ind)]
if loc:
    f = f[f.Location_Category.isin(loc)]
if f.empty:
    st.warning("No businesses match these filters.")
    st.stop()

approved = f.decision.isin(["Approve", "Approve with conditions"])
flagged = (f.rule_flag_count > 0) | f.is_outlier
ntc = f[f.Customer_Segment == "NTC"]
k = st.columns(6)
with k[0]:
    card("Businesses", f"{len(f):,}", "scored in batch")
with k[1]:
    card("Avg health score", f"{f.score_overall.mean():.1f}", f"median default risk {f.pred_pd.median():.1%}")
with k[2]:
    card("Approval rate", f"{approved.mean():.0%}", "approve + approve-with-conditions", GREEN)
with k[3]:
    card("NTC approval rate", f"{(ntc.decision.isin(['Approve', 'Approve with conditions'])).mean():.0%}" if len(ntc) else "—",
         "no credit history at all", INDIGO)
with k[4]:
    card("Credit recommended", inr(f.rec_limit.sum()), f"avg {inr(f.loc[approved, 'rec_limit'].mean())} per approval")
with k[5]:
    card("Sent for review", f"{flagged.sum():,}", f"{flagged.mean():.1%} anomaly / consistency flags", AMBER)

st.markdown("")
a, b = st.columns([1.3, 1])
with a:
    with chart_card("Score distribution by risk band", "How many businesses sit at each Financial Health Score"):
        fig = px.histogram(f, x="score_overall", color="risk_band", nbins=50, color_discrete_map=BAND_COLOR,
                           category_orders={"risk_band": ["Low", "Medium", "High"]}, opacity=0.95,
                           labels={"score_overall": "Financial Health Score", "risk_band": "Risk band"})
        fig.update_layout(bargap=0.08, barmode="stack", legend_title_text="")
        fig.update_traces(marker_line_width=0, marker_cornerradius=3)
        fig.update_yaxes(title="Businesses", showgrid=True)
        fig.update_xaxes(showgrid=False)
        show_fig(fig, 330)
with b:
    with chart_card("Credit decisions", "Share of businesses by recommended decision"):
        d = f.decision.value_counts().reindex(list(DECISION_COLOR)).fillna(0)
        fig = go.Figure(go.Pie(labels=d.index, values=d.values, hole=0.64, marker=dict(colors=list(DECISION_COLOR.values()),
                               line=dict(color="#FFFFFF", width=3)), textinfo="percent", textfont=dict(size=12, color="#FFFFFF"),
                               sort=False, hovertemplate="%{label}: %{value:,} (%{percent})<extra></extra>"))
        fig.update_layout(annotations=[dict(text=f"<b style='font-size:20px'>{len(f):,}</b><br><span style='color:#64748B'>MSMEs</span>",
                                            showarrow=False, font_size=13)],
                          legend=dict(orientation="v", x=1.0, y=0.5, xanchor="left", yanchor="middle"))
        style_fig(fig, 330)
        fig.update_layout(title=None, margin=dict(l=8, r=8, t=6, b=8),
                          legend=dict(orientation="v", x=1.02, y=0.5, xanchor="left", yanchor="middle"))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

a, b = st.columns(2)
with a:
    with chart_card("Risk mix by customer segment", "New-to-credit (NTC) businesses are not far behind existing borrowers"):
        g = f.groupby("Customer_Segment").risk_band.value_counts(normalize=True).rename("share").reset_index()
        fig = px.bar(g, x="share", y="Customer_Segment", color="risk_band", orientation="h", color_discrete_map=BAND_COLOR,
                     category_orders={"risk_band": ["Low", "Medium", "High"], "Customer_Segment": ["NTC", "NTB", "Existing-to-Credit"]},
                     labels={"share": "Share of businesses", "Customer_Segment": ""}, text_auto=".0%")
        fig.update_traces(marker_line_color="#FFFFFF", marker_line_width=2, textfont=dict(color="#FFFFFF", size=12),
                          insidetextanchor="middle")
        fig.update_layout(xaxis_tickformat=".0%", legend_title_text="", bargap=0.35)
        fig.update_xaxes(showgrid=False, range=[0, 1])
        show_fig(fig, 300)
with b:
    g = (f.assign(ok=approved).groupby("Industry", observed=True)
         .agg(rate=("ok", "mean"), n=("MSME_ID", "size")).sort_values("rate"))
    overall = approved.mean()
    g = g.sort_values("rate", ascending=False)
    rows = ""
    for name, r in g.iterrows():
        above = r.rate >= overall
        rows += (
            '<div style="display:grid;grid-template-columns:175px 1fr 44px;align-items:center;gap:12px;padding:5px 0;'
            'border-top:1px solid #F1F5F9">'
            f'<div style="white-space:nowrap;overflow:hidden;text-overflow:ellipsis"><span style="font-size:.82rem;font-weight:600;color:#0F172A">{name}</span>'
            f'<span style="font-size:.7rem;color:#94A3B8"> · {int(r.n):,}</span></div>'
            '<div style="position:relative;height:10px;background:#EEF2FF;border-radius:999px">'
            f'<div style="width:{r.rate * 100:.1f}%;height:10px;background:{INDIGO if above else "#A5B4FC"};border-radius:999px"></div>'
            f'<div style="position:absolute;left:{overall * 100:.1f}%;top:-4px;height:18px;border-left:2px dotted #334155"></div></div>'
            f'<div style="text-align:right;font-size:.86rem;font-weight:700;color:{INDIGO if above else "#475569"}">{r.rate:.0%}</div>'
            '</div>'
        )
    with chart_card("Approval rate by industry",
                    f"Share approved · number of businesses in grey · dotted line = all businesses ({overall:.0%})"):
        st.markdown(f'<div style="padding:2px 2px 4px">{rows}</div>', unsafe_allow_html=True)

a, b = st.columns([1.2, 1])
with a:
    cols = {"score_" + c.replace("_Score", ""): n for c, n in PILLARS.items()}
    h = f.groupby("risk_band", observed=True)[list(cols)].mean().rename(columns=cols)
    h = h.reindex([x for x in ["Low", "Medium", "High"] if x in h.index])
    # Table: rows = risk groups, columns = pillars. Each column is shaded on its own scale,
    # because pillars have different natural ranges (Growth ~40, Compliance ~85).
    lo, hi = h.min(), h.max()
    shades = ["#EFF6FF", "#BFDBFE", "#60A5FA", "#1D4ED8"]  # light -> dark blue

    def cell(col, v):
        t = 0.0 if hi[col] == lo[col] else (v - lo[col]) / (hi[col] - lo[col])
        bg = shades[min(int(t * len(shades)), len(shades) - 1)]
        fg = "#FFFFFF" if t >= 0.75 else "#0F172A"
        return (f'<td style="background:{bg};color:{fg};font-weight:600;text-align:center;padding:10px 4px;'
                f'border:3px solid #FFFFFF;border-radius:10px">{v:.1f}</td>')

    short = {"Business Stability": "Stability", "Cash Flow": "Cash flow", "Revenue Consistency": "Revenue",
             "Payment Behaviour": "Payment", "Business Growth": "Growth", "Compliance": "Compliance"}
    th = "".join(f'<th title="{c}" style="font-size:.72rem;font-weight:600;color:#475569;padding:4px 0 8px;text-align:center;white-space:nowrap;border:none">'
                 f'{short.get(c, c)}</th>' for c in h.columns)
    body = ""
    for band in h.index:
        dot = f'<span style="display:inline-block;width:9px;height:9px;border-radius:50%;background:{BAND_COLOR[band]};margin-right:6px"></span>'
        body += (f'<tr><td style="font-size:.8rem;font-weight:600;color:#0F172A;white-space:nowrap;padding-right:8px;border:none">{dot}{band} risk</td>'
                 + "".join(cell(c, h.loc[band, c]) for c in h.columns) + "</tr>")
    st.markdown(
        '<style>.ptab table,.ptab th,.ptab tr{border:none !important}.ptab tr:nth-child(even){background:none}</style>'
        '<div class="ptab" style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:14px;padding:14px 16px">'
        '<div style="font-size:.95rem;font-weight:700;color:#0F172A">Where high-risk businesses fall short</div>'
        '<div style="font-size:.76rem;color:#64748B;margin-bottom:10px">Average pillar score by risk group. '
        'Darker blue = higher score within that pillar.</div>'
        f'<table style="width:100%;border-collapse:separate;border-spacing:0;table-layout:fixed">'
        f'<colgroup><col style="width:100px"></colgroup><tr><th style="border:none"></th>{th}</tr>{body}</table></div>',
        unsafe_allow_html=True,
    )
with b:
    t = f.loc[approved, "binding_constraint"].map({
        "policy (% of turnover)": "Turnover policy (16% / 8%)",
        "affordability (cash surplus)": "Cash surplus (affordability)",
    }).value_counts()
    with chart_card("What sets the loan amount", "Approved businesses: which of the two limits was lower"):
        fig = go.Figure(go.Bar(x=t.values, y=t.index, orientation="h",
                               marker=dict(color=[INDIGO if "Turnover" in i else AMBER for i in t.index], cornerradius=6),
                               text=[f"<b>{v:,}</b> ({v / t.sum():.0%})" for v in t.values], textposition="outside",
                               cliponaxis=False, hovertemplate="%{y}: %{x:,}<extra></extra>"))
        fig.update_xaxes(range=[0, t.max() * 1.8], showticklabels=False, showgrid=False)
        fig.update_layout(bargap=0.45)
        show_fig(fig, 160, legend=False)
        st.caption("For about 1 in 8 approved businesses, the monthly cash surplus cannot pay the EMI the turnover "
                   "policy allows, so the affordability check lowers the limit.")

st.markdown("")
review = chart_card("🚩 Businesses sent for manual review",
                    "A data check failed or the profile is unusual. Most unusual first (top 200).")
rv = f[flagged].sort_values("anomaly_percentile", ascending=False).head(200)
rv = pd.DataFrame({
    "MSME ID": rv.MSME_ID, "Segment": rv.Customer_Segment, "Industry": rv.Industry, "Location": rv.Location_Category,
    "Health score": rv.score_overall.round(1), "Default risk": rv.pred_pd.map("{:.1%}".format),
    "Unusualness (0-100)": rv.anomaly_percentile.round(0).astype(int), "Rule flags": rv.rule_flag_count, "Decision": rv.decision,
})
review.dataframe(rv, use_container_width=True, hide_index=True, height=300)
review.caption("Copy an MSME ID into the **MSME Health Card** page to see which checks failed.")
