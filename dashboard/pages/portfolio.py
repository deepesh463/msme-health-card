import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from msme_health.config import PILLARS
from ui import AMBER, BAND_COLOR, DECISION_COLOR, GREEN, INDIGO, RED, SLATE, card, inr, scored, setup, style_fig

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
    fig = px.histogram(f, x="score_overall", color="risk_band", nbins=60, color_discrete_map=BAND_COLOR,
                       category_orders={"risk_band": ["Low", "Medium", "High"]}, opacity=0.9,
                       labels={"score_overall": "Financial Health Score", "risk_band": "Risk band"})
    fig.update_layout(title="Score distribution by risk band", bargap=0.05, barmode="stack")
    st.plotly_chart(style_fig(fig, 360), use_container_width=True)
with b:
    d = f.decision.value_counts().reindex(list(DECISION_COLOR)).fillna(0)
    fig = go.Figure(go.Pie(labels=d.index, values=d.values, hole=0.62, marker_colors=list(DECISION_COLOR.values()),
                           textinfo="percent", sort=False))
    fig.update_layout(title="Credit decisions", annotations=[dict(text=f"<b>{len(f):,}</b><br>MSMEs", showarrow=False, font_size=15)])
    st.plotly_chart(style_fig(fig, 360), use_container_width=True)

a, b = st.columns(2)
with a:
    g = f.groupby("Customer_Segment").risk_band.value_counts(normalize=True).rename("share").reset_index()
    fig = px.bar(g, x="share", y="Customer_Segment", color="risk_band", orientation="h", color_discrete_map=BAND_COLOR,
                 category_orders={"risk_band": ["Low", "Medium", "High"], "Customer_Segment": ["NTC", "NTB", "Existing-to-Credit"]},
                 labels={"share": "Share of businesses", "Customer_Segment": ""}, text_auto=".0%")
    fig.update_layout(title="Risk mix by customer segment: NTC businesses are not far behind", xaxis_tickformat=".0%")
    st.plotly_chart(style_fig(fig, 320), use_container_width=True)
with b:
    g = (f.assign(ok=approved).groupby("Industry", observed=True)
         .agg(rate=("ok", "mean"), n=("MSME_ID", "size")).sort_values("rate"))
    fig = go.Figure(go.Bar(x=g.rate, y=g.index.astype(str), orientation="h", marker_color=INDIGO,
                           text=[f"{r:.0%}  (n={n:,})" for r, n in zip(g.rate, g.n)], textposition="inside",
                           insidetextanchor="end", textfont_color="white"))
    fig.add_vline(x=approved.mean(), line_dash="dot", line_color=SLATE,
                  annotation_text=f"all {approved.mean():.0%}", annotation_position="top")
    fig.update_xaxes(range=[0, 1], tickformat=".0%")
    fig.update_layout(title="Approval rate by industry")
    st.plotly_chart(style_fig(fig, 340, legend=False), use_container_width=True)

a, b = st.columns([1.2, 1])
with a:
    cols = {"score_" + c.replace("_Score", ""): n for c, n in PILLARS.items()}
    h = f.groupby("risk_band", observed=True)[list(cols)].mean().rename(columns=cols)
    h = h.reindex([x for x in ["Low", "Medium", "High"] if x in h.index])
    h.index = [f"{x} risk" for x in h.index]
    # Colour each pillar column on its own scale (pillars have different natural ranges); show real values as text.
    z = (h - h.min()) / (h.max() - h.min()).replace(0, 1)
    fig = go.Figure(go.Heatmap(z=z.values, x=list(h.columns), y=list(h.index), text=h.round(1).values,
                               texttemplate="%{text}", colorscale="Blues", showscale=False, hoverinfo="skip"))
    fig.update_layout(title="Average pillar scores by risk band: where High-risk businesses fall short")
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(style_fig(fig, 300, legend=False), use_container_width=True)
with b:
    t = f.loc[approved, "binding_constraint"].map({
        "policy (% of turnover)": "Turnover policy (16% / 8%)",
        "affordability (cash surplus)": "Cash surplus (affordability)",
    }).value_counts()
    fig = go.Figure(go.Bar(x=t.values, y=t.index, orientation="h", marker_color=[INDIGO, AMBER][:len(t)],
                           text=[f"{v:,} ({v / t.sum():.0%})" for v in t.values], textposition="outside", cliponaxis=False))
    fig.update_xaxes(range=[0, t.max() * 1.6], showticklabels=False)
    fig.update_layout(title="What sets the loan amount (approved businesses)")
    st.plotly_chart(style_fig(fig, 220, legend=False), use_container_width=True)
    st.caption("Most limits come from the turnover policy. For about 1 in 8 approved businesses, the monthly cash surplus "
               "cannot service an EMI that large, so the affordability check lowers the limit. That keeps the portfolio prudent.")

st.markdown("#### 🚩 Businesses sent for manual review")
rv = f[flagged].sort_values("anomaly_percentile", ascending=False).head(200)
rv = pd.DataFrame({
    "MSME ID": rv.MSME_ID, "Segment": rv.Customer_Segment, "Industry": rv.Industry, "Location": rv.Location_Category,
    "Health score": rv.score_overall.round(1), "Default risk": rv.pred_pd.map("{:.1%}".format),
    "Unusualness (0-100)": rv.anomaly_percentile.round(0).astype(int), "Rule flags": rv.rule_flag_count, "Decision": rv.decision,
})
st.dataframe(rv, use_container_width=True, hide_index=True, height=300)
st.caption("Copy an MSME ID into the **MSME Health Card** page to see which checks failed.")
