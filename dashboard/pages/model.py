import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from msme_health.config import ARTIFACTS
from msme_health.features import FEATURE_PILLAR, LABELS
from ui import AMBER, GREEN, INDIGO, card, metrics, setup, style_fig

setup("🧠 Model & methodology", "How the score is built, how well it works, and where its limits are.")
m = metrics()
tp = pd.read_parquet(ARTIFACTS / "test_predictions.parquet")

main = m["pd_models"]["LightGBM (monotonic)"]
k = st.columns(5)
with k[0]:
    card("Default-risk R²", f"{main['pd_r2']:.3f}", "vs supplied PD, held-out test set")
with k[1]:
    card("Risk-band accuracy", f"{main['risk_band_accuracy']:.1%}", "Low / Medium / High")
with k[2]:
    card("Eligibility accuracy", f"{main['eligibility_accuracy']:.1%}", "eligible vs not")
with k[3]:
    card("High-risk AUC", f"{main['high_risk_auc']:.3f}", "separating High from the rest")
with k[4]:
    card("Gini (simulated defaults)", f"{main['sim_default_gini']:.3f}",
         f"ceiling {m['pd_models']['Given PD (ceiling)']['sim_default_gini']:.3f}")

st.markdown("")
t1, t2, t3, t4, t5 = st.tabs(["Model comparison", "Explainability", "Calibration & fit", "Fairness by segment", "Anomaly detection"])
with t1:
    rows = []
    for name, v in m["pd_models"].items():
        if "pd_r2" in v:
            rows.append({"Model": name, "PD R²": v["pd_r2"], "PD MAE": v["pd_mae"], "Band accuracy": v["risk_band_accuracy"],
                         "High-risk AUC": v["high_risk_auc"], "Gini": v["sim_default_gini"], "KS": v["sim_default_ks"]})
    st.markdown("**Default-probability models** (test set, 7,500 businesses)")
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    sm = pd.DataFrame(m["score_models"]).T.reset_index().rename(columns={"index": "Model", "r2": "Score R²", "mae": "Score MAE (points)"})
    pm = pd.DataFrame(m["pillar_models"]).T.reset_index().rename(columns={"index": "Pillar", "r2": "R²", "mae": "MAE (points)"})
    a, b = st.columns(2)
    a.markdown("**Financial Health Score models**")
    a.dataframe(sm, hide_index=True, use_container_width=True)
    b.markdown("**Pillar models**")
    b.dataframe(pm, hide_index=True, use_container_width=True)
    st.info("**Key finding:** the linear baseline matches LightGBM. The supplied scores look like a near-linear formula plus noise, "
            "so a more complex model gains nothing. Gini against simulated defaults is within 0.02 of the ceiling set by the supplied PD itself.")
with t2:
    imp = pd.Series(m["pd_shap_importance"]).head(15)[::-1]
    a, b = st.columns([1.3, 1])
    with a:
        fig = go.Figure(go.Bar(x=imp.values, y=[LABELS.get(i, i) for i in imp.index], orientation="h", marker_color=INDIGO))
        fig.update_layout(title="Top drivers of default risk (mean |SHAP|, log-odds)")
        st.plotly_chart(style_fig(fig, 480), use_container_width=True)
    with b:
        allimp = pd.Series(m["pd_shap_importance"])
        by_p = allimp.groupby(allimp.index.map(lambda f: FEATURE_PILLAR.get(f, "Other"))).sum().sort_values()
        share = by_p / by_p.sum()
        fig = go.Figure(go.Bar(
            x=share.values, y=list(share.index), orientation="h",
            marker_color=[INDIGO if v == share.max() else "#A5B4FC" for v in share.values],
            text=[f"{v:.0%}" for v in share.values], textposition="outside", cliponaxis=False,
            hovertemplate="%{y}: %{x:.1%} of the model's explanation<extra></extra>",
        ))
        fig.update_xaxes(visible=False, range=[0, float(share.max()) * 1.25])
        fig.update_layout(title="Which pillar explains default risk most")
        style_fig(fig, 380, legend=False)
        fig.update_layout(margin=dict(r=40))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.caption("Share of the model's total SHAP importance, grouped by pillar.")
    st.caption("Each Health Card shows its own local SHAP explanation (waterfall) and adverse-action reason codes.")
with t3:
    a, b = st.columns(2)
    with a:
        cal = pd.DataFrame(m["pd_calibration"])
        fig = go.Figure()
        fig.add_scatter(x=cal["pred"], y=cal["true"], mode="lines+markers", name="Model", line_color=INDIGO)
        mx = max(cal.max())
        fig.add_scatter(x=[0, mx], y=[0, mx], mode="lines", name="Perfect", line=dict(dash="dot", color="#94A3B8"))
        fig.update_layout(title="Calibration: predicted vs supplied PD by decile", xaxis_title="Predicted PD", yaxis_title="Supplied PD")
        st.plotly_chart(style_fig(fig, 380), use_container_width=True)
    with b:
        smp = tp.sample(2500, random_state=0)
        fig = px.scatter(smp, x="true_score", y="pred_score", opacity=0.35, color_discrete_sequence=[INDIGO],
                         labels={"true_score": "Supplied score", "pred_score": "Predicted score"})
        fig.add_scatter(x=[45, 100], y=[45, 100], mode="lines", line=dict(dash="dot", color="#94A3B8"), showlegend=False)
        fig.update_layout(title="Health score: predicted vs supplied (test set)")
        st.plotly_chart(style_fig(fig, 380), use_container_width=True)
with t4:
    seg = pd.DataFrame(m["pd_by_segment"]).T[["pd_r2", "risk_band_accuracy", "eligibility_accuracy", "high_risk_auc", "sim_default_gini"]]
    st.dataframe(seg.rename(columns={"pd_r2": "PD R²", "risk_band_accuracy": "Band accuracy", "eligibility_accuracy": "Eligibility accuracy",
                                     "high_risk_auc": "High-risk AUC", "sim_default_gini": "Gini"}), use_container_width=True)
    st.caption("The model works as well for NTC businesses (no credit history) as for existing borrowers. Alternate data carries the signal.")
with t5:
    an = m["anomaly"]
    rows = [{"Planted case": k, **{c.replace("_", " ").title(): f"{v:.0%}" for c, v in d.items()}} for k, d in an["detection_rate_planted"].items()]
    st.markdown("**Detection rate on planted fraud cases** (300 each, test set)")
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    fa = an["false_alarm_rate_clean"]
    st.markdown(f"False alarms on clean businesses: Isolation Forest **{fa['isolation_forest']:.1%}**, rules **{fa['rules']:.1%}**, combined **{fa['combined']:.1%}**.")
    st.caption("Cross-source rules catch the fraud types they encode. The Isolation Forest is a safety net for patterns we did not anticipate.")

with st.expander("Methodology in brief", expanded=False):
    st.markdown("""
1. **Data**: 50,000 synthetic MSMEs with 33 business inputs from GSTN, UPI, Account Aggregator bank data, bureau, EPFO and invoices.
2. **Features**: 11 analyst ratios on top of the raw data: cross-source gaps (GST vs bank), cash surplus margin, balance cover,
   EMI burden, gross margin, payroll share, a seasonal *lean-month* stress test, and a thin-file flag.
3. **Models**: LightGBM regressors for default probability (log-odds), the overall score and each pillar, with **monotonic
   constraints** so behaviour matches credit common sense. A Ridge regression is the scorecard-style baseline.
4. **Decision**: risk band from predicted PD (6% / 15% cut-offs). Limit = lower of a turnover-based policy limit and an
   affordability limit (EMI ≤ 50% of monthly surplus). Seasonal stress and data flags add conditions or send to review.
5. **Explainability**: SHAP per business (waterfall + reason codes), peer benchmarks, and model-based what-if actions.
6. **Validation**: 70/15/15 split stratified by segment and risk; test-set metrics; credit metrics (Gini/KS) on defaults
   simulated from the supplied PD, because the dataset has no observed defaults.
""")
