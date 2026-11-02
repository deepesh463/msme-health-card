"""Shared look-and-feel and chart helpers for the dashboard."""
from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from msme_health.config import ARTIFACTS, DATA_PATH, ID_COL
from msme_health.engine import get_engine

INDIGO, GREEN, AMBER, RED, SLATE = "#4F46E5", "#10B981", "#F59E0B", "#EF4444", "#64748B"
BAND_COLOR = {"Low": GREEN, "Medium": AMBER, "High": RED}
LEVEL_COLOR = {"OK": GREEN, "Watch": AMBER, "High": RED}
DECISION_COLOR = {"Approve": GREEN, "Approve with conditions": "#0EA5E9", "Refer to underwriter": AMBER, "Decline": RED}
FONT = "Inter, system-ui, -apple-system, Segoe UI, sans-serif"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, .stText { font-family: 'Inter', system-ui, sans-serif; }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1400px; }
h1, h2, h3 { letter-spacing: -0.02em; }
.hero { background: linear-gradient(120deg, #312E81 0%, #4F46E5 55%, #7C3AED 100%); color: #fff;
        border-radius: 20px; padding: 22px 28px; margin-bottom: 18px; box-shadow: 0 10px 30px rgba(79,70,229,.25); }
.hero h1 { color:#fff; margin: 0 0 4px 0; font-size: 1.75rem; font-weight: 800; }
.hero p { color: #E0E7FF; margin: 0; font-size: .95rem; }
.card { background:#fff; border:1px solid #E5E7EB; border-radius:16px; padding:16px 18px;
        box-shadow: 0 1px 2px rgba(15,23,42,.04), 0 4px 16px rgba(15,23,42,.04); height:100%; }
.card h4 { margin:0 0 6px 0; font-size:.78rem; text-transform:uppercase; letter-spacing:.06em; color:#64748B; font-weight:600; }
.kpi { font-size:1.55rem; font-weight:800; color:#0F172A; line-height:1.1; white-space:nowrap; }
.kpi-sub { color:#64748B; font-size:.82rem; margin-top:4px; }
.chip { display:inline-block; padding:3px 10px; border-radius:999px; font-size:.78rem; font-weight:600;
        margin:2px 6px 2px 0; background:#EEF2FF; color:#3730A3; border:1px solid #E0E7FF; }
.chip.grey { background:#F1F5F9; color:#334155; border-color:#E2E8F0; }
.badge { display:inline-block; padding:4px 12px; border-radius:999px; font-weight:700; font-size:.85rem; color:#fff; }
.pillar { background:#fff; border:1px solid #E5E7EB; border-radius:14px; padding:12px 14px; height:100%; min-height:186px; }
.pillar .pct { font-size:.76rem; color:#0F172A; font-weight:600; margin-bottom:4px; }
.pillar .name { font-size:.8rem; color:#64748B; font-weight:600; }
.pillar .val { font-size:1.5rem; font-weight:800; color:#0F172A; }
.pillar .grade { float:right; font-weight:800; font-size:.9rem; padding:2px 9px; border-radius:8px; }
.bar { height:6px; background:#EEF2FF; border-radius:4px; margin:8px 0 6px 0; overflow:hidden; }
.bar > div { height:100%; border-radius:4px; }
.drv { font-size:.74rem; color:#475569; }
.ind-row { display:flex; justify-content:space-between; align-items:center; padding:8px 2px; border-bottom:1px dashed #E5E7EB; font-size:.9rem; }
.dot { height:10px; width:10px; border-radius:50%; display:inline-block; margin-right:8px; }
.insight { border-left:4px solid; padding:8px 12px; border-radius:8px; background:#fff; margin-bottom:8px; font-size:.9rem; }
.small { font-size:.8rem; color:#64748B; }
div[data-testid="stMetricValue"] { font-weight: 800; }
section[data-testid="stSidebar"] { background: #0F172A; }
section[data-testid="stSidebar"] * { color: #E2E8F0 !important; }
section[data-testid="stSidebar"] input, section[data-testid="stSidebar"] [data-baseweb="select"] * { color:#0F172A !important; }
section[data-testid="stSidebar"] .stButton button { background:#4F46E5; border:1px solid #6366F1; }
section[data-testid="stSidebar"] .stButton button:hover { background:#6366F1; }
section[data-testid="stSidebar"] .stButton button p { color:#FFFFFF !important; font-weight:600; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] { background:#fff; border:1px solid #E5E7EB; border-radius:10px; padding:6px 14px; }
.stTabs [aria-selected="true"] { background:#EEF2FF; border-color:#C7D2FE; }
</style>
"""

GRADE_COLOR = {"A+": ("#DCFCE7", "#166534"), "A": ("#DCFCE7", "#15803D"), "B": ("#E0F2FE", "#075985"),
               "C": ("#FEF3C7", "#92400E"), "D": ("#FEE2E2", "#991B1B")}
GRADE_MEANING = {"A+": "Excellent", "A": "Strong", "B": "Average", "C": "Below average", "D": "Weak"}
# Pillar grades rank a business against all MSMEs (see engine.grade_from_pct).
PILLAR_GRADE_RANGE = {"A+": "better than 90%+", "A": "better than 70–90%", "B": "better than 40–70%",
                      "C": "better than 15–40%", "D": "better than under 15%"}


def grade_key(ranges: dict) -> str:
    """One-line colour key: grade, meaning and range."""
    items = "".join(
        f'<span style="display:inline-block;margin:0 10px 6px 0;font-size:.8rem">'
        f'<b style="background:{GRADE_COLOR[g][0]};color:{GRADE_COLOR[g][1]};padding:2px 8px;border-radius:7px">{g}</b> '
        f'{GRADE_MEANING[g]} <span style="color:#64748B">({r})</span></span>'
        for g, r in ranges.items()
    )
    return f'<div>{items}</div>'


def setup(title: str, subtitle: str):
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading models…")
def engine():
    return get_engine()


@st.cache_resource(show_spinner=False)
def scored() -> pd.DataFrame:
    return engine().scored


@st.cache_resource(show_spinner=False)
def raw_data() -> pd.DataFrame:
    """Raw inputs + supplied answers, indexed by ID (reuses the engine's table to save memory)."""
    s = scored().set_index(ID_COL, drop=False)
    for c in s.columns:
        if isinstance(s[c].dtype, pd.CategoricalDtype):
            s[c] = s[c].astype(str)
    return s


@st.cache_data(show_spinner=False)
def metrics() -> dict:
    return json.loads((ARTIFACTS / "metrics.json").read_text())


def inr(x: float | None, short: bool = True) -> str:
    if x is None or pd.isna(x):
        return "—"
    x = float(x)
    sign = "-" if x < 0 else ""
    x = abs(x)
    if short and x >= 1e9:
        return f"{sign}₹{x / 1e7:,.0f} Cr"
    if short and x >= 1e7:
        return f"{sign}₹{x / 1e7:,.2f} Cr"
    if short and x >= 1e5:
        return f"{sign}₹{x / 1e5:,.2f} L"
    return f"{sign}₹{x:,.0f}"


def card(title: str, value: str, sub: str = "", color: str | None = None):
    style = f"color:{color}" if color else ""
    st.markdown(f'<div class="card"><h4>{title}</h4><div class="kpi" style="{style}">{value}</div>'
                f'<div class="kpi-sub">{sub}</div></div>', unsafe_allow_html=True)


def chips(items: list[str], grey: bool = False) -> str:
    cls = "chip grey" if grey else "chip"
    return "".join(f'<span class="{cls}">{i}</span>' for i in items)


def badge(text: str, color: str) -> str:
    return f'<span class="badge" style="background:{color}">{text}</span>'


def style_fig(fig: go.Figure, height: int = 320, legend: bool = True) -> go.Figure:
    # Title sits at the very top; the legend gets its own row underneath it.
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=78 if legend else 48, b=10), paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(family=FONT, size=12, color="#0F172A"),
        title=dict(x=0.01, xanchor="left", y=0.98, yanchor="top", font=dict(size=14)),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0, title_text=""),
    )
    fig.update_xaxes(gridcolor="#EEF2F7", zeroline=False)
    fig.update_yaxes(gridcolor="#EEF2F7", zeroline=False)
    return fig


def gauge(score: float, grade: str, title: str = "Financial Health Score") -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score, number=dict(font=dict(size=46, color="#0F172A"), suffix=""),
        title=dict(text=f"{title}<br><span style='font-size:13px;color:#64748B'>Grade {grade}</span>", font=dict(size=14)),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=0, tickcolor="#CBD5E1"),
            bar=dict(color=INDIGO, thickness=0.28), bgcolor="white", borderwidth=0,
            steps=[dict(range=[0, 62], color="#FEE2E2"), dict(range=[62, 70], color="#FEF3C7"),
                   dict(range=[70, 78], color="#E0F2FE"), dict(range=[78, 100], color="#DCFCE7")],
            threshold=dict(line=dict(color="#0F172A", width=3), thickness=0.8, value=score),
        ),
    ))
    return style_fig(fig, 260, legend=False)


def pillar_tile(p: dict):
    bg, fg = GRADE_COLOR.get(p["grade"], ("#EEF2FF", INDIGO))
    width = max(2, min(100, p["score"]))
    pct = p.get("population_percentile")
    drivers = "".join(
        f'<div class="drv">{"▲" if d["impact"] > 0 else "▼"} {d["label"]}</div>' for d in p["drivers"]
    )
    st.markdown(
        f'<div class="pillar"><span class="grade" style="background:{bg};color:{fg}">{p["grade"]}</span>'
        f'<div class="name">{p["name"]}</div><div class="val">{p["score"]:.1f}</div>'
        f'<div class="bar"><div style="width:{width}%;background:{fg}"></div></div>'
        f'<div class="pct">{GRADE_MEANING.get(p["grade"], "")}: better than {pct:.0f}% of MSMEs</div>{drivers}</div>'
        if pct is not None else
        f'<div class="pct">{GRADE_MEANING.get(p["grade"], "")}</div>{drivers}</div>',
        unsafe_allow_html=True,
    )


def pick_business(key: str = "biz") -> str:
    """Sidebar picker shared by single-business pages."""
    s = scored()
    st.sidebar.markdown("### Find a business")
    seg = st.sidebar.selectbox("Segment", ["All", "NTC", "NTB", "Existing-to-Credit"], key=f"{key}_seg")
    band = st.sidebar.selectbox("Risk band", ["All", "Low", "Medium", "High"], key=f"{key}_band")
    ind = st.sidebar.selectbox("Industry", ["All"] + sorted(s.Industry.unique()), key=f"{key}_ind")
    f = s
    if seg != "All":
        f = f[f.Customer_Segment == seg]
    if band != "All":
        f = f[f.risk_band == band]
    if ind != "All":
        f = f[f.Industry == ind]
    if f.empty:
        st.sidebar.warning("No business matches. Showing all.")
        f = s
    ss = st.session_state
    id_key, typed_key, filt_key = f"{key}_id", f"{key}_typed", f"{key}_filters"

    def show(msme_id: str):
        # Keep the ID box in sync, otherwise its old text would switch the business back.
        ss[id_key] = msme_id
        ss[typed_key] = msme_id

    filters = (seg, band, ind)
    if id_key not in ss or (ss.get(filt_key) != filters and ss[id_key] not in set(f[ID_COL])):
        show(f[ID_COL].iloc[0])
    ss[filt_key] = filters
    if st.sidebar.button("🎲  Random business", use_container_width=True, key=f"{key}_rand"):
        show(f[ID_COL].sample(1).iloc[0])
    with st.sidebar.form(f"{key}_search", border=False):
        st.text_input("Search by MSME ID", key=typed_key, placeholder="e.g. MSME0035912")
        searched = st.form_submit_button("🔍  Search", use_container_width=True)
    typed = (ss.get(typed_key) or "").strip().upper()
    if searched and typed and typed != ss[id_key]:
        if typed in set(s[ID_COL]):
            ss[id_key] = typed
        else:
            st.sidebar.warning(f"{typed} not found.")
    st.sidebar.caption(f"{len(f):,} businesses match the filters")
    return st.session_state[f"{key}_id"]
