"""Streamlit entry point.  Run from the repo root:  streamlit run dashboard/app.py"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(page_title="MSME Health Card", page_icon="📇", layout="wide")

pages = st.navigation({
    "Lender view": [
        st.Page("pages/portfolio.py", title="Portfolio overview", icon="📊", default=True),
        st.Page("pages/health_card.py", title="MSME Health Card", icon="📇"),
        st.Page("pages/what_if.py", title="What-if simulator", icon="🧪"),
        st.Page("pages/new_business.py", title="Score a new business", icon="➕"),
    ],
    "Under the hood": [
        st.Page("pages/model.py", title="Model & methodology", icon="🧠"),
    ],
})
st.sidebar.markdown("**Cars24 · MSME Financial Health Card**")
st.sidebar.caption("Alternate-data credit scoring for NTC / NTB small businesses")
pages.run()
