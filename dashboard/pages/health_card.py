import streamlit as st

import card_view
from ui import engine, pick_business, raw_data, setup

setup("📇 MSME Financial Health Card",
      "One view of a business's financial health from GST, UPI, bank (Account Aggregator), EPFO and invoice data.")

biz = pick_business("hc")
raw = raw_data()
row = raw.loc[biz].to_dict()
row["MSME_ID"] = biz
with st.spinner("Scoring…"):
    c = engine().health_card(row)
card_view.render(c, row, given=row)
