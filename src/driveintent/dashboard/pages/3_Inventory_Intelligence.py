"""Inventory Intelligence: aging, sell-through, demand-supply."""
import plotly.express as px
import streamlit as st

from driveintent.dashboard.state import load_cfg, raw_table, sql

st.set_page_config(page_title="Inventory Intelligence", layout="wide")
st.title("📦 Inventory Intelligence")
cfg = load_cfg()

cars = raw_table("cars")
c1, c2 = st.columns(2)
city = c1.selectbox("City", ["All"] + sorted(cars["city"].unique()))
body = c2.selectbox("Body type", ["All"] + sorted(cars["body_type"].unique()))

aging = sql("inventory_aging.sql")
if city != "All":
    aging = aging[aging["city"] == city]
if body != "All":
    aging = aging[aging["body_type"] == body]
order = ["0-15", "16-30", "31-45", "46-60", "61-90", "90+"]
agg = aging.groupby("aging_bucket", as_index=False)["n_cars"].sum()
st.plotly_chart(px.bar(agg, x="aging_bucket", y="n_cars",
                       category_orders={"aging_bucket": order},
                       title="Inventory by aging bucket"), use_container_width=True)

st.subheader("Demand–supply gaps (acquisition opportunities)")
gap = sql("demand_supply_gap.sql")
if city != "All":
    gap = gap[gap["city"] == city]
st.dataframe(gap.head(20), use_container_width=True)

st.subheader("Price-review candidates (high engagement, zero bookings)")
from driveintent.analytics.analytics import price_review_candidates

st.dataframe(price_review_candidates(cfg), use_container_width=True)

st.subheader("Sell-through probability distribution (test window)")
import pandas as pd

from driveintent.config import load_config

p = load_config().artifacts / "metrics" / "sellthrough_test_predictions.parquet"
if p.exists():
    preds = pd.read_parquet(p)
    st.plotly_chart(px.histogram(preds, x="p_cal", nbins=30,
                                 title="Calibrated P(sale within 30 days)"),
                    use_container_width=True)
