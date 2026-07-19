"""Advanced SQL portfolio: cohorts, attribution, funnel timing and survival."""
import pandas as pd
import plotly.express as px
import streamlit as st

from driveintent.dashboard.state import load_cfg, sql

st.set_page_config(page_title="SQL Analytics", layout="wide")
st.title("🧮 SQL Analytics Lab")
st.caption("DuckDB marts, window functions, cohorts, multi-touch attribution and quality checks")
cfg = load_cfg()

queries = {
    "Cohort retention": "cohort_retention.sql",
    "Campaign attribution": "campaign_attribution.sql",
    "Funnel transition times": "funnel_transition_times.sql",
    "Inventory survival": "inventory_survival.sql",
    "Customer 360": "customer_360.sql",
    "Recommendation IPS": "recommendation_ips.sql",
    "Demand-supply gap": "demand_supply_gap.sql",
}
label = st.selectbox("Analysis", list(queries))
frame = sql(queries[label])
c1, c2, c3 = st.columns(3)
c1.metric("Rows returned", f"{len(frame):,}")
c2.metric("Columns", len(frame.columns))
c3.metric("SQL engine", "DuckDB")

if label == "Cohort retention" and len(frame):
    pivot = frame.pivot(index="cohort_month", columns="months_since_signup", values="retention_rate")
    st.plotly_chart(px.imshow(pivot, aspect="auto", color_continuous_scale="Blues",
                             title="Monthly cohort retention"), use_container_width=True)
elif label == "Campaign attribution" and len(frame):
    st.plotly_chart(px.bar(frame.head(20), x="campaign_name", y=["first_touch_margin",
                             "last_touch_margin", "linear_attributed_margin"],
                             barmode="group", title="Attribution model comparison"),
                    use_container_width=True)
elif label == "Inventory survival" and len(frame):
    st.plotly_chart(px.line(frame, x="interval_start", y="survival_probability",
                            color="body_type", markers=True,
                            title="Inventory survival probability"), use_container_width=True)
elif label == "Recommendation IPS" and len(frame):
    st.plotly_chart(px.bar(frame, x="body_type", y=["naive_ctr", "ips_ctr"],
                           barmode="group", title="Naive vs propensity-adjusted CTR"),
                    use_container_width=True)

st.dataframe(frame, use_container_width=True)

quality_path = cfg.artifacts / "reports" / "sql" / "data_quality_checks.csv"
if quality_path.exists():
    quality = pd.read_csv(quality_path)
    st.subheader("Warehouse data-quality checks")
    st.metric("Total violations", int(quality["violations"].sum()))
    st.dataframe(quality, use_container_width=True)
