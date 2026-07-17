"""Marketing Intelligence: campaign quality + budget simulator."""
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from driveintent.dashboard.state import load_cfg

st.set_page_config(page_title="Marketing Intelligence", layout="wide")
st.title("📣 Marketing Intelligence")
cfg = load_cfg()

from driveintent.analytics.analytics import budget_optimizer, campaign_performance, inventory_match_score

perf = campaign_performance(cfg)
st.subheader("Campaign lead quality")
st.dataframe(perf[["campaign_name", "channel", "sessions", "qualified_sessions",
                   "leads", "bookings", "conversion_rate", "cost_per_lead",
                   "expected_lead_value", "expected_roas", "actual_roas"]],
             use_container_width=True)

c1, c2 = st.columns(2)
c1.plotly_chart(px.scatter(perf, x="sessions", y="conversion_rate", size="leads",
                           color="channel", hover_name="campaign_name",
                           title="Volume vs quality: traffic ≠ conversions"),
                use_container_width=True)
inv = inventory_match_score(cfg).merge(perf[["campaign_id", "campaign_name"]],
                                       on="campaign_id", how="left")
c2.plotly_chart(px.bar(inv, x="campaign_name", y="satisfiable_share",
                       title="Inventory-match score (share of demanded segments in stock)"),
                use_container_width=True)

st.subheader("Budget optimization simulation")
st.caption("Scenario simulation on synthetic diminishing-return curves — "
           "not a production bidding system.")
total = st.number_input("Total budget (₹, 0 = keep current total)", 0, 10_000_000, 0, 50_000)
alloc = budget_optimizer(cfg, total_budget=total or None)
if len(alloc):
    st.dataframe(alloc, use_container_width=True)
    fig = go.Figure(data=[
        go.Bar(name="Current", x=alloc["campaign"], y=alloc["current_budget"]),
        go.Bar(name="Recommended", x=alloc["campaign"], y=alloc["recommended_budget"])])
    fig.update_layout(barmode="group", title="Before vs after allocation")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("A/B test design")
from driveintent.analytics.analytics import sample_size_two_proportions, simulate_experiment

c1, c2, c3 = st.columns(3)
base = c1.number_input("Baseline booking rate", 0.001, 0.5, 0.05, 0.005, format="%.3f")
mde = c2.number_input("Minimum detectable relative lift", 0.01, 1.0, 0.10, 0.01)
power = c3.selectbox("Power", [0.8, 0.9])
st.metric("Required users per arm",
          f"{sample_size_two_proportions(base, mde, power=power):,}")
if st.button("Simulate experiment"):
    st.json(simulate_experiment(cfg, n_per_arm=5000, true_lift=mde, baseline=base))
