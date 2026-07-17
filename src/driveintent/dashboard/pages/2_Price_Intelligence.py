"""Price Intelligence: fair value with uncertainty + SHAP."""
import plotly.graph_objects as go
import streamlit as st

from driveintent.dashboard.state import load_cfg, raw_table

st.set_page_config(page_title="Price Intelligence", layout="wide")
st.title("💰 Price Intelligence")
cfg = load_cfg()
cars = raw_table("cars")

mode = st.radio("Mode", ["Existing car", "Hypothetical car"], horizontal=True)
if mode == "Existing car":
    cid = st.selectbox("Car", cars["car_id"].head(500))
    row = cars[cars["car_id"] == cid].iloc[0].to_dict()
else:
    c1, c2, c3 = st.columns(3)
    row = dict(make=c1.selectbox("Make", sorted(cars["make"].unique())),
               body_type=c2.selectbox("Body", sorted(cars["body_type"].unique())),
               city=c3.selectbox("City", sorted(cars["city"].unique())),
               fuel_type="Petrol", transmission="Automatic", variant="Mid",
               state="", engine_cc=1300, claimed_mileage_kmpl=18,
               insurance_valid=True, service_history_available=True,
               accident_history=False, exterior_score=80, interior_score=80,
               engine_score=80, tyre_score=75, number_of_features=10)
    sub = cars[(cars["make"] == row["make"]) & (cars["body_type"] == row["body_type"])]
    row["model"] = sub["model"].mode().iloc[0] if len(sub) else cars["model"].iloc[0]
c1, c2, c3, c4 = st.columns(4)
row["kilometres_driven"] = c1.slider("Kilometres", 1000, 200000,
                                     int(row.get("kilometres_driven", 40000)), 1000)
row["manufacturing_year"] = c2.slider("Manufacturing year", 2012, 2025,
                                      int(row.get("manufacturing_year", 2021)))
row["inspection_score"] = c3.slider("Inspection score", 40, 99,
                                    int(row.get("inspection_score", 82)))
listed = c4.number_input("Listed price (₹)", 100000, 5000000,
                         int(row.get("listed_price", 800000)), 10000)
row["registration_year"] = row["manufacturing_year"]
row["vehicle_age"] = max(2026 - row["manufacturing_year"], 0)
row["kilometres_per_year"] = row["kilometres_driven"] / max(row["vehicle_age"], 1)
row["owner_count"] = int(row.get("owner_count", 1))
row.setdefault("inventory_entry_month", 7)
row.setdefault("market_demand_index", 1.0)
row.setdefault("local_supply_index", 0.2)
row.setdefault("model_popularity", 0.75)

from driveintent.models import price as price_model

pred = price_model.predict(cfg, row)
fair, lo, hi = pred["predicted_fair_price"], pred["lower_price"], pred["upper_price"]
deal = (fair - listed) / fair
label = ("🟢 great deal" if deal > 0.07 else "🟢 good deal" if deal > 0.02
         else "🟡 fair price" if deal > -0.05 else "🔴 overpriced")
m1, m2, m3 = st.columns(3)
m1.metric("Fair value (P50)", f"₹{fair:,.0f}")
m2.metric("P10 – P90 range", f"₹{lo:,.0f} – ₹{hi:,.0f}")
m3.metric("Deal score", f"{deal*100:+.1f}%", label)

fig = go.Figure()
fig.add_trace(go.Scatter(x=[lo, hi], y=[0, 0], mode="lines",
                         line=dict(width=14, color="#cbd5e1"), name="P10–P90"))
fig.add_trace(go.Scatter(x=[fair], y=[0], mode="markers",
                         marker=dict(size=18, color="#2563eb"), name="Fair (P50)"))
fig.add_trace(go.Scatter(x=[listed], y=[0], mode="markers",
                         marker=dict(size=18, color="#dc2626", symbol="diamond"),
                         name="Listed"))
fig.update_layout(height=180, yaxis=dict(visible=False), title="Price interval")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Why this price? (SHAP)")
for f in price_model.explain(cfg, row, top_k=6):
    arrow = "⬆️" if f["direction"] == "increases_price" else "⬇️"
    st.progress(min(float(f["importance"]), 1.0),
                text=f"{arrow} {f['feature']} ({f['importance']:.0%} of |impact|)")

st.subheader("Comparable segment")
comp = cars[(cars["make"] == row["make"]) & (cars["model"] == row["model"])]
if len(comp) >= 3:
    st.dataframe(comp[["car_id", "city", "manufacturing_year", "kilometres_driven",
                       "inspection_score", "listed_price"]].head(10),
                 use_container_width=True)
