"""Model Monitoring: metrics, calibration, lift, ranking, data health."""
import pandas as pd
import plotly.express as px
import streamlit as st

from driveintent.dashboard.state import load_cfg, metrics_csv, metrics_json, raw_table

st.set_page_config(page_title="Model Monitoring", layout="wide")
st.title("📈 Model Monitoring")
cfg = load_cfg()

tab_r, tab_c, tab_k, tab_d = st.tabs(
    ["Regression", "Classification", "Recommendations", "Data health"])

with tab_r:
    m = metrics_json("price_metrics.json")
    if m:
        cb = m["catboost"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("MAE", f"₹{cb['mae']:,.0f}")
        c2.metric("RMSE", f"₹{cb['rmse']:,.0f}")
        c3.metric("R²", f"{cb['r2']:.3f}")
        c4.metric("P10–P90 coverage", f"{cb['p10_p90_coverage']:.1%}")
        st.caption(f"vs make-model median baseline MAE "
                   f"₹{m['baseline_make_model_median']['mae']:,.0f}")
        p = cfg.artifacts / "metrics" / "price_test_predictions.parquet"
        if p.exists():
            d = pd.read_parquet(p)
            c1, c2 = st.columns(2)
            c1.plotly_chart(px.scatter(d, x="target_price", y="predicted",
                                       title="Predicted vs actual", opacity=0.5),
                            use_container_width=True)
            d["residual"] = d["target_price"] - d["predicted"]
            c2.plotly_chart(px.histogram(d, x="residual", nbins=40,
                                         title="Residual distribution"),
                            use_container_width=True)

with tab_c:
    which = st.selectbox("Model", ["booking", "sellthrough"])
    m = metrics_json(f"{which}_metrics.json")
    if m:
        cb = m["catboost_calibrated"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("PR-AUC", f"{cb['pr_auc']:.3f}")
        c2.metric("ROC-AUC", f"{cb['roc_auc']:.3f}")
        c3.metric("Brier", f"{cb['brier']:.4f}")
        c4.metric("Lift @ top 10%", f"{cb['lift_top10pct']:.2f}x")
        st.caption(f"Calibration: {m['calibration_method']} · "
                   f"threshold {m['selected_threshold']['threshold']} "
                   f"({m['selected_threshold']['basis']})")
        rel = metrics_csv(f"{which}_reliability.csv")
        lift = metrics_csv(f"{which}_lift_table.csv")
        tt = metrics_csv(f"{which}_threshold_table.csv")
        c1, c2 = st.columns(2)
        if len(rel):
            fig = px.line(rel, x="mean_predicted", y="observed_rate", markers=True,
                          title="Reliability diagram")
            fig.add_shape(type="line", x0=0, y0=0, x1=rel["mean_predicted"].max(),
                          y1=rel["mean_predicted"].max(), line=dict(dash="dash"))
            c1.plotly_chart(fig, use_container_width=True)
        if len(lift):
            c2.plotly_chart(px.bar(lift, x="decile", y="lift", title="Lift by decile"),
                            use_container_width=True)
        if len(tt):
            st.dataframe(tt, use_container_width=True)

with tab_k:
    m = metrics_json("ranking_metrics.json")
    if m:
        rows = [{"model": k, **{kk: round(vv, 4) for kk, vv in v.items()
                                if isinstance(vv, (int, float))}}
                for k, v in m.items() if isinstance(v, dict)]
        df = pd.DataFrame(rows)
        st.dataframe(df[["model", "ndcg_at_5", "ndcg_at_10", "map_at_10",
                         "recall_at_10", "mrr", "coverage_at_10",
                         "brand_concentration"]], use_container_width=True)
        st.plotly_chart(px.bar(df, x="model", y="ndcg_at_10",
                               title="NDCG@10 by ranking strategy"),
                        use_container_width=True)
    ab = cfg.artifacts / "reports" / "ablation_results.csv"
    if ab.exists():
        st.subheader("Ablation study")
        st.dataframe(pd.read_csv(ab), use_container_width=True)

with tab_d:
    for t in ("cars", "users", "sessions", "events"):
        df = raw_table(t)
        with st.expander(f"{t} — {len(df):,} rows"):
            miss = df.isna().mean().sort_values(ascending=False).head(8)
            st.write("Missing-value share (top 8):")
            st.dataframe(miss.rename("missing_share").to_frame())
    ev = raw_table("events")
    ev["event_date"] = pd.to_datetime(ev["event_date"])
    daily = ev.groupby(ev["event_date"].dt.to_period("M")).size().rename("events")
    st.plotly_chart(px.bar(x=daily.index.astype(str), y=daily.values,
                           title="Event volume by month",
                           labels={"x": "month", "y": "events"}),
                    use_container_width=True)
