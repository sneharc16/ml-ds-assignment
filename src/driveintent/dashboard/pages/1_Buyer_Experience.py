"""Buyer Experience: intent inference + personalized recommendations."""
import plotly.express as px
import streamlit as st

from driveintent.dashboard.state import load_cfg, raw_table

st.set_page_config(page_title="Buyer Experience", layout="wide")
st.title("🧭 Buyer Experience")
cfg = load_cfg()

events = raw_table("events")
users = raw_table("users")
active = events["user_id"].value_counts()
uid = st.selectbox("User", active.head(200).index.tolist())
user_events = events[events["user_id"] == uid]
sessions = user_events.sort_values("event_timestamp")["session_id"].unique().tolist()
sid = st.selectbox("Session", sessions, index=len(sessions) - 1)
c1, c2 = st.columns(2)
limit = c1.slider("Recommendations", 3, 15, 8)
div = c2.selectbox("Diversity level", ["auto", "low", "medium", "high"])

from driveintent.features.intent import infer_profile

cars = raw_table("cars")
prof = infer_profile(user_events, cars, session_id=sid)

st.subheader("Inferred intent")
m1, m2, m3 = st.columns(3)
m1.metric("Preference confidence", f"{prof.confidence:.2f}")
m2.metric("Session entropy", f"{prof.session_entropy:.2f}",
          help="Near 0 = concentrated intent; near 1 = exploratory")
m3.metric("Hard constraints", len(prof.hard_constraints))

cc1, cc2 = st.columns(2)
lt = prof.long_term.get("body_type", {})
ss = prof.session.get("body_type", {})
if lt:
    cc1.plotly_chart(px.bar(x=list(lt.keys()), y=list(lt.values()),
                            title="Historical body-type preference",
                            labels={"x": "", "y": "weight"}), use_container_width=True)
if ss:
    cc2.plotly_chart(px.bar(x=list(ss.keys()), y=list(ss.values()),
                            title="Current-session intent",
                            labels={"x": "", "y": "weight"}), use_container_width=True)
with st.expander("Hard constraints & soft preferences"):
    st.json({"hard_constraints": prof.hard_constraints,
             "soft_preferences": prof.soft_preferences,
             "entropy_by_dimension": prof.entropy_by_dim})

st.subheader("Personalized recommendations")
if st.button("Generate recommendations", type="primary"):
    from driveintent.models.ranking import recommend_for_user
    with st.spinner("Scoring candidates..."):
        out = recommend_for_user(cfg, uid, session_id=sid, limit=limit,
                                 diversity_level=None if div == "auto" else div)
    for r in out["recommendations"]:
        with st.container(border=True):
            a, b, c = st.columns([3, 2, 3])
            a.markdown(f"**#{r['rank']} {r['make']} {r['model']}** · {r['body_type']} · "
                       f"{r['transmission']} · {r['city']}")
            a.caption(f"source: {r['candidate_source']}")
            b.metric("Listed", f"₹{r['listed_price']:,.0f}",
                     delta=f"{r['deal_score']*100:+.1f}% vs fair value")
            c.markdown("  \n".join(f"🏷️ {x}" for x in r["reasons"]))
            c.caption(f"P(booking)={r['booking_probability']:.3f} · "
                      f"inspection {r['inspection_score']:.0f}/100")
