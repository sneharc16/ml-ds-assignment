"""DriveIntent Streamlit dashboard entrypoint."""
import streamlit as st

from driveintent.dashboard.state import load_cfg

st.set_page_config(page_title="DriveIntent", page_icon="🚗", layout="wide")
cfg = load_cfg()

st.title("🚗 DriveIntent")
st.subheader("Intent-aware used-car marketplace intelligence platform")
st.markdown(
    """
Use the pages in the sidebar:

1. **Buyer Experience** – intent inference & personalized recommendations
2. **Price Intelligence** – fair-price estimation with uncertainty & SHAP
3. **Inventory Intelligence** – aging, sell-through, demand-supply gaps
4. **Marketing Intelligence** – campaign lead quality & budget simulation
5. **Model Monitoring** – regression / classification / ranking metrics

> All data are **synthetic**. This platform demonstrates methodology, not
> real CARS24 results.
"""
)
c1, c2, c3, c4 = st.columns(4)
try:
    from driveintent.dashboard.state import table_counts
    counts = table_counts(cfg)
    c1.metric("Cars", f"{counts['cars']:,}")
    c2.metric("Users", f"{counts['users']:,}")
    c3.metric("Sessions", f"{counts['sessions']:,}")
    c4.metric("Events", f"{counts['events']:,}")
except Exception as e:
    st.warning(f"Run the pipeline first: `make pipeline` ({e})")
