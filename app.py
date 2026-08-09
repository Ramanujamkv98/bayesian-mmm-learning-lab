import streamlit as st

from mmm.model_loader import load_model_artifacts
from mmm.ui import setup_page

setup_page(
    "Make the model move",
    "🪄",
    "Explore a trained Bayesian marketing mix model one idea at a time. Change a spend, follow its carryover, and see uncertainty instead of hiding it behind one number.",
)

artifacts = load_model_artifacts()

left, right = st.columns([1.25, 0.75], gap="large")
with left:
    st.header("A learning lab, not a black box")
    st.write(
        "This app uses the existing 4,000 posterior draws from the trained model. Sliders recalculate predictions with NumPy; they never rerun MCMC. Pick a lab from the sidebar and move one assumption at a time."
    )
    st.subheader("Start with a question")
    st.page_link("pages/1_Media_Playground.py", label="What if I change this week’s media spend?", icon="🎛️")
    st.page_link("pages/2_Adstock_Lab.py", label="How long can an ad keep working?", icon="🌊")
    st.page_link("pages/3_Saturation_Lab.py", label="Where do diminishing returns appear?", icon="🌱")
    st.page_link("pages/4_Uncertainty_Lab.py", label="Why doesn’t Bayesian prediction give one answer?", icon="🎲")
    st.page_link("pages/5_Channel_Contributions.py", label="What built the prediction?", icon="🧩")
    st.page_link("pages/6_Budget_Sandbox.py", label="How might I rearrange a budget?", icon="💰")

with right:
    st.subheader("The fitted model")
    st.metric("Posterior worlds", f"{artifacts.n_samples:,}")
    st.metric("Historical weeks", f"{len(artifacts.data):,}")
    st.metric("Show-seasons", len(artifacts.coords["show_season"]))
    st.caption("3 media channels · 8-week maximum lag · show-level media response · show-season baselines")

st.divider()
st.subheader("The idea in one line")
st.code("Revenue = show-season baseline + controls + saturated(adstocked media) × show response + noise", language=None)
with st.expander("What’s happening mathematically?"):
    st.markdown(
        "Spend is divided by its training-set 95th percentile, combined across the current and previous eight weeks with normalized geometric weights, and passed through `x / (x + k)`. Each posterior draw supplies its own retention, half-saturation, show-level channel coefficient, baseline, control effects, and residual noise."
    )

st.caption("Scenario predictions are model-based estimates, not guaranteed causal outcomes.")
