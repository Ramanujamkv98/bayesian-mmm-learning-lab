import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from mmm.model_loader import CHANNELS, load_model_artifacts
from mmm.prediction import predict_scenario
from mmm.ui import PALETTE, clean_figure, money, scenario_picker, setup_page

setup_page("Bayesian Uncertainty Lab", "🎲", "Each posterior draw is one plausible combination of model parameters supported by the data and assumptions.")
artifacts = load_model_artifacts()
scenario, row = scenario_picker(artifacts, "uncertainty")
level = st.select_slider("Interval level", options=[50, 80, 90, 95], value=90, format_func=lambda x: f"{x}%") / 100
if "world_draw" not in st.session_state:
    st.session_state.world_draw = 0
if st.button("🎲 Draw another plausible world", type="primary"):
    st.session_state.world_draw = int(np.random.default_rng().integers(artifacts.n_samples))

ids = np.random.default_rng(91).choice(artifacts.n_samples, 200, replace=False)
result = predict_scenario(scenario, artifacts, draw_indices=ids, interval=level, include_noise=True, seed=91)
selected_id = st.session_state.world_draw
world = predict_scenario(scenario, artifacts, draw_indices=np.array([selected_id]), include_noise=False)
samples = result["samples"][:, row]
plot = pd.DataFrame({"Posterior world": np.arange(len(samples)), "Predicted revenue": samples})
fig = px.scatter(plot, x="Posterior world", y="Predicted revenue", color_discrete_sequence=["rgba(91,95,239,.32)"], title="200 plausible posterior predictions")
fig.add_hline(y=np.median(samples), line_width=4, line_color=PALETTE[2], annotation_text="Median")
fig.update_yaxes(tickprefix="$", tickformat="~s")
st.plotly_chart(clean_figure(fig), use_container_width=True)
m = st.columns(4)
m[0].metric("Posterior median", money(np.median(samples)))
tail=(1-level)/2; lo,hi=np.quantile(samples,[tail,1-tail])
m[1].metric(f"{level:.0%} interval", f"{money(lo)} – {money(hi)}")
m[2].metric("Selected world", f"#{selected_id + 1:,}")
m[3].metric("Its prediction", money(world["expected_samples"][0, row]))

with st.expander("See this world’s important parameters"):
    table = pd.DataFrame({
        "Channel": CHANNELS,
        "Adstock α": artifacts.posterior["alpha"][selected_id],
        "Half-saturation": artifacts.posterior["half_saturation"][selected_id],
        "Show response": artifacts.posterior["beta_show"][selected_id, artifacts.show_lookup[scenario.loc[row, 'Show']]],
    })
    st.dataframe(table.style.format({"Adstock α":"{:.3f}","Half-saturation":"{:.3f}","Show response":"{:.3f}"}), hide_index=True, use_container_width=True)
st.info("MCMC is not fitting a separate model for every observation. It samples plausible shared parameter combinations, then each combination can predict the same scenario.")
