import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from mmm.model_loader import CHANNELS, SPEND_COLUMNS, load_model_artifacts
from mmm.prediction import predict_scenario
from mmm.ui import PALETTE, clean_figure, money, scenario_picker, setup_page

setup_page("Media Playground", "🎛️", "Move a media input and watch the trained posterior respond. These are legitimate scenario inputs—not changes to fitted parameters.")
artifacts = load_model_artifacts()
scenario, row = scenario_picker(artifacts, "media")
current = scenario.copy()
st.subheader("Shape this week’s media mix")
cols = st.columns(3)
for i, (channel, column) in enumerate(zip(CHANNELS, SPEND_COLUMNS)):
    historical_max = float(artifacts.data[column].max())
    scenario.loc[row, column] = cols[i].slider(
        channel,
        0.0,
        historical_max * 1.5,
        float(current.loc[row, column]),
        step=max(100.0, historical_max / 250),
        format="$%.0f",
    )

ids = np.random.default_rng(17).choice(artifacts.n_samples, 500, replace=False)
base = predict_scenario(current, artifacts, draw_indices=ids, seed=17)
user = predict_scenario(scenario, artifacts, draw_indices=ids, seed=17)
i = row
delta = user["expected_samples"][:, i] - base["expected_samples"][:, i]
metrics = st.columns(4)
metrics[0].metric("Expected revenue", money(user["median"][i]))
metrics[1].metric("90% predictive interval", f"{money(user['lower'][i])} – {money(user['upper'][i])}")
metrics[2].metric("Change vs current", money(np.median(delta)), f"{np.median(delta) / max(abs(base['median'][i]), 1):+.1%}")
media_increment = user["channel_contribution_samples"][:, i, :].sum(axis=1)
metrics[3].metric("Media contribution", money(np.median(media_increment)))

chart = pd.DataFrame({
    "Channel": CHANNELS,
    "Current": base["channel_contributions"][i],
    "Your scenario": user["channel_contributions"][i],
}).melt("Channel", var_name="Scenario", value_name="Contribution")
fig = px.bar(chart, x="Channel", y="Contribution", color="Scenario", barmode="group", color_discrete_sequence=["#B9BCD0", PALETTE[0]], title="Posterior median channel contribution")
fig.update_yaxes(tickprefix="$", tickformat="~s")
st.plotly_chart(clean_figure(fig), use_container_width=True)
with st.expander("💡 What changed?"):
    st.write("The model rebuilt the selected show-season’s lag history, applied each posterior draw’s fitted carryover and saturation, and compared the same draws in both scenarios. Keeping draws paired makes the difference easier to interpret.")
