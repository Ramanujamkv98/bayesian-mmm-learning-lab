import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from mmm.contributions import contribution_frame
from mmm.model_loader import load_model_artifacts
from mmm.prediction import predict_scenario
from mmm.ui import PALETTE, clean_figure, money, setup_page

setup_page("Channel Contributions", "🧩", "Decompose the prediction into its show-season baseline, control variables, and media channels.")
artifacts = load_model_artifacts(); data = artifacts.data
cols = st.columns(2)
show = cols[0].selectbox("Show", artifacts.coords["show"])
seasons = sorted(data.loc[data.Show == show, "Season"].unique())
season = cols[1].selectbox("Season", seasons, index=len(seasons)-1)
scenario = data[(data.Show == show) & (data.Season == season)].copy().reset_index(drop=True)
week_range = st.slider("Week range", int(scenario.Week_Number.min()), int(scenario.Week_Number.max()), (int(scenario.Week_Number.min()), int(scenario.Week_Number.max())))
mode = st.radio("View", ["Absolute contribution", "Percentage contribution"], horizontal=True)
result = predict_scenario(scenario, artifacts, draws=500, seed=33)
wide = contribution_frame(result, scenario)
wide = wide[(wide.Week >= week_range[0]) & (wide.Week <= week_range[1])]
components = [c for c in wide.columns if c != "Week"]

long = wide.melt("Week", var_name="Component", value_name="Contribution")
if mode.startswith("Percentage"):
    totals = long.groupby("Week")["Contribution"].transform(lambda x: np.abs(x).sum())
    long["Contribution"] = np.where(totals > 0, long["Contribution"] / totals * 100, 0)
fig = px.area(long, x="Week", y="Contribution", color="Component", color_discrete_sequence=PALETTE, title="How the prediction is assembled")
fig.update_yaxes(ticksuffix="%" if mode.startswith("Percentage") else "", tickprefix="" if mode.startswith("Percentage") else "$", tickformat=".0f" if mode.startswith("Percentage") else "~s")
st.plotly_chart(clean_figure(fig), use_container_width=True)

selected = np.flatnonzero((scenario.Week_Number >= week_range[0]) & (scenario.Week_Number <= week_range[1]))
pred = np.median(result["expected_samples"], axis=0)[selected]
actual = scenario.Revenue.to_numpy()[selected]
line = go.Figure()
line.add_trace(go.Scatter(x=scenario.Week_Number.to_numpy()[selected], y=actual, name="Actual", line=dict(color="#252742", width=3)))
line.add_trace(go.Scatter(x=scenario.Week_Number.to_numpy()[selected], y=pred, name="Predicted", line=dict(color=PALETTE[0], width=3, dash="dash")))
line.update_layout(title="Actual and posterior-median expected revenue", xaxis_title="Week", yaxis_title="Revenue"); line.update_yaxes(tickprefix="$", tickformat="~s")
st.plotly_chart(clean_figure(line, 330), use_container_width=True)

totals = wide[components].sum().sort_values(ascending=False)
cum = totals.cumsum().rename("Cumulative").reset_index(names="Component")
st.subheader("Selected-range totals")
m=st.columns(3); m[0].metric("Predicted", money(pred.sum())); m[1].metric("Actual", money(actual.sum())); m[2].metric("Difference", money(pred.sum()-actual.sum()))
st.dataframe(cum.style.format({"Cumulative": "${:,.0f}"}), hide_index=True, use_container_width=True)
st.caption("Contributions are an additive decomposition of the fitted prediction. They inherit posterior uncertainty but should not be read as experimentally proven causal effects.")
