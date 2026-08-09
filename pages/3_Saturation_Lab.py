import numpy as np
import plotly.graph_objects as go
import streamlit as st

from mmm.model_loader import CHANNELS, SPEND_COLUMNS, load_model_artifacts
from mmm.ui import PALETTE, clean_figure, money, sensitivity_banner, setup_page

setup_page("Saturation Lab", "🌱", "The first dollars can have a large effect. As spend rises, each additional dollar may produce less incremental response.")
artifacts = load_model_artifacts()
cols = st.columns(2)
channel = cols[0].selectbox("Channel", CHANNELS)
show = cols[1].selectbox("Show", artifacts.coords["show"])
c, s = CHANNELS.index(channel), artifacts.show_lookup[show]
half_draws = artifacts.posterior["half_saturation"][:, c]
beta_draws = artifacts.posterior["beta_show"][:, s, c]
fitted_half = float(np.median(half_draws))
key = f"half_{c}"
if key not in st.session_state:
    st.session_state[key] = fitted_half
half = st.slider("Half-saturation (scaled spend)", 0.05, 2.0, key=key, step=0.01)
if st.button("🔄 Return to fitted posterior"):
    st.session_state[key] = fitted_half
    st.rerun()
active = not np.isclose(half, fitted_half, atol=.005)
sensitivity_banner(active)

max_spend = float(artifacts.data[SPEND_COLUMNS[c]].max() * 1.5)
spend = np.linspace(0, max_spend, 180)
x = spend / artifacts.spend_scale[c]
curve_draws = artifacts.y_sd * beta_draws[:, None] * x[None, :] / (x[None, :] + (half if active else half_draws[:, None]))
median = np.median(curve_draws, axis=0)
lo, hi = np.quantile(curve_draws, [.05, .95], axis=0)
current = float(artifacts.data[SPEND_COLUMNS[c]].median())
current_y = float(np.interp(current, spend, median))

fig = go.Figure()
fig.add_trace(go.Scatter(x=spend, y=hi, line=dict(width=0), showlegend=False))
fig.add_trace(go.Scatter(x=spend, y=lo, fill="tonexty", fillcolor="rgba(25,181,165,.14)", line=dict(width=0), name="90% credible interval"))
fig.add_trace(go.Scatter(x=spend, y=median, line=dict(color=PALETTE[1], width=4), name="Fitted response"))
fig.add_trace(go.Scatter(x=[current], y=[current_y], mode="markers+text", text=["You are here"], textposition="top center", marker=dict(size=14, color=PALETTE[2]), name="Observed median"))
fig.update_layout(title=f"{show} · {channel} response curve", xaxis_title="Weekly spend", yaxis_title="Expected incremental revenue")
fig.update_xaxes(tickprefix="$", tickformat="~s"); fig.update_yaxes(tickprefix="$", tickformat="~s")
st.plotly_chart(clean_figure(fig), use_container_width=True)

step = 100_000.0
y_next = float(np.interp(current + step, spend, median))
m = st.columns(3)
m[0].metric("Current observed median", money(current))
m[1].metric("Half-response spend", money(half * artifacts.spend_scale[c]))
m[2].metric("Next $100K response", money(y_next - current_y))
st.caption("The notebook’s fitted Hill function has slope fixed at 1; there is no learned slope parameter to expose. The half-saturation control is therefore the faithful teaching control.")
