import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from mmm.model_loader import CHANNELS, load_model_artifacts
from mmm.ui import PALETTE, clean_figure, sensitivity_banner, setup_page

setup_page("Adstock Lab", "🌊", "Advertising can continue influencing outcomes after the week it runs. Adstock represents how quickly that effect fades.")
artifacts = load_model_artifacts()
channel = st.selectbox("Channel", CHANNELS)
c = CHANNELS.index(channel)
posterior = artifacts.posterior["alpha"][:, c]
median = float(np.median(posterior))
low, high = np.quantile(posterior, [0.05, 0.95])

if f"alpha_{c}" not in st.session_state:
    st.session_state[f"alpha_{c}"] = median
cols = st.columns([2, 1])
manual = cols[0].slider("Adstock retention α", 0.10, 0.95, key=f"alpha_{c}", step=0.01)
if cols[1].button("🔄 Use fitted value", use_container_width=True):
    st.session_state[f"alpha_{c}"] = median
    st.rerun()
active = not np.isclose(manual, median, atol=0.005)
sensitivity_banner(active)

weeks = np.arange(9)
curves = posterior[:, None] ** weeks[None, :]
fig = go.Figure()
fig.add_trace(go.Scatter(x=weeks, y=np.quantile(curves, .95, axis=0), mode="lines", line=dict(width=0), showlegend=False))
fig.add_trace(go.Scatter(x=weeks, y=np.quantile(curves, .05, axis=0), mode="lines", fill="tonexty", fillcolor="rgba(91,95,239,.15)", line=dict(width=0), name="90% fitted interval"))
fig.add_trace(go.Scatter(x=weeks, y=manual ** weeks, mode="lines+markers", line=dict(color=PALETTE[0], width=4), name="Manual curve" if active else "Fitted median"))
fig.update_layout(title=f"A one-unit {channel} impulse over time", xaxis_title="Weeks after exposure", yaxis_title="Remaining weight")
st.plotly_chart(clean_figure(fig), use_container_width=True)
m = st.columns(3)
m[0].metric("Fitted median α", f"{median:.2f}")
m[1].metric("90% credible interval", f"{low:.2f} – {high:.2f}")
m[2].metric("Weight after 4 weeks", f"{manual ** 4:.1%}")
with st.expander("What’s happening mathematically?"):
    st.write("Lag ℓ receives weight α^ℓ. The fitted model normalizes the available weights before averaging spend, so early weeks are not mechanically depressed by missing history. The displayed impulse isolates the decay weights for teaching.")
