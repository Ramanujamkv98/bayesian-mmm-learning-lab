import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from mmm.contributions import marginal_roas
from mmm.model_loader import CHANNELS, SPEND_COLUMNS, load_model_artifacts
from mmm.optimization import normalize_allocation, suggest_allocation
from mmm.prediction import predict_scenario
from mmm.ui import PALETTE, clean_figure, money, scenario_picker, setup_page

setup_page("Budget Sandbox", "💰", "Rearrange a fixed weekly budget across channels. Recommendations reflect fitted response curves and uncertainty—not deterministic truth.")
artifacts = load_model_artifacts(); scenario, row = scenario_picker(artifacts, "budget")
current = scenario.loc[row, SPEND_COLUMNS].to_numpy(float)
default_budget = float(current.sum())
budget = st.number_input("Total weekly media budget", min_value=1_000.0, value=max(1_000.0, default_budget), step=10_000.0, format="%.0f")

st.subheader("Experimental allocation")
cols = st.columns(3)
weights = np.array([cols[i].slider(f"{CHANNELS[i]} share", 0, 100, int(round(current[i] / max(current.sum(),1) * 100)), key=f"share_{i}") for i in range(3)], float)
allocation = normalize_allocation(weights, budget)
st.caption("Shares are normalized automatically so the dollars always add to the total budget.")

if st.button("✨ Find a suggested allocation", type="primary"):
    with st.spinner("Comparing plausible posterior response curves…"):
        st.session_state.suggested = suggest_allocation(scenario, row, budget, artifacts)
if "suggested" in st.session_state:
    suggested = normalize_allocation(st.session_state.suggested, budget)
    st.info("Suggested allocation: " + " · ".join(f"{name} {money(value)}" for name, value in zip(CHANNELS, suggested)))

trial = scenario.copy(); trial.loc[row, SPEND_COLUMNS] = allocation
ids = np.random.default_rng(77).choice(artifacts.n_samples, 500, replace=False)
base_result = predict_scenario(scenario, artifacts, draw_indices=ids, seed=77)
trial_result = predict_scenario(trial, artifacts, draw_indices=ids, seed=77)
delta = trial_result["expected_samples"][:, row] - base_result["expected_samples"][:, row]
tail=.05; lo,hi=np.quantile(trial_result["samples"][:,row],[tail,1-tail])
m=st.columns(4)
m[0].metric("Expected response", money(np.median(trial_result["expected_samples"][:,row])))
m[1].metric("Change vs current", money(np.median(delta)))
m[2].metric("90% predictive interval", f"{money(lo)} – {money(hi)}")
m[3].metric("Total allocated", money(allocation.sum()))

chart = pd.DataFrame({"Channel": CHANNELS, "Current": current, "Experimental": allocation}).melt("Channel", var_name="Mix", value_name="Spend")
fig=px.bar(chart,x="Channel",y="Spend",color="Mix",barmode="group",color_discrete_sequence=["#B9BCD0",PALETTE[0]],title="Current vs experimental allocation"); fig.update_yaxes(tickprefix="$",tickformat="~s")
st.plotly_chart(clean_figure(fig),use_container_width=True)

with st.expander("Marginal ROAS: what might the next dollar do?"):
    bump=10_000.0; rows=[]
    for i,(name,column) in enumerate(zip(CHANNELS,SPEND_COLUMNS)):
        bumped=trial.copy(); bumped.loc[row,column]+=bump
        bumped_result=predict_scenario(bumped,artifacts,draw_indices=ids,include_noise=False)
        mroas=marginal_roas(trial_result,bumped_result,bump)[:,row]
        rows.append({"Channel":name,"Median mROAS":np.median(mroas),"P05":np.quantile(mroas,.05),"P95":np.quantile(mroas,.95)})
    st.dataframe(pd.DataFrame(rows).style.format({"Median mROAS":"{:.3f}","P05":"{:.3f}","P95":"{:.3f}"}),hide_index=True,use_container_width=True)
    st.write("Marginal ROAS estimates expected incremental revenue from the next unit of spending, rather than average return across all historical spend.")

st.warning("Suggested allocations depend on the fitted response functions, posterior uncertainty, historical support, and the chosen constraints. Treat them as a scenario to investigate, not a guaranteed optimum.")
