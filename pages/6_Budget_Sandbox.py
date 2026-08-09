import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from mmm.cache import historical_row_prediction
from mmm.contributions import marginal_roas
from mmm.model_loader import CHANNELS, SPEND_COLUMNS, load_model_artifacts
from mmm.optimization import normalize_allocation, suggest_allocation
from mmm.prediction import predict_scenario_row
from mmm.ui import PALETTE, clean_figure, money, scenario_picker, setup_page

setup_page(
    "Budget Sandbox",
    "💰",
    "Rearrange a fixed weekly budget across channels. Recommendations reflect fitted response curves and uncertainty—not deterministic truth.",
)
artifacts = load_model_artifacts()
scenario, row = scenario_picker(artifacts, "budget")
current = scenario.loc[row, SPEND_COLUMNS].to_numpy(float)
default_budget = float(current.sum())
show = str(scenario.loc[row, "Show"])
season = int(scenario.loc[row, "Season"])
allocation_key = (show, season, int(row))
widget_suffix = f"{show}_{season}_{row}"

st.subheader("Experimental allocation")
with st.form("budget_allocation_form"):
    budget_input = st.number_input(
        "Total weekly media budget",
        min_value=1_000.0,
        value=max(1_000.0, default_budget),
        step=10_000.0,
        format="%.0f",
        key=f"budget_total_{widget_suffix}",
    )
    cols = st.columns(3)
    weights = np.array(
        [
            cols[i].slider(
                f"{CHANNELS[i]} share",
                0,
                100,
                int(round(current[i] / max(current.sum(), 1) * 100)),
                key=f"share_{i}_{widget_suffix}",
            )
            for i in range(3)
        ],
        float,
    )
    apply_allocation = st.form_submit_button("Apply allocation", type="primary")

if apply_allocation or st.session_state.get("budget_allocation_key") != allocation_key:
    st.session_state.budget_allocation = normalize_allocation(weights, budget_input)
    st.session_state.budget_total = float(budget_input)
    st.session_state.budget_allocation_key = allocation_key
allocation = np.asarray(st.session_state.budget_allocation, float)
budget = float(st.session_state.budget_total)
st.caption(
    "Shares are normalized automatically so the dollars always add to the total budget. Apply once after adjusting the controls."
)

if st.button("✨ Find a suggested allocation"):
    with st.spinner("Comparing plausible posterior response curves…"):
        st.session_state.suggested = suggest_allocation(
            scenario, row, budget, artifacts
        )
        st.session_state.suggested_key = (allocation_key, budget)
if st.session_state.get("suggested_key") == (allocation_key, budget):
    suggested = normalize_allocation(st.session_state.suggested, budget)
    st.info(
        "Suggested allocation: "
        + " · ".join(
            f"{name} {money(value)}" for name, value in zip(CHANNELS, suggested)
        )
    )

trial = scenario.copy()
trial.loc[row, SPEND_COLUMNS] = allocation
base_result = historical_row_prediction(
    show, season, row, 500, 77, artifacts.fingerprint, artifacts
)
ids = base_result["draw_indices"]
trial_result = predict_scenario_row(
    trial, row, artifacts, draw_indices=ids, seed=77
)
delta = trial_result["expected_samples"] - base_result["expected_samples"]
lo, hi = np.quantile(trial_result["samples"], [.05, .95])
m = st.columns(4)
m[0].metric(
    "Expected response", money(np.median(trial_result["expected_samples"]))
)
m[1].metric("Change vs current", money(np.median(delta)))
m[2].metric("90% predictive interval", f"{money(lo)} – {money(hi)}")
m[3].metric("Total allocated", money(allocation.sum()))

chart = pd.DataFrame(
    {"Channel": CHANNELS, "Current": current, "Experimental": allocation}
).melt("Channel", var_name="Mix", value_name="Spend")
fig = px.bar(
    chart,
    x="Channel",
    y="Spend",
    color="Mix",
    barmode="group",
    color_discrete_sequence=["#B9BCD0", PALETTE[0]],
    title="Current vs experimental allocation",
)
fig.update_yaxes(tickprefix="$", tickformat="~s")
st.plotly_chart(clean_figure(fig), use_container_width=True)

st.subheader("Marginal ROAS")
mroas_key = (allocation_key, float(budget), tuple(float(v) for v in allocation))
if st.button("Calculate mROAS"):
    bump = 10_000.0
    rows = []
    for name, column in zip(CHANNELS, SPEND_COLUMNS):
        bumped = trial.copy()
        bumped.loc[row, column] += bump
        bumped_result = predict_scenario_row(
            bumped, row, artifacts, draw_indices=ids, include_noise=False
        )
        mroas = marginal_roas(trial_result, bumped_result, bump)
        rows.append(
            {
                "Channel": name,
                "Median mROAS": np.median(mroas),
                "P05": np.quantile(mroas, .05),
                "P95": np.quantile(mroas, .95),
            }
        )
    st.session_state.mroas_result = pd.DataFrame(rows)
    st.session_state.mroas_key = mroas_key
if st.session_state.get("mroas_key") == mroas_key:
    st.dataframe(
        st.session_state.mroas_result.style.format(
            {"Median mROAS": "{:.3f}", "P05": "{:.3f}", "P95": "{:.3f}"}
        ),
        hide_index=True,
        use_container_width=True,
    )
    st.write(
        "Marginal ROAS estimates expected incremental revenue from the next unit of spending, rather than average return across all historical spend."
    )
else:
    st.caption("Calculate this on demand after applying an allocation.")

st.warning(
    "Suggested allocations depend on the fitted response functions, posterior uncertainty, historical support, and the chosen constraints. Treat them as a scenario to investigate, not a guaranteed optimum."
)
