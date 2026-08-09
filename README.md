# Bayesian MMM Learning Lab

An interactive, educational Streamlit application powered by the existing trained hierarchical Bayesian marketing mix model. It turns posterior samples into fast what-if predictions so learners can see carryover, diminishing returns, uncertainty, contributions, and budget tradeoffs without rerunning MCMC.

## Why this exists

Traditional MMM dashboards often present a coefficient or ROI without showing how the model arrived there. This lab teaches through interaction: move one input, keep posterior draws paired, and watch the response distribution change.

## What the existing model does

The supplied Model 2 was trained on 420 weekly observations from eight shows and 28 show-seasons. Earlier complete seasons form the 300-row training set; the latest season of every show forms a 120-row test set.

For each posterior draw, expected standardized revenue is:

```text
show-season intercept
+ scaled control effects
+ Σ(show-specific channel coefficient × saturation(normalized adstock(spend)))
```

- Channels: Network TV, Cable TV, Digital
- Controls: holiday, lead-in bonus, lifecycle week, and indicators for finale, pre-launch, and premiere (regular is the reference)
- Carryover: normalized geometric adstock over the current week plus eight lags
- Saturation: `x / (x + half_saturation)`; the Hill slope is fixed at 1
- Hierarchy: partially pooled show-specific media coefficients and show-season intercepts
- Likelihood: Normal residual variation on standardized revenue

The saved model is an ArviZ-compatible NetCDF `InferenceData` with posterior, sample statistics, constant data, and observed data groups. It contains four chains and 1,000 draws per chain.

## How posterior predictions power the app

`mmm/prediction.py` is the single source of truth. It reconstructs lag histories, applies the exact fitted transforms, evaluates selected posterior draws with vectorized NumPy, decomposes expected revenue additively, and optionally adds residual noise for posterior predictive intervals. Streamlit caches the loaded artifact once; moving a slider does not invoke PyMC or MCMC.

Manual adstock or half-saturation changes are clearly marked **Sensitivity / What-if mode** because they leave the fitted posterior. Spend, show, season, lifecycle week, and budget are scenario inputs and therefore remain posterior predictions from the trained model.

## Architecture

```text
mmm-learning-lab/
├── app.py
├── pages/
│   ├── 1_Media_Playground.py
│   ├── 2_Adstock_Lab.py
│   ├── 3_Saturation_Lab.py
│   ├── 4_Uncertainty_Lab.py
│   ├── 5_Channel_Contributions.py
│   └── 6_Budget_Sandbox.py
├── mmm/
│   ├── model_loader.py
│   ├── transforms.py
│   ├── prediction.py
│   ├── contributions.py
│   ├── uncertainty.py
│   ├── optimization.py
│   └── ui.py
├── data/
├── model/
├── tests/
├── requirements.txt
└── README.md
```

## Run locally

Use Python 3.11 or 3.12.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Run the validation suite with `pytest -q`. The central regression test reconstructs every saved `mu_all` value across all 4,000 draws and 420 observations, then requires numerical equality with the notebook output.

## Performance and profiling

The posterior is loaded once with `st.cache_resource`. Deterministic draw selections, historical predictions, contribution inputs, and fitted saturation curves use `st.cache_data`. Pages that display one selected week use an exact row evaluator, while the contribution page retains the full-sequence engine. Budget controls submit as a form, optimization evaluates only its selected-week objective, and mROAS runs on demand.

Run the repeatable benchmark with:

```bash
python benchmarks/performance.py
```

Set `MMM_PROFILE=1` before starting Streamlit to log model-load, scenario-prediction, and budget-optimization timings during development. Profiling is silent by default in deployed environments.

## Deploy to Streamlit Community Cloud

1. Put this folder in a GitHub repository. Keep `data/gold_model_dataset.csv` and `model/hierarchical_mmm_model_2.nc` at their current relative paths.
2. In Streamlit Community Cloud, create an app from the repository and choose `app.py` as the entry point.
3. Deploy. No secrets, local absolute paths, GPU, or notebook state are required.

The model file is about 17.6 MB and fits under GitHub’s normal per-file limit. Git LFS is unnecessary at its current size. Community Cloud will install only the packages in `requirements.txt`.

## Screenshots

Add screenshots here after deployment:

- Home / learning path
- Media Playground
- Adstock and Saturation Labs
- Bayesian Uncertainty Lab
- Budget Sandbox

## Limitations

- The trained model supports only known shows and show-seasons; it is not a cold-start model.
- Scenario controls are anchored to observed historical rows in this first version.
- Budget optimization uses the fitted expected response across a sample of posterior draws and a fixed total weekly budget. It does not encode operational minimums, maximums, flighting rules, or costs beyond the selected constraint.
- Contribution is a decomposition of the model prediction, not experimental attribution.
- Spend far outside the historical range is extrapolation and should be treated cautiously.
- The source model’s saturation slope is fixed at 1, so the app does not invent a fitted slope parameter.

## Disclaimer

Scenario predictions and suggested allocations are model-based estimates, not guaranteed outcomes or proof of causality. Decisions should consider posterior uncertainty, data quality, historical support, business constraints, and external evidence.
