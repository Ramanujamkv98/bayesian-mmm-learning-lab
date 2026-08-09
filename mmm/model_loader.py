from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "model" / "hierarchical_mmm_model_2.nc"
DATA_PATH = ROOT / "data" / "gold_model_dataset.csv"

SPEND_COLUMNS = ["Network_TV_Spend", "Cable_TV_Spend", "Digital_Spend"]
CHANNELS = ["Network TV", "Cable TV", "Digital"]
MAX_LAG = 8


def _decode(values: np.ndarray) -> list[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


@dataclass(frozen=True)
class ModelArtifacts:
    posterior: dict[str, np.ndarray]
    coords: dict[str, list[str]]
    data: pd.DataFrame
    train_mask: np.ndarray
    control_names: list[str]
    control_mean: np.ndarray
    control_scale: np.ndarray
    spend_scale: np.ndarray
    y_mean: float
    y_sd: float
    show_lookup: dict[str, int]
    show_season_lookup: dict[str, int]
    constant_data: dict[str, np.ndarray]
    model_path: Path

    @property
    def n_samples(self) -> int:
        return int(self.posterior["sigma"].shape[0])


def _control_frame(frame: pd.DataFrame, control_names: list[str]) -> pd.DataFrame:
    episode = pd.get_dummies(frame["Episode_Type"], prefix="episode", dtype=float)
    controls = pd.concat(
        [frame[["Holiday", "LeadIn_Bonus", "Week_Number"]].astype(float), episode],
        axis=1,
    )
    return controls.reindex(columns=control_names, fill_value=0.0)


def load_artifacts(model_path: Path = MODEL_PATH, data_path: Path = DATA_PATH) -> ModelArtifacts:
    data = pd.read_csv(data_path, parse_dates=["Air_Date"])
    data = data.sort_values(["Show", "Season", "Week_Number", "Air_Date"]).reset_index(drop=True)
    latest = data.groupby("Show")["Season"].transform("max")
    train_mask = (data["Season"] != latest).to_numpy()

    with h5py.File(model_path, "r") as file:
        posterior_group = file["posterior"]
        coords = {
            name: _decode(posterior_group[name][...])
            for name in ("channel", "control", "show", "show_season")
        }
        posterior: dict[str, np.ndarray] = {}
        coordinate_names = {"chain", "draw", "channel", "control", "show", "show_season", "obs"}
        for name, dataset in posterior_group.items():
            if name in coordinate_names:
                continue
            values = dataset[...]
            posterior[name] = values.reshape((-1,) + values.shape[2:])
        constant_data = {
            name: dataset[...]
            for name, dataset in file["constant_data"].items()
            if name not in {"obs", "channel", "control", "lag"}
        }

    controls = _control_frame(data, coords["control"])
    train_controls = controls.loc[train_mask]
    control_mean = train_controls.mean(axis=0).to_numpy(float)
    control_scale = train_controls.std(axis=0, ddof=0).to_numpy(float)
    control_scale = np.where(control_scale == 0, 1.0, control_scale)

    spend_raw = data[SPEND_COLUMNS].to_numpy(float)
    y_raw = data["Revenue"].to_numpy(float)
    spend_scale = np.quantile(spend_raw[train_mask], 0.95, axis=0)
    y_mean = float(y_raw[train_mask].mean())
    y_sd = float(y_raw[train_mask].std())

    return ModelArtifacts(
        posterior=posterior,
        coords=coords,
        data=data,
        train_mask=train_mask,
        control_names=coords["control"],
        control_mean=control_mean,
        control_scale=control_scale,
        spend_scale=spend_scale,
        y_mean=y_mean,
        y_sd=y_sd,
        show_lookup={name: i for i, name in enumerate(coords["show"])},
        show_season_lookup={name: i for i, name in enumerate(coords["show_season"])},
        constant_data=constant_data,
        model_path=model_path,
    )


@st.cache_resource(show_spinner="Loading the trained posterior…")
def load_model_artifacts() -> ModelArtifacts:
    return load_artifacts()


def controls_for(frame: pd.DataFrame, artifacts: ModelArtifacts) -> np.ndarray:
    raw = _control_frame(frame.reset_index(drop=True), artifacts.control_names).to_numpy(float)
    return (raw - artifacts.control_mean) / artifacts.control_scale
