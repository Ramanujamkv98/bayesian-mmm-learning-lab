from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from .model_loader import ModelArtifacts

PALETTE = ["#5B5FEF", "#19B5A5", "#FF9F43", "#EC5B8C", "#7D8CA3"]


def setup_page(title: str, icon: str, intro: str) -> None:
    st.set_page_config(page_title=f"{title} · MMM Learning Lab", page_icon=icon, layout="wide")
    st.markdown(
        """
        <style>
        .stApp {background: linear-gradient(145deg,#f7f8ff 0%,#fffaf4 52%,#f3fbfa 100%);}
        [data-testid="stSidebar"] {background:#17192e;}
        [data-testid="stSidebar"] * {color:#f7f7ff;}
        .block-container {max-width:1220px;padding-top:2.4rem;padding-bottom:4rem;}
        h1,h2,h3 {letter-spacing:-0.035em;color:#191b31;}
        h1 {font-size:clamp(2.3rem,5vw,4.2rem)!important;line-height:1.02!important;}
        div[data-testid="stMetric"] {background:rgba(255,255,255,.84);border:1px solid #e8e8f4;
          border-radius:18px;padding:1rem 1.15rem;box-shadow:0 8px 30px rgba(45,47,105,.07)}
        div[data-testid="stPlotlyChart"] {background:rgba(255,255,255,.72);border:1px solid #ececf5;border-radius:22px;padding:.45rem;}
        .lab-callout {background:#ececff;border-left:5px solid #5B5FEF;border-radius:14px;padding:1rem 1.15rem;margin:.7rem 0 1.2rem;}
        .sensitivity {background:#fff1df;border:1px solid #ffbd72;border-radius:14px;padding:.8rem 1rem;font-weight:650;color:#78420d;}
        .eyebrow {font-size:.78rem;font-weight:800;text-transform:uppercase;letter-spacing:.14em;color:#5B5FEF;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(f'<div class="eyebrow">Bayesian MMM Learning Lab</div>', unsafe_allow_html=True)
    st.title(f"{icon} {title}")
    st.markdown(f'<div class="lab-callout">{intro}</div>', unsafe_allow_html=True)


def money(value: float) -> str:
    sign = "−" if value < 0 else ""
    value = abs(float(value))
    if value >= 1_000_000:
        return f"{sign}${value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{sign}${value / 1_000:.1f}K"
    return f"{sign}${value:,.0f}"


def scenario_picker(artifacts: ModelArtifacts, prefix: str = "scenario") -> tuple[pd.DataFrame, int]:
    data = artifacts.data
    columns = st.columns([1, 1, 1])
    show = columns[0].selectbox("Show", artifacts.coords["show"], key=f"{prefix}_show")
    seasons = sorted(data.loc[data["Show"] == show, "Season"].unique().tolist())
    season = columns[1].selectbox("Season", seasons, index=len(seasons) - 1, key=f"{prefix}_season")
    frame = data[(data["Show"] == show) & (data["Season"] == season)].copy().reset_index(drop=True)
    week = columns[2].selectbox("Lifecycle week", frame["Week_Number"].tolist(), key=f"{prefix}_week")
    row_index = int(frame.index[frame["Week_Number"] == week][0])
    return frame, row_index


def sensitivity_banner(active: bool) -> None:
    if active:
        st.markdown(
            '<div class="sensitivity">🧪 Sensitivity / What-if mode — these values are user-defined and are not necessarily supported by the fitted posterior.</div>',
            unsafe_allow_html=True,
        )


def clean_figure(fig, height: int = 390):
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=55, b=25),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, ui-sans-serif, system-ui", color="#252742"),
        hoverlabel=dict(bgcolor="white"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(gridcolor="#e7e8f0", zeroline=False)
    fig.update_yaxes(gridcolor="#e7e8f0", zeroline=False)
    return fig
