
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


BASE_DIR = Path(__file__).parent
INPUT_FILE = BASE_DIR / "data" / "nasa_ims_anomaly_results.csv"

st.title("MachinaSense — Daily Anomaly Trend")

data = pd.read_csv(INPUT_FILE)
data["timestamp"] = pd.to_datetime(data["timestamp"])
data["date"] = data["timestamp"].dt.date

daily = (
    data.groupby("date")
    .agg(
        snapshots=("is_anomaly", "size"),
        anomalies=("is_anomaly", "sum"),
        mean_rms=("channel_1_rms", "mean"),
    )
    .reset_index()
)

daily["anomaly_rate_pct"] = (
    daily["anomalies"] / daily["snapshots"] * 100
)

fig = px.line(
    daily,
    x="date",
    y="anomaly_rate_pct",
    markers=True,
    title="Daily Anomaly Rate",
    labels={
        "date": "Date",
        "anomaly_rate_pct": "Snapshots flagged (%)",
    },
)

st.plotly_chart(fig, use_container_width=True)
st.dataframe(daily, use_container_width=True, hide_index=True)
