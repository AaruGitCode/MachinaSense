
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


BASE_DIR = Path(__file__).parent
INPUT_FILE = BASE_DIR / "data" / "nasa_ims_anomaly_results.csv"

st.set_page_config(
    page_title="MachinaSense | Vibration Monitoring",
    page_icon="⚙️",
    layout="wide",
)

st.title("⚙️ MachinaSense")
st.subheader("Industrial Vibration Anomaly Monitoring")
st.caption("NASA IMS bearing vibration dataset | Isolation Forest")

if not INPUT_FILE.exists():
    st.error("Anomaly results not found. Run nasa_ml_detection.py first.")
    st.stop()

data = pd.read_csv(INPUT_FILE)
data["timestamp"] = pd.to_datetime(data["timestamp"])
data = data.sort_values("timestamp")

total = len(data)
anomaly_count = int(data["is_anomaly"].sum())
normal_count = total - anomaly_count

col1, col2, col3 = st.columns(3)

col1.metric("Snapshots Analyzed", total)
col2.metric("Anomaly Candidates", anomaly_count)
col3.metric(
    "Candidate Rate",
    f"{anomaly_count / total * 100:.1f}%" if total else "N/A",
)

st.info(
    "Anomaly candidates are unusual vibration patterns flagged by "
    "the model. They are not confirmed equipment failures."
)

st.divider()
st.subheader("Vibration Trend")

rms_columns = [
    column for column in data.columns
    if column.endswith("_rms")
]

if rms_columns:
    selected_channel = st.selectbox(
        "Select vibration channel",
        rms_columns,
    )

    
    # Smooth short-term fluctuations with a rolling average.
    chart_data = data.copy()
    chart_data["rolling_rms"] = (
        chart_data[selected_channel]
        .rolling(window=12, min_periods=1)
        .mean()
    )

    # Compare the latest rolling average with the earlier baseline.
    baseline = chart_data[selected_channel].iloc[
        :max(1, len(chart_data) // 2)
    ].median()

    latest_rolling = chart_data["rolling_rms"].iloc[-1]

    st.subheader("Vibration Trend Assessment")

    if baseline > 0:
        change_pct = ((latest_rolling - baseline) / baseline) * 100

        col1, col2, col3 = st.columns(3)

        col1.metric("Baseline RMS", f"{baseline:.4f}")
        col2.metric("Latest Rolling RMS", f"{latest_rolling:.4f}")
        col3.metric("Change vs Baseline", f"{change_pct:+.1f}%")

        if change_pct >= 50:
            st.warning(
                "Vibration has increased substantially compared with "
                "the historical baseline. Inspect the equipment."
            )
        else:
            st.info(
                "Review the trend and anomaly markers for unusual "
                "vibration patterns."
            )


    fig = px.line(
        chart_data,
        x="timestamp",
        y=selected_channel,
        title=f"{selected_channel}: Vibration and Trend",
        labels={
            "timestamp": "Timestamp",
            selected_channel: "RMS vibration",
        },
    )

    fig.add_scatter(
        x=chart_data["timestamp"],
        y=chart_data["rolling_rms"],
        mode="lines",
        name="12-snapshot rolling average",
        line={"width": 3},
    )

    anomalies = chart_data[chart_data["is_anomaly"]]

    fig.add_scatter(
        x=anomalies["timestamp"],
        y=anomalies[selected_channel],
        mode="markers",
        name="Anomaly candidates",
        marker={"size": 9, "symbol": "x"},
    )

    st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("No RMS vibration features were found.")

st.divider()
st.subheader("Most Unusual Snapshots")

if "anomaly_score" in data.columns:
    unusual = data.sort_values("anomaly_score").head(20)

    display_columns = [
        "timestamp",
        "anomaly_score",
        "is_anomaly",
    ]

    st.dataframe(
        unusual[display_columns],
        use_container_width=True,
        hide_index=True,
    )

st.subheader("All Anomaly Candidates")

st.divider()
st.subheader("Machine Health Assessment")

def get_health_status(row):
    if row["is_anomaly"]:
        return "Warning"
    return "Normal"

data["health_status"] = data.apply(get_health_status, axis=1)

latest = data.sort_values("timestamp").iloc[-1]

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Latest Snapshot",
        latest["timestamp"].strftime("%d %b %Y, %H:%M"),
    )

with col2:
    st.metric(
        "Latest Snapshot Status",
        latest["health_status"],
    )

st.caption(
    "Status is based on the Isolation Forest anomaly prediction. "
    "It is an indicator of unusual vibration, not a confirmed diagnosis."
)


candidates = data[data["is_anomaly"]].sort_values("timestamp")
st.dataframe(candidates, use_container_width=True, hide_index=True)


st.divider()
st.subheader("🛠️ Maintenance Alerts")

alerts = data[data["is_anomaly"]].copy()

if alerts.empty:
    st.success("No anomaly candidates detected in this dataset.")
else:
    alerts["severity"] = alerts["anomaly_score"].apply(
        lambda score: "Critical" if score < -0.10 else "Warning"
    )

    alerts["recommended_action"] = alerts["severity"].map({
        "Critical": "Prioritize inspection and review vibration readings.",
        "Warning": "Inspect the trend and schedule a bearing check.",
    })

    alerts = alerts.sort_values("timestamp", ascending=False)

    st.dataframe(
        alerts[
            [
                "timestamp",
                "severity",
                "anomaly_score",
                "recommended_action",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    csv_data = alerts.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Maintenance Alerts CSV",
        data=csv_data,
        file_name="machinasense_maintenance_alerts.csv",
        mime="text/csv",
    )


st.divider()
st.subheader("📈 Degradation Detection")

DEGRADATION_FILE = (
    BASE_DIR / "data" / "nasa_ims_degradation_results.csv"
)

if DEGRADATION_FILE.exists():
    degradation = pd.read_csv(DEGRADATION_FILE)
    degradation["timestamp"] = pd.to_datetime(
        degradation["timestamp"]
    )

    st.metric(
        "Snapshots Above Degradation Threshold",
        int(degradation["degradation_candidate"].sum()),
    )

    candidate_rows = degradation[
        degradation["degradation_candidate"]
    ]

    col1, col2 = st.columns(2)

    if not candidate_rows.empty:
        first_crossing = candidate_rows["timestamp"].min()

        col1.metric(
            "First Threshold Crossing",
            first_crossing.strftime("%d %b %Y, %H:%M"),
        )

        col2.metric(
            "Baseline RMS",
            f"{degradation['channel_1_rms'].iloc[:max(1, len(degradation) * 30 // 100)].median():.4f}",
        )
    else:
        st.info("No snapshots crossed the degradation threshold.")

    fig_deg = px.line(
        degradation,
        x="timestamp",
        y="rolling_rms",
        title="Vibration Degradation Trend",
        labels={
            "timestamp": "Timestamp",
            "rolling_rms": "Rolling RMS vibration",
        },
    )

    candidates = degradation[
        degradation["degradation_candidate"]
    ]

    fig_deg.add_scatter(
        x=candidates["timestamp"],
        y=candidates["rolling_rms"],
        mode="markers",
        name="Degradation candidates",
        marker={"size": 7, "symbol": "x"},
    )

    st.plotly_chart(fig_deg, use_container_width=True)
else:
    st.warning(
        "Run degradation_detection.py to generate degradation results."
    )
