
import streamlit as st
import plotly.express as px

from ingestion import load_sensor_data
from anomaly_detection import detect_anomalies


st.set_page_config(
    page_title="MachinaSense",
    page_icon="⚙️",
    layout="wide",
)

st.title("⚙️ MachinaSense")
st.subheader("Industrial Fault Detection & Predictive Maintenance")

st.caption(
    "Prototype dashboard using simulated motor current data. "
    "The 10 A threshold is for demonstration only."
)

# Load and process sensor data
try:
    df = load_sensor_data()
    results = detect_anomalies(df)

except (FileNotFoundError, ValueError) as error:
    st.error(f"Unable to load sensor data: {error}")
    st.stop()

# Calculate dashboard metrics
total_readings = len(results)
anomaly_count = int(results["is_anomaly"].sum())
normal_count = total_readings - anomaly_count

col1, col2, col3 = st.columns(3)

col1.metric("Total Readings", total_readings)
col2.metric("Normal Readings", normal_count)
col3.metric("Anomaly Candidates", anomaly_count)

st.divider()

# Sensor current trend
st.subheader("Motor Current Monitoring")

fig = px.line(
    results,
    x="timestamp",
    y="current_rms_a",
    title="Current Consumption Over Time",
    labels={
        "timestamp": "Timestamp",
        "current_rms_a": "RMS Current (A)",
    },
)

fig.add_hline(
    y=10.0,
    line_dash="dash",
    annotation_text="Demo threshold: 10 A",
)

anomalies = results[results["is_anomaly"]]

if not anomalies.empty:
    fig.add_scatter(
        x=anomalies["timestamp"],
        y=anomalies["current_rms_a"],
        mode="markers",
        name="Anomaly candidates",
        marker={"size": 11, "symbol": "x"},
    )

st.plotly_chart(fig, use_container_width=True)

# Display detected alerts
st.subheader("Detected Alerts")

if anomalies.empty:
    st.success("No readings exceeded the configured threshold.")
else:
    alerts = anomalies[
        ["timestamp", "machine_id", "current_rms_a"]
    ].copy()

    alerts["severity"] = alerts["current_rms_a"].apply(
        lambda value: "Critical" if value > 12.0 else "Warning"
    )

    alerts["evidence"] = alerts["current_rms_a"].apply(
        lambda value: (
            f"{value:.2f} A exceeds the 10 A "
            "demonstration threshold"
        )
    )

    st.dataframe(
        alerts,
        use_container_width=True,
        hide_index=True,
    )

# Show the complete dataset
with st.expander("View all sensor readings"):
    st.dataframe(results, use_container_width=True)

st.caption(
    "High current is an anomaly candidate, not proof of equipment failure. "
    "Real thresholds must be based on equipment specifications and "
    "validated operating data."
)
