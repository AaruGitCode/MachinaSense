
from pathlib import Path
from anomaly_detection import detect_anomalies
from ingestion import load_sensor_data

# Example thresholds for simulated data
WARNING_THRESHOLD = 10.0
CRITICAL_THRESHOLD = 12.0


def classify_severity(current):
    if current > CRITICAL_THRESHOLD:
        return "Critical"
    elif current > WARNING_THRESHOLD:
        return "Warning"
    else:
        return "Normal"


def generate_alerts(df):
    results = detect_anomalies(df).copy()

    results["severity"] = results["current_rms_a"].apply(
        classify_severity
    )

    # Only generate alerts for abnormal readings
    alerts = results.loc[
        results["is_anomaly"],
        ["timestamp", "machine_id", "current_rms_a", "severity"]
    ].copy()

    alerts["alert_type"] = "High Current"

    alerts["evidence"] = alerts["current_rms_a"].apply(
        lambda value: (
            f"Current {value:.2f} A exceeds "
            f"the {WARNING_THRESHOLD:.2f} A threshold"
        )
    )

    return alerts


if __name__ == "__main__":
    sensor_data = load_sensor_data()
    alerts = generate_alerts(sensor_data)

    print("\nGenerated maintenance alerts:")
    print(alerts.to_string(index=False))

    # Save alert records for dashboard/API integration
    output_path = (
        Path(__file__).parent / "data" / "maintenance_alerts.csv"
    )

    alerts.to_csv(output_path, index=False)

    print(f"\nAlerts saved to: {output_path}")
