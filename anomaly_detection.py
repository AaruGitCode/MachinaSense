
from ingestion import load_sensor_data

# Demonstration threshold, not an actual motor safety limit
CURRENT_THRESHOLD_A = 10.0


def detect_anomalies(df):
    # Copy the data so the original DataFrame is unchanged
    results = df.copy()

    # Flag readings above the configured threshold
    results["is_anomaly"] = (
        results["current_rms_a"] > CURRENT_THRESHOLD_A
    )

    # Assign a simple alert label
    results["alert"] = results["is_anomaly"].map(
        {True: "High Current", False: "Normal"}
    )

    return results


if __name__ == "__main__":
    sensor_data = load_sensor_data()
    results = detect_anomalies(sensor_data)

    print("\nDetected alerts:")
    print(
        results.loc[
            results["is_anomaly"],
            ["timestamp", "machine_id",
             "current_rms_a", "alert"]
        ].to_string(index=False)
    )

    print("\nTotal anomalies:", int(results["is_anomaly"].sum()))
