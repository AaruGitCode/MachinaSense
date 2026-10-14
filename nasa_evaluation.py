
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).parent
INPUT_FILE = BASE_DIR / "data" / "nasa_ims_anomaly_results.csv"

# Documented end of NASA IMS Test 2.
FAILURE_ENDPOINT = pd.Timestamp("2004-02-19 06:22:39")


def main():
    data = pd.read_csv(INPUT_FILE)
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data = data.sort_values("timestamp").reset_index(drop=True)

    if data.empty:
        raise ValueError("The anomaly results file is empty.")

    print("NASA IMS Test 2 - Exploratory Evaluation")
    print("-" * 45)
    print("Snapshots:", len(data))
    print("First timestamp:", data["timestamp"].min())
    print("Last timestamp:", data["timestamp"].max())
    print("Documented experiment endpoint:", FAILURE_ENDPOINT)

    # Evaluate anomaly candidates only in the later period.
    later_data = data[
        data["evaluation_period"] == "Later evaluation"
    ].copy()

    anomalies = later_data[
        later_data["is_anomaly"].astype(bool)
    ].copy()

    print("\nLater-period evaluation")
    print("Snapshots evaluated:", len(later_data))
    print("Anomaly candidates:", len(anomalies))

    candidate_rate = (
        len(anomalies) / len(later_data) * 100
        if len(later_data) else 0
    )
    print("Candidate rate:", f"{candidate_rate:.2f}%")

    if anomalies.empty:
        print("No anomaly candidates were detected.")
    else:
        first_anomaly = anomalies["timestamp"].min()

        hours_before_endpoint = (
            FAILURE_ENDPOINT - first_anomaly
        ).total_seconds() / 3600

        print("First flagged timestamp:", first_anomaly)
        print(
            "Time from first flag to experiment endpoint:",
            f"{hours_before_endpoint:.2f} hours",
        )
        
    # Compare early and late vibration levels.
    # This is a descriptive comparison, not a fault label.
    feature = "channel_1_rms"

    if feature in data.columns:
        split = max(1, int(len(data) * 0.70))
        early = data.iloc[:split][feature]
        late = data.iloc[split:][feature]

        print("\nBearing 1 vibration comparison")
        print("Early-period median RMS:", round(early.median(), 6))
        print("Late-period median RMS:", round(late.median(), 6))

        if early.median() != 0:
            ratio = late.median() / early.median()
            print("Late / early median ratio:", round(ratio, 3))

    print(
        "\nCaution: the endpoint is not a confirmed onset timestamp. "
        "Anomaly flags are not confirmed failure labels."
    )


if __name__ == "__main__":
    main()
