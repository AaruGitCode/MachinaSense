
from pathlib import Path

import pandas as pd
from sklearn.ensemble import IsolationForest


BASE_DIR = Path(__file__).parent
INPUT_FILE = BASE_DIR / "data" / "nasa_ims_vibration_features.csv"
OUTPUT_FILE = BASE_DIR / "data" / "nasa_ims_anomaly_results.csv"


def main():
    data = pd.read_csv(INPUT_FILE)
    data["timestamp"] = pd.to_datetime(data["timestamp"])

    feature_columns = [
        column
        for column in data.columns
        if column.endswith(("_rms", "_std", "_peak_to_peak"))
    ]

    if not feature_columns:
        raise ValueError("No vibration features found.")

    X = data[feature_columns].replace(
        [float("inf"), float("-inf")], float("nan")
    )

    valid = X.notna().all(axis=1)
    data = data.loc[valid].copy()
    X = X.loc[valid]

    model = IsolationForest(
        n_estimators=200,
        contamination=0.01,
        random_state=42,
    )

    # Keep the snapshots in chronological order.
    split_index = int(len(X) * 0.70)

    X_train = X.iloc[:split_index]
    X_all = X

    model = IsolationForest(
        n_estimators=200,
        contamination=0.01,
        random_state=42,
    )

    # Learn the baseline using only earlier snapshots.
    model.fit(X_train)

    # Score all snapshots using the early-period model.
    data["anomaly_score"] = model.decision_function(X_all)
    data["anomaly_prediction"] = model.predict(X_all)
    data["is_anomaly"] = data["anomaly_prediction"] == -1

    data["evaluation_period"] = "Later evaluation"
    data.loc[data.index[:split_index], "evaluation_period"] = "Training"

    # Save the results.
    data.to_csv(OUTPUT_FILE, index=False)

    print("Chronological anomaly detection complete!")
    print("Training snapshots:", split_index)
    print("Total snapshots scored:", len(data))
    print(
        "Training-period candidates:",
        int(data.iloc[:split_index]["is_anomaly"].sum()),
    )
    print(
        "Later-period candidates:",
        int(data.iloc[split_index:]["is_anomaly"].sum()),
    )
    print("Output file:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
