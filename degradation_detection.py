
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).parent
INPUT_FILE = BASE_DIR / "data" / "nasa_ims_vibration_features.csv"
OUTPUT_FILE = BASE_DIR / "data" / "nasa_ims_degradation_results.csv"


def main():
    data = pd.read_csv(INPUT_FILE)
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data = data.sort_values("timestamp").reset_index(drop=True)

    feature = "channel_1_rms"

    if feature not in data.columns:
        raise ValueError(f"Missing feature: {feature}")

    # Use the first 30% as a reference baseline.
    baseline_end = max(1, int(len(data) * 0.30))
    baseline = data.loc[:baseline_end - 1, feature].median()

    if baseline <= 0:
        raise ValueError("Baseline RMS must be positive.")

    # Smooth the signal using a 12-snapshot rolling average.
    data["rolling_rms"] = (
        data[feature].rolling(window=12, min_periods=1).mean()
    )

    data["change_from_baseline_pct"] = (
        (data["rolling_rms"] - baseline) / baseline * 100
    )

    # Exploratory threshold; not an industrial safety limit.
    data["degradation_candidate"] = (
        data["change_from_baseline_pct"] >= 50
    )

    data.to_csv(OUTPUT_FILE, index=False)

    candidates = data[data["degradation_candidate"]]

    print("Vibration degradation analysis complete!")
    print("Snapshots analyzed:", len(data))
    print("Baseline RMS:", round(baseline, 6))
    print("Snapshots above 50% increase:", len(candidates))

    if not candidates.empty:
        print(
            "First threshold crossing:",
            candidates.iloc[0]["timestamp"],
        )

    print("Output file:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
