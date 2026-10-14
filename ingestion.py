
from pathlib import Path
import pandas as pd

# Locate the CSV file relative to this Python script
DATA_FILE = (
    Path(__file__).parent
    / "data"
    / "sample_motor_current_simulated.csv"
)

REQUIRED_COLUMNS = [
    "timestamp",
    "machine_id",
    "current_rms_a",
    "operating_state",
]


def load_sensor_data():
    # 1. Check whether the file exists
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Sensor data file not found: {DATA_FILE}"
        )

    # 2. Read the CSV
    df = pd.read_csv(DATA_FILE)

    # 3. Check the required columns
    missing_columns = (
        set(REQUIRED_COLUMNS) - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing columns: {sorted(missing_columns)}"
        )

    # 4. Validate timestamps
    df["timestamp"] = pd.to_datetime(
        df["timestamp"], errors="coerce"
    )

    # 5. Convert current measurements to numeric values
    df["current_rms_a"] = pd.to_numeric(
        df["current_rms_a"], errors="coerce"
    )

    # 6. Identify invalid records
    invalid_rows = (
        df["timestamp"].isna()
        | df["current_rms_a"].isna()
        | (df["current_rms_a"] < 0)
        | df["machine_id"].isna()
        | df["operating_state"].isna()
    )

    print("Total records:", len(df))
    print("Invalid records:", int(invalid_rows.sum()))

    # 7. Keep valid records for the next processing stage
    valid_df = df.loc[~invalid_rows].copy()

    print("Valid records:", len(valid_df))
    print("\nFirst five valid records:")
    print(valid_df.head().to_string(index=False))

    return valid_df


if __name__ == "__main__":
    sensor_data = load_sensor_data()
