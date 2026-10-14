
from pathlib import Path
import re

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).parent
RAW_DIR = BASE_DIR / "data" / "nasa_ims" / "raw"
OUTPUT_FILE = BASE_DIR / "data" / "nasa_ims_vibration_features.csv"

TIMESTAMP_PATTERN = re.compile(
    r"(\d{4}\.\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{2})"
)


def extract_timestamp(filename):
    match = TIMESTAMP_PATTERN.search(filename)

    if not match:
        return None

    return pd.to_datetime(
        match.group(1),
        format="%Y.%m.%d.%H.%M.%S",
        errors="coerce",
    )


def summarize_snapshot(file_path):
    timestamp = extract_timestamp(file_path.name)

    if timestamp is None or pd.isna(timestamp):
        print(f"Skipping invalid timestamp: {file_path.name}")
        return None

    try:
        data = pd.read_csv(
            file_path,
            sep=r"\s+",
            header=None,
            engine="c",
        )
    except (OSError, pd.errors.ParserError) as exc:
        print(f"Could not read {file_path.name}: {exc}")
        return None

    # The sample files we inspected contain four columns.
    if data.shape[1] != 4:
        print(
            f"Skipping {file_path.name}: "
            f"expected 4 columns, found {data.shape[1]}"
        )
        return None

    data = data.apply(pd.to_numeric, errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    if data.empty:
        print(f"Skipping empty snapshot: {file_path.name}")
        return None

    record = {"timestamp": timestamp}

    for i in range(4):
        values = data.iloc[:, i].to_numpy(dtype=float)

        record[f"channel_{i + 1}_rms"] = float(
            np.sqrt(np.mean(values ** 2))
        )
        record[f"channel_{i + 1}_std"] = float(
            np.std(values)
        )
        record[f"channel_{i + 1}_peak_to_peak"] = float(
            np.max(values) - np.min(values)
        )

    record["sample_count"] = len(data)

    return record


def main():
    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {RAW_DIR}"
        )

    # Search subfolders too, if the archive contains them.
    files = sorted(
        path
        for path in RAW_DIR.rglob("*")
        if path.is_file()
        and TIMESTAMP_PATTERN.search(path.name)
    )

    if not files:
        raise FileNotFoundError(
            f"No timestamped files found under: {RAW_DIR}"
        )

    print(f"Found {len(files)} timestamped files.")

    records = []

    for index, file_path in enumerate(files, start=1):
        record = summarize_snapshot(file_path)

        if record is not None:
            records.append(record)

        if index % 100 == 0 or index == len(files):
            print(f"Processed {index}/{len(files)} files.")

    if not records:
        raise ValueError(
            "No usable snapshots found. Check the file format."
        )

    result = pd.DataFrame(records)
    result = result.sort_values("timestamp")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_FILE, index=False)

    print("\nPreparation complete!")
    print("Snapshots processed:", len(result))
    print("Output file:", OUTPUT_FILE)
    print("\nPreview:")
    print(result.head().to_string(index=False))


if __name__ == "__main__":
    main()
