
from ingestion import load_sensor_data

df = load_sensor_data()

# Simulate a missing sensor reading
df.loc[5, "current_rms_a"] = float("nan")

missing_count = df["current_rms_a"].isna().sum()

print("Missing current readings:", missing_count)

# Do not silently classify missing readings as normal
valid_data = df.dropna(subset=["current_rms_a"])

print("Records available for processing:", len(valid_data))
print("Records requiring investigation:", missing_count)

