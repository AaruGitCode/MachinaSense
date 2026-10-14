
from sklearn.metrics import classification_report, confusion_matrix

from ingestion import load_sensor_data
from anomaly_detection import detect_anomalies
from ml_detection import detect_with_ml


df = load_sensor_data()

# Ground truth from our simulated sample
actual = df["operating_state"].eq("simulated_high_current")

# Existing threshold-based detector
threshold_results = detect_anomalies(df)
threshold_predictions = threshold_results["is_anomaly"]

# Isolation Forest detector
ml_results = detect_with_ml(df)
ml_predictions = ml_results["ml_anomaly"]


def evaluate(name, actual, predicted):
    tn, fp, fn, tp = confusion_matrix(
        actual, predicted, labels=[False, True]
    ).ravel()

    print(f"\n{name}")
    print("-" * len(name))
    print(f"True negatives:  {tn}")
    print(f"False positives: {fp}")
    print(f"False negatives: {fn}")
    print(f"True positives:  {tp}")

    print(
        classification_report(
            actual,
            predicted,
            labels=[False, True],
            target_names=["Normal", "High current"],
            zero_division=0,
        )
    )


evaluate("Threshold detector", actual, threshold_predictions)
evaluate("Isolation Forest", actual, ml_predictions)
