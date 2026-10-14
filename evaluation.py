
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
)
from ingestion import load_sensor_data
from anomaly_detection import detect_anomalies


def evaluate_model(df):
    # Ground-truth labels supplied by our simulated dataset
    actual = (
        df["operating_state"] == "simulated_high_current"
    )

    # Predictions from our existing detection rule
    results = detect_anomalies(df)
    predicted = results["is_anomaly"]

    # Compare actual labels with predicted labels
    tn, fp, fn, tp = confusion_matrix(
        actual, predicted, labels=[False, True]
    ).ravel()

    print("Evaluation results")
    print("------------------")
    print("True negatives:", tn)
    print("False positives:", fp)
    print("False negatives:", fn)
    print("True positives:", tp)

    print("\nClassification report:")
    print(
        classification_report(
            actual,
            predicted,
            labels=[False, True],
            target_names=["Normal", "High current"],
            zero_division=0,
        )
    )


if __name__ == "__main__":
    sensor_data = load_sensor_data()
    evaluate_model(sensor_data)
