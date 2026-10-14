
from sklearn.ensemble import IsolationForest
from ingestion import load_sensor_data


def detect_with_ml(df):
    results = df.copy()

    # Use current consumption as the model feature
    features = results[["current_rms_a"]]

    # Create and train the anomaly-detection model
    model = IsolationForest(
        n_estimators=100,
        contamination="auto",
        random_state=42,
    )

    model.fit(features)

    # 1 = normal, -1 = anomaly
    predictions = model.predict(features)

    results["ml_anomaly"] = predictions == -1
    results["anomaly_score"] = model.decision_function(features)

    return results


if __name__ == "__main__":
    sensor_data = load_sensor_data()
    results = detect_with_ml(sensor_data)

    anomalies = results[results["ml_anomaly"]]

    print("Total readings:", len(results))
    print("ML anomaly candidates:", len(anomalies))

    print(
        anomalies[
            ["timestamp", "machine_id",
             "current_rms_a", "anomaly_score"]
        ].to_string(index=False)
    )
