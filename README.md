# MachinaSense ⚙️
### Intelligent Industrial Fault Detection & Predictive Maintenance

## 1. Overview

MachinaSense is a machine-condition monitoring prototype that analyzes industrial vibration data to identify unusual patterns and potential signs of equipment degradation.

It combines machine-learning-based anomaly detection with vibration trend analysis to help maintenance teams identify readings that may warrant further inspection.

## 2. Problem Statement

Industrial equipment can develop mechanical problems that cause changes in vibration patterns. If these changes go unnoticed, maintenance teams may miss opportunities to inspect equipment before a serious failure occurs.

MachinaSense explores how vibration analysis and anomaly detection can support condition monitoring and maintenance decisions.

## 3. Solution

MachinaSense provides:

- Vibration data ingestion and feature extraction.
- Anomaly detection using Isolation Forest.
- Rolling-average vibration trend analysis.
- Baseline-based degradation candidate detection.
- A Streamlit dashboard with interactive charts.
- Maintenance alert tables and CSV export.
- Daily anomaly-rate visualization.

## 4. Technology Stack

- **Language:** Python
- **Data processing:** Pandas, NumPy
- **Machine learning:** Scikit-learn, Isolation Forest
- **Visualization:** Plotly
- **Dashboard:** Streamlit
- **Dataset:** NASA IMS bearing vibration dataset

## 5. System Workflow

1. Load the NASA vibration snapshots.
2. Calculate RMS, standard deviation, and peak-to-peak features for four channels.
3. Apply Isolation Forest to identify unusual vibration patterns.
4. Calculate rolling averages and compare vibration with an early-period baseline.
5. Display trends and anomaly candidates in the dashboard.
6. Generate maintenance inspection recommendations and export alert records.

## 6. Current Experimental Results

- Vibration snapshots processed: 984
- Vibration channels analyzed: 4
- Anomaly candidates in the original model run: 30
- Degradation candidates using the illustrative 50% threshold: 226
- Channel 1 early-period median RMS: approximately 0.0777
- Channel 1 late-period median RMS: approximately 0.1557

In a separate chronological experiment, the model was trained on the first 70% of snapshots. With a 1% contamination setting, it flagged 288 of the 296 later snapshots.

These results indicate a substantial change in vibration behavior, but they do not establish predictive accuracy or confirm individual bearing failures.

## 7. Installation

Install Python and open a terminal in the project directory.

Install the dependencies:

```bash
python -m pip install pandas numpy scikit-learn streamlit plotly
```

## 8. Run the Project

### Prepare vibration features

```bash
python prepare_vibration_data.py
```

### Run anomaly detection

```bash
python nasa_ml_detection.py
```

### Run degradation analysis

```bash
python degradation_detection.py
```

### Launch the main dashboard

```bash
python -m streamlit run nasa_dashboard.py
```

### Launch the daily anomaly trend

```bash
python -m streamlit run anomaly_trend.py
```

Run the scripts in the order shown when regenerating the outputs from the raw dataset.

## 9. Project Structure

```text
MachinaSense/
├── data/
│   ├── nasa_ims/raw/
│   ├── nasa_ims_vibration_features.csv
│   ├── nasa_ims_anomaly_results.csv
│   └── nasa_ims_degradation_results.csv
├── prepare_vibration_data.py
├── nasa_ml_detection.py
├── nasa_evaluation.py
├── degradation_detection.py
├── nasa_dashboard.py
├── anomaly_trend.py
└── README.md
```

## 10. Limitations and Future Work

- Validate anomaly predictions against documented fault information.
- Investigate the high anomaly rate in the later evaluation period.
- Calibrate alert severity using validated criteria.
- Evaluate performance on independent data and known fault labels.
- Explore adaptive baselines, additional vibration features, and real-time sensor integration.

## 11. Disclaimer

MachinaSense is an experimental prototype. Anomaly candidates and degradation indicators are not confirmed equipment failures, and the current thresholds are not validated industrial safety limits.
