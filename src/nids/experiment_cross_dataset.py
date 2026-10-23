
"""Initial five-feature cross-dataset NIDS transfer experiment.

Source: UNSW-NB15 training CSV only.
Target: CICIDS2017 Friday DDoS CSV.
The official UNSW-NB15 test set is not used.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[2]

SOURCE_PATH = ROOT / "data/raw/nids/UNSW_NB15_training-set.csv"
TARGET_PATH = (
    ROOT
    / "data/raw/nids/cicids2017/"
    / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
)

FEATURES = [
    "duration_seconds",
    "forward_packets",
    "backward_packets",
    "forward_bytes",
    "backward_bytes",
]


def load_source():
    columns = ["dur", "spkts", "dpkts", "sbytes", "dbytes", "label"]
    data = pd.read_csv(SOURCE_PATH, usecols=columns)

    X = pd.DataFrame({
        "duration_seconds": pd.to_numeric(data["dur"], errors="coerce"),
        "forward_packets": pd.to_numeric(data["spkts"], errors="coerce"),
        "backward_packets": pd.to_numeric(data["dpkts"], errors="coerce"),
        "forward_bytes": pd.to_numeric(data["sbytes"], errors="coerce"),
        "backward_bytes": pd.to_numeric(data["dbytes"], errors="coerce"),
    })

    y = pd.to_numeric(data["label"], errors="coerce")

    if y.isna().any() or not set(y.unique()).issubset({0, 1}):
        raise ValueError("Unexpected or missing source labels.")

    return X, y.astype(int)


def load_target():
    data = pd.read_csv(TARGET_PATH)
    data.columns = data.columns.str.strip()

    required = [
        "Flow Duration",
        "Total Fwd Packets",
        "Total Backward Packets",
        "Total Length of Fwd Packets",
        "Total Length of Bwd Packets",
        "Label",
    ]

    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing CICIDS2017 columns: {missing}")

    duration = pd.to_numeric(data["Flow Duration"], errors="coerce")

    X = pd.DataFrame({
        "duration_seconds": duration.where(duration >= 0) / 1_000_000,
        "forward_packets": pd.to_numeric(
            data["Total Fwd Packets"], errors="coerce"
        ),
        "backward_packets": pd.to_numeric(
            data["Total Backward Packets"], errors="coerce"
        ),
        "forward_bytes": pd.to_numeric(
            data["Total Length of Fwd Packets"], errors="coerce"
        ),
        "backward_bytes": pd.to_numeric(
            data["Total Length of Bwd Packets"], errors="coerce"
        ),
    })

    labels = data["Label"].astype(str).str.strip().str.upper()
    unexpected = sorted(set(labels.unique()) - {"BENIGN", "DDOS"})

    if unexpected:
        raise ValueError(f"Unexpected target labels: {unexpected}")

    y = labels.map({"BENIGN": 0, "DDOS": 1})

    if y.isna().any():
        raise ValueError("Missing or unmapped target labels.")

    print(f"Target negative durations: {int((duration < 0).sum())}")
    print(f"Target missing feature values before imputation: {int(X.isna().sum().sum())}")

    return X, y.astype(int)


def main():
    X_source, y_source = load_source()
    X_target, y_target = load_target()

    if list(X_source.columns) != FEATURES or list(X_target.columns) != FEATURES:
        raise ValueError("Source and target feature order does not match.")

    X_train, X_val, y_train, y_val = train_test_split(
        X_source,
        y_source,
        test_size=0.2,
        random_state=42,
        stratify=y_source,
    )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )),
    ])

    # The pipeline is fitted only on the source training partition.
    model.fit(X_train, y_train)

    val_predictions = model.predict(X_val)
    val_probabilities = model.predict_proba(X_val)[:, 1]

    target_predictions = model.predict(X_target)
    target_probabilities = model.predict_proba(X_target)[:, 1]

    print("\n=== Dataset sizes ===")
    print(f"Source training rows: {len(X_train):,}")
    print(f"Source validation rows: {len(X_val):,}")
    print(f"Target evaluation rows: {len(X_target):,}")
    print(f"Source training labels:\n{y_train.value_counts().sort_index()}")
    print(f"Target labels:\n{y_target.value_counts().sort_index()}")

    for name, y_true, predictions, probabilities in [
        ("UNSW-NB15 source validation", y_val, val_predictions, val_probabilities),
        ("CICIDS2017 cross-dataset evaluation", y_target, target_predictions, target_probabilities),
    ]:
        print(f"\n=== {name} ===")
        print(f"Accuracy:  {accuracy_score(y_true, predictions):.4f}")
        print(f"Precision: {precision_score(y_true, predictions, zero_division=0):.4f}")
        print(f"Recall:    {recall_score(y_true, predictions, zero_division=0):.4f}")
        print(f"F1-score:  {f1_score(y_true, predictions, zero_division=0):.4f}")
        print(f"ROC-AUC:   {roc_auc_score(y_true, probabilities):.4f}")
        print("Confusion matrix [[TN, FP], [FN, TP]]:")
        print(confusion_matrix(y_true, predictions, labels=[0, 1]))
        print("\nClassification report:")
        print(classification_report(
            y_true,
            predictions,
            labels=[0, 1],
            target_names=["Benign", "Attack"],
            zero_division=0,
        ))


if __name__ == "__main__":
    main()
