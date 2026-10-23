
"""Within-dataset NIDS experiment on CICIDS2017 Friday DDoS data.

Training and testing use separate stratified partitions of CICIDS2017.
The UNSW-NB15 official test set is not used.
"""

from pathlib import Path

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

TARGET_PATH = (
    ROOT
    / "data"
    / "raw"
    / "nids"
    / "cicids2017"
    / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
)

FEATURES = [
    "duration_seconds",
    "forward_packets",
    "backward_packets",
    "forward_bytes",
    "backward_bytes",
]


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

    print(f"Dataset rows: {len(data):,}")
    print(f"Negative durations: {int((duration < 0).sum())}")
    print(f"Missing feature values: {int(X.isna().sum().sum())}")
    print(f"Class counts:\n{y.value_counts().sort_index()}")

    return X, y.astype(int)


def main():
    X, y = load_target()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
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

    # Fit preprocessing and model using the training partition only.
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("\n=== Within-CICIDS2017 evaluation ===")
    print(f"Training rows: {len(X_train):,}")
    print(f"Testing rows: {len(X_test):,}")
    print(f"Accuracy:  {accuracy_score(y_test, predictions):.4f}")
    print(
        f"Precision: {precision_score(y_test, predictions, zero_division=0):.4f}"
    )
    print(f"Recall:    {recall_score(y_test, predictions, zero_division=0):.4f}")
    print(f"F1-score:  {f1_score(y_test, predictions, zero_division=0):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, probabilities):.4f}")

    print("\nConfusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Benign", "DDoS"],
        zero_division=0,
    ))


if __name__ == "__main__":
    main()

