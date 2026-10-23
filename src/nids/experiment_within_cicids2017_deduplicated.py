
"""Within-CICIDS2017 experiment after removing exact duplicate records.

Exact duplicates are removed before the stratified train/test split.
This is a diagnostic experiment, not a fully independent-flow evaluation.
"""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
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


def main():
    data = pd.read_csv(TARGET_PATH)
    data.columns = data.columns.str.strip()

    original_rows = len(data)
    original_duplicates = int(data.duplicated().sum())

    # Remove complete duplicate rows, including their labels.
    data = data.drop_duplicates().reset_index(drop=True)

    print(f"Original rows: {original_rows:,}")
    print(f"Exact duplicates removed: {original_duplicates:,}")
    print(f"Rows after deduplication: {len(data):,}")

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

    y = y.astype(int)

    print(f"\nClass counts after deduplication:\n{y.value_counts().sort_index()}")
    print(f"Missing feature values: {int(X.isna().sum().sum())}")

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

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("\n=== Deduplicated within-CICIDS2017 evaluation ===")
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


if __name__ == "__main__":
    main()

