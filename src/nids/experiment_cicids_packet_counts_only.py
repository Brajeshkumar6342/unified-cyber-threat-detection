"""Evaluate CICIDS2017 using only forward/backward packet counts."""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    ROOT
    / "data"
    / "raw"
    / "nids"
    / "cicids2017"
    / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
)


def main():
    data = pd.read_csv(DATA_PATH)
    data.columns = data.columns.str.strip()

    X = pd.DataFrame({
        "forward_packets": pd.to_numeric(
            data["Total Fwd Packets"], errors="coerce"
        ),
        "backward_packets": pd.to_numeric(
            data["Total Backward Packets"], errors="coerce"
        ),
    })

    labels = data["Label"].astype(str).str.strip().str.upper()

    unexpected = sorted(set(labels.unique()) - {"BENIGN", "DDOS"})
    if unexpected:
        raise ValueError(f"Unexpected labels found: {unexpected}")

    y = labels.map({"BENIGN": 0, "DDOS": 1}).astype(int)

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

    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()

    print("=== CICIDS2017: Packet Counts Only ===")
    print("Features: forward_packets, backward_packets")
    print(f"Training samples: {len(y_train)}")
    print(f"Test samples: {len(y_test)}")
    print(f"Accuracy:  {accuracy_score(y_test, predictions):.4f}")
    print(
        f"Precision: {precision_score(y_test, predictions, zero_division=0):.4f}"
    )
    print(
        f"Recall:    {recall_score(y_test, predictions, zero_division=0):.4f}"
    )
    print(f"F1-score:  {f1_score(y_test, predictions, zero_division=0):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, probabilities):.4f}")
    print(f"Confusion matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")


if __name__ == "__main__":
    main()
    