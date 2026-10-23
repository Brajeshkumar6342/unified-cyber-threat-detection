
"""Exploratory feature-ablation study for cross-dataset NIDS transfer.

Source: UNSW-NB15 training CSV only.
Target: CICIDS2017 Friday DDoS CSV.
The official UNSW-NB15 test set is not used.

This experiment compares the existing five-feature baseline with
five configurations, each excluding one feature.
CICIDS2017 results are diagnostic because its labels have already
been used in earlier exploratory analyses.
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

EXPERIMENTS = {
    "E0_baseline_all_features": FEATURES,
    "E1_without_duration": [
        f for f in FEATURES if f != "duration_seconds"
    ],
    "E2_without_forward_packets": [
        f for f in FEATURES if f != "forward_packets"
    ],
    "E3_without_backward_packets": [
        f for f in FEATURES if f != "backward_packets"
    ],
    "E4_without_forward_bytes": [
        f for f in FEATURES if f != "forward_bytes"
    ],
    "E5_without_backward_bytes": [
        f for f in FEATURES if f != "backward_bytes"
    ],
}


def load_data():
    source = pd.read_csv(
        SOURCE_PATH,
        usecols=["dur", "spkts", "dpkts", "sbytes", "dbytes", "label"],
    )

    X_source = pd.DataFrame({
        "duration_seconds": pd.to_numeric(source["dur"], errors="coerce"),
        "forward_packets": pd.to_numeric(source["spkts"], errors="coerce"),
        "backward_packets": pd.to_numeric(source["dpkts"], errors="coerce"),
        "forward_bytes": pd.to_numeric(source["sbytes"], errors="coerce"),
        "backward_bytes": pd.to_numeric(source["dbytes"], errors="coerce"),
    })

    y_source = pd.to_numeric(source["label"], errors="coerce")

    if y_source.isna().any() or not set(y_source.unique()).issubset({0, 1}):
        raise ValueError("Unexpected or missing UNSW-NB15 labels.")

    target = pd.read_csv(TARGET_PATH)
    target.columns = target.columns.str.strip()

    duration = pd.to_numeric(target["Flow Duration"], errors="coerce")

    X_target = pd.DataFrame({
        "duration_seconds": duration.where(duration >= 0) / 1_000_000,
        "forward_packets": pd.to_numeric(
            target["Total Fwd Packets"], errors="coerce"
        ),
        "backward_packets": pd.to_numeric(
            target["Total Backward Packets"], errors="coerce"
        ),
        "forward_bytes": pd.to_numeric(
            target["Total Length of Fwd Packets"], errors="coerce"
        ),
        "backward_bytes": pd.to_numeric(
            target["Total Length of Bwd Packets"], errors="coerce"
        ),
    })

    labels = target["Label"].astype(str).str.strip().str.upper()

    if not set(labels.unique()).issubset({"BENIGN", "DDOS"}):
        raise ValueError("Unexpected CICIDS2017 target labels.")

    y_target = labels.map({"BENIGN": 0, "DDOS": 1})

    if y_target.isna().any():
        raise ValueError("Missing or unmapped CICIDS2017 labels.")

    return X_source, y_source.astype(int), X_target, y_target.astype(int)


def calculate_metrics(y_true, predictions, probabilities):
    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    return {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "fnr": fn / (fn + tp) if fn + tp else 0.0,
    }


def main():
    X_source, y_source, X_target, y_target = load_data()

    X_train, X_val, y_train, y_val = train_test_split(
        X_source,
        y_source,
        test_size=0.2,
        random_state=42,
        stratify=y_source,
    )

    results = []

    for experiment_name, selected_features in EXPERIMENTS.items():
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

        # Fit only on the UNSW-NB15 training partition.
        model.fit(X_train[selected_features], y_train)

        for dataset_name, X_eval, y_eval in [
            ("UNSW_validation", X_val, y_val),
            ("CICIDS2017_diagnostic", X_target, y_target),
        ]:
            predictions = model.predict(X_eval[selected_features])
            probabilities = model.predict_proba(
                X_eval[selected_features]
            )[:, 1]

            metrics = calculate_metrics(y_eval, predictions, probabilities)

            results.append({
                "experiment": experiment_name,
                "dataset": dataset_name,
                "features": ", ".join(selected_features),
                **metrics,
            })

            print(
                f"{experiment_name} | {dataset_name} | "
                f"F1={metrics['f1']:.4f} | "
                f"ROC-AUC={metrics['roc_auc']:.4f} | "
                f"FPR={metrics['fpr']:.4f} | "
                f"FNR={metrics['fnr']:.4f}"
            )

    results_df = pd.DataFrame(results)

    output_dir = ROOT / "reports/tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "cross_dataset_feature_ablation.csv"
    results_df.to_csv(output_path, index=False)

    print("\n=== Experiment complete ===")
    print(results_df[
        ["experiment", "dataset", "f1", "roc_auc", "fpr", "fnr"]
    ].to_string(index=False))
    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()
