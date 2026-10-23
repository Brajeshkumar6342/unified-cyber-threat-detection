
"""Compare raw and log-transformed features for cross-dataset NIDS.

Source: UNSW-NB15 training CSV only.
Target: CICIDS2017 Friday DDoS CSV.
The official UNSW-NB15 test set is not used.

E4: Four features, excluding forward_bytes.
E6: The same four features transformed using log1p.

CICIDS2017 results are diagnostic, not an independent final evaluation.
"""

from pathlib import Path

import numpy as np
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
from sklearn.preprocessing import FunctionTransformer


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
    "backward_bytes",
]


def load_data():
    source = pd.read_csv(
        SOURCE_PATH,
        usecols=["dur", "spkts", "dpkts", "dbytes", "label"],
    )

    X_source = pd.DataFrame({
        "duration_seconds": pd.to_numeric(source["dur"], errors="coerce"),
        "forward_packets": pd.to_numeric(source["spkts"], errors="coerce"),
        "backward_packets": pd.to_numeric(source["dpkts"], errors="coerce"),
        "backward_bytes": pd.to_numeric(source["dbytes"], errors="coerce"),
    })

    y_source = pd.to_numeric(source["label"], errors="coerce")

    if y_source.isna().any() or not set(y_source.unique()).issubset({0, 1}):
        raise ValueError("Unexpected or missing UNSW-NB15 labels.")

    target = pd.read_csv(TARGET_PATH)
    target.columns = target.columns.str.strip()

    required = [
        "Flow Duration",
        "Total Fwd Packets",
        "Total Backward Packets",
        "Total Length of Bwd Packets",
        "Label",
    ]
    missing = [column for column in required if column not in target.columns]

    if missing:
        raise ValueError(f"Missing CICIDS2017 columns: {missing}")

    duration = pd.to_numeric(target["Flow Duration"], errors="coerce")

    X_target = pd.DataFrame({
        "duration_seconds": duration.where(duration >= 0) / 1_000_000,
        "forward_packets": pd.to_numeric(
            target["Total Fwd Packets"], errors="coerce"
        ),
        "backward_packets": pd.to_numeric(
            target["Total Backward Packets"], errors="coerce"
        ),
        "backward_bytes": pd.to_numeric(
            target["Total Length of Bwd Packets"], errors="coerce"
        ),
    })

    labels = target["Label"].astype(str).str.strip().str.upper()

    if not set(labels.unique()).issubset({"BENIGN", "DDOS"}):
        raise ValueError("Unexpected CICIDS2017 labels.")

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
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
        "fnr": fn / (fn + tp) if (fn + tp) else 0.0,
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

    transformations = {
        "E4_raw_four_features": "raw",
        "E6_log1p_four_features": "log1p",
    }

    results = []

    for experiment, transformation in transformations.items():
        steps = [("imputer", SimpleImputer(strategy="median"))]

        if transformation == "log1p":
            # Preserve missing values and transform only nonnegative values.
            # Negative values are treated as missing for this experiment.
            def safe_log1p(X):
                X = np.asarray(X, dtype=float).copy()
                X[X < 0] = np.nan
                return np.log1p(X)

            steps.append((
                "log_transform",
                FunctionTransformer(safe_log1p, validate=False),
            ))

        steps.append((
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                max_features="sqrt",
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            ),
        ))

        model = Pipeline(steps)
        model.fit(X_train[FEATURES], y_train)

        for dataset, X_eval, y_eval in [
            ("UNSW_validation", X_val, y_val),
            ("CICIDS2017_diagnostic", X_target, y_target),
        ]:
            predictions = model.predict(X_eval[FEATURES])
            probabilities = model.predict_proba(X_eval[FEATURES])[:, 1]

            metrics = calculate_metrics(y_eval, predictions, probabilities)

            results.append({
                "experiment": experiment,
                "dataset": dataset,
                "transformation": transformation,
                **metrics,
            })

            print(
                f"{experiment} | {dataset} | "
                f"F1={metrics['f1']:.4f} | "
                f"ROC-AUC={metrics['roc_auc']:.4f} | "
                f"FPR={metrics['fpr']:.4f} | "
                f"FNR={metrics['fnr']:.4f}"
            )

    results_df = pd.DataFrame(results)

    output_dir = ROOT / "reports/tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "cross_dataset_log_transform.csv"
    results_df.to_csv(output_path, index=False)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

