
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


ROOT = Path(__file__).resolve().parents[2]

SOURCE_PATH = ROOT / "data" / "raw" / "nids" / "UNSW_NB15_training-set.csv"
TARGET_PATH = (
    ROOT
    / "data"
    / "raw"
    / "nids"
    / "cicids2017"
    / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
)

OUTPUT_PATH = ROOT / "reports" / "tables" / "cross_dataset_e7_comparison.csv"

SEEDS = [21, 42, 84, 123, 2026]

FEATURES = {
    "E4": [
        "duration_seconds",
        "forward_packets",
        "backward_packets",
        "backward_bytes",
    ],
    "E7": [
        "duration_seconds",
        "forward_packets",
        "backward_packets",
    ],
}

SOURCE_MAPPING = {
    "duration_seconds": "dur",
    "forward_packets": "spkts",
    "backward_packets": "dpkts",
    "backward_bytes": "dbytes",
}

TARGET_MAPPING = {
    "duration_seconds": "Flow Duration",
    "forward_packets": "Total Fwd Packets",
    "backward_packets": "Total Backward Packets",
    "backward_bytes": "Total Length of Bwd Packets",
}


def load_data():
    source = pd.read_csv(SOURCE_PATH)
    target = pd.read_csv(TARGET_PATH, low_memory=False)
    target.columns = target.columns.str.strip()

    X_source = pd.DataFrame({
        common: pd.to_numeric(source[column], errors="coerce")
        for common, column in SOURCE_MAPPING.items()
    })

    y_source = pd.to_numeric(source["label"], errors="coerce")

    valid_source = y_source.isin([0, 1])
    X_source = X_source.loc[valid_source].reset_index(drop=True)
    y_source = y_source.loc[valid_source].astype(int).reset_index(drop=True)

    X_target = pd.DataFrame({
        common: pd.to_numeric(target[column], errors="coerce")
        for common, column in TARGET_MAPPING.items()
    })

    # CICIDS Flow Duration is recorded in microseconds.
    X_target["duration_seconds"] = X_target["duration_seconds"] / 1_000_000

    # Negative durations are invalid; retain zero-duration records.
    X_target.loc[
        X_target["duration_seconds"] < 0, "duration_seconds"
    ] = np.nan

    labels = target["Label"].astype(str).str.strip().str.upper()
    valid_target = labels.isin(["BENIGN", "DDOS"])

    X_target = X_target.loc[valid_target].reset_index(drop=True)
    y_target = (
        labels.loc[valid_target].eq("DDOS").astype(int).reset_index(drop=True)
    )

    return X_source, y_source, X_target, y_target


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
        "fpr": fp / (fp + tn) if fp + tn else np.nan,
        "fnr": fn / (fn + tp) if fn + tp else np.nan,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def main():
    X_source, y_source, X_target, y_target = load_data()
    records = []

    for split_seed in SEEDS:
        X_train, X_val, y_train, y_val = train_test_split(
            X_source,
            y_source,
            test_size=0.20,
            random_state=split_seed,
            stratify=y_source,
        )

        for experiment, feature_list in FEATURES.items():
            model = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_features="sqrt",
                        min_samples_leaf=2,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ])

            model.fit(X_train[feature_list], y_train)

            for evaluation, X_eval, y_eval in [
                ("source_validation", X_val, y_val),
                ("cicids_transfer", X_target, y_target),
            ]:
                predictions = model.predict(X_eval[feature_list])
                probabilities = model.predict_proba(
                    X_eval[feature_list]
                )[:, 1]

                metrics = calculate_metrics(
                    y_eval, predictions, probabilities
                )

                records.append({
                    "experiment": experiment,
                    "features": ",".join(feature_list),
                    "split_seed": split_seed,
                    "evaluation": evaluation,
                    **metrics,
                })

        print(f"Completed source split seed {split_seed}.")

    results = pd.DataFrame(records)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)

    summary = (
        results.groupby(["experiment", "evaluation"])[
            ["f1", "roc_auc", "fpr", "fnr"]
        ]
        .agg(["mean", "std"])
        .round(4)
    )

    print("\nE4 versus E7: mean and standard deviation across five splits")
    print(summary.to_string())
    print(f"\nDetailed results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

