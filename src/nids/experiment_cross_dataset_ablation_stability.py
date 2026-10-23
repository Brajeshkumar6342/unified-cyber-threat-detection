
"""Test stability of cross-dataset feature-ablation results.

Source: UNSW-NB15 training CSV only.
Target: CICIDS2017 Friday DDoS CSV.
The official UNSW-NB15 test set is not used.

Compare:
E0: all five candidate common features.
E4: without forward_bytes.

Only the source split seed changes. The Random Forest seed remains fixed.
CICIDS2017 is used for exploratory diagnostic evaluation, not as an
independent untouched test set.
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

ALL_FEATURES = [
    "duration_seconds",
    "forward_packets",
    "backward_packets",
    "forward_bytes",
    "backward_bytes",
]

EXPERIMENTS = {
    "E0_baseline_all_features": ALL_FEATURES,
    "E4_without_forward_bytes": [
        feature for feature in ALL_FEATURES
        if feature != "forward_bytes"
    ],
}

SPLIT_SEEDS = [21, 42, 84, 123, 2026]
MODEL_SEED = 42


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

    required = [
        "Flow Duration",
        "Total Fwd Packets",
        "Total Backward Packets",
        "Total Length of Fwd Packets",
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
        "forward_bytes": pd.to_numeric(
            target["Total Length of Fwd Packets"], errors="coerce"
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


def metrics(y_true, predictions, probabilities):
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
        "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
        "fnr": fn / (fn + tp) if (fn + tp) else 0.0,
    }


def main():
    X_source, y_source, X_target, y_target = load_data()
    rows = []

    for split_seed in SPLIT_SEEDS:
        X_train, X_val, y_train, y_val = train_test_split(
            X_source,
            y_source,
            test_size=0.2,
            random_state=split_seed,
            stratify=y_source,
        )

        for experiment, features in EXPERIMENTS.items():
            model = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("classifier", RandomForestClassifier(
                    n_estimators=200,
                    max_features="sqrt",
                    min_samples_leaf=2,
                    random_state=MODEL_SEED,
                    n_jobs=-1,
                )),
            ])

            # Fit exclusively on the current source training partition.
            model.fit(X_train[features], y_train)

            for dataset, X_eval, y_eval in [
                ("UNSW_validation", X_val, y_val),
                ("CICIDS2017_diagnostic", X_target, y_target),
            ]:
                predictions = model.predict(X_eval[features])
                probabilities = model.predict_proba(
                    X_eval[features]
                )[:, 1]

                result = metrics(y_eval, predictions, probabilities)
                rows.append({
                    "split_seed": split_seed,
                    "model_seed": MODEL_SEED,
                    "experiment": experiment,
                    "dataset": dataset,
                    "feature_count": len(features),
                    **result,
                })

                print(
                    f"Split={split_seed} | {experiment} | {dataset} | "
                    f"F1={result['f1']:.4f} | "
                    f"ROC-AUC={result['roc_auc']:.4f} | "
                    f"FPR={result['fpr']:.4f} | "
                    f"FNR={result['fnr']:.4f}"
                )

    results = pd.DataFrame(rows)

    output_dir = ROOT / "reports/tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "cross_dataset_ablation_stability.csv"
    results.to_csv(output_path, index=False)

    target_results = results[
        results["dataset"] == "CICIDS2017_diagnostic"
    ]

    summary = target_results.groupby("experiment").agg(
        mean_f1=("f1", "mean"),
        std_f1=("f1", "std"),
        mean_roc_auc=("roc_auc", "mean"),
        std_roc_auc=("roc_auc", "std"),
        mean_fpr=("fpr", "mean"),
        std_fpr=("fpr", "std"),
        mean_fnr=("fnr", "mean"),
        std_fnr=("fnr", "std"),
    )

    print("\n=== CICIDS2017 diagnostic summary across source splits ===")
    print(summary.round(4).to_string())
    print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    main()
