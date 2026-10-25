import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

DATA_FILE = Path(
    "data/raw/malware/ember_dataset_2018_2.tar.bz2/test_features.jsonl"
)
SAMPLE_FILE = Path("data/processed/malware/ember_sample.npz")
MODEL_FILE = Path("models/malware/ember_random_forest_full.joblib")
REPORT_FILE = Path("reports/ember_full_random_forest_official_test.json")

SAMPLES_PER_CLASS = 10000
RANDOM_SEED = 42


def extract_features(record):
    features = {}

    for field in ("histogram", "byteentropy"):
        values = record.get(field, [])
        for index, value in enumerate(values):
            features[f"{field}_{index}"] = (
                float(value)
                if isinstance(value, (int, float))
                and not isinstance(value, bool)
                else 0.0
            )

    for field in ("general", "strings"):
        values = record.get(field, {})
        for key in sorted(values):
            value = values[key]

            if isinstance(value, bool):
                features[f"{field}_{key}"] = float(value)
            elif isinstance(value, (int, float)):
                features[f"{field}_{key}"] = float(value)
            else:
                features[f"{field}_{key}"] = 0.0

    return features


def main():
    for path in (DATA_FILE, SAMPLE_FILE, MODEL_FILE):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    with np.load(SAMPLE_FILE, allow_pickle=False) as data:
        feature_names = data["feature_names"].tolist()

    model = joblib.load(MODEL_FILE)

    samples = {0: [], 1: []}
    rng = np.random.default_rng(RANDOM_SEED)
    eligible_counts = {0: 0, 1: 0}

    print("Scanning official EMBER test file...", flush=True)

    with DATA_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            record = json.loads(line)
            label = record.get("label", -1)

            if label not in (0, 1):
                continue

            eligible_counts[label] += 1
            bucket = samples[label]

            # Reservoir sampling: choose a reproducible random sample
            # without storing the entire test dataset in memory.
            if len(bucket) < SAMPLES_PER_CLASS:
                bucket.append(record)
            else:
                index = int(rng.integers(0, eligible_counts[label]))
                if index < SAMPLES_PER_CLASS:
                    bucket[index] = record

    for label in (0, 1):
        if len(samples[label]) < SAMPLES_PER_CLASS:
            raise ValueError(
                f"Not enough test records for label {label}: "
                f"{len(samples[label])}"
            )

    print("Eligible official test records:", eligible_counts, flush=True)
    print("Sampled records per class:", SAMPLES_PER_CLASS, flush=True)

    X_rows = []
    y_true = []

    for label in (0, 1):
        for record in samples[label]:
            features = extract_features(record)
            X_rows.append([features.get(name, 0.0) for name in feature_names])
            y_true.append(label)

    X_test = np.asarray(X_rows, dtype=np.float32)
    y_true = np.asarray(y_true, dtype=np.int8)

    del X_rows, samples

    print("Evaluation sample shape:", X_test.shape, flush=True)
    print("Predicting...", flush=True)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    results = {
        "dataset": "EMBER 2018 official test_features.jsonl",
        "model": str(MODEL_FILE),
        "features": len(feature_names),
        "random_seed": RANDOM_SEED,
        "samples_per_class": SAMPLES_PER_CLASS,
        "eligible_test_records": eligible_counts,
        "evaluation_records": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_malicious": float(
            precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "recall_malicious": float(
            recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "f1_malicious": float(
            f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["Benign", "Malicious"],
            zero_division=0,
        ),
    }

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\nOfficial test results")
    print("---------------------")
    for key in (
        "accuracy",
        "precision_malicious",
        "recall_malicious",
        "f1_malicious",
        "roc_auc",
    ):
        print(f"{key}: {results[key]:.4f}")

    print("Confusion matrix:", results["confusion_matrix"])
    print(results["classification_report"])
    print(f"Results saved: {REPORT_FILE}")


if __name__ == "__main__":
    main()
