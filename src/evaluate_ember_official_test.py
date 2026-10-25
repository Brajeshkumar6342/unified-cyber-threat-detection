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
MODEL_FILE = Path("models/malware/ember_random_forest_baseline.joblib")

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
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Test data not found: {DATA_FILE}")
    if not SAMPLE_FILE.exists():
        raise FileNotFoundError(f"Feature sample not found: {SAMPLE_FILE}")
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Trained model not found: {MODEL_FILE}")

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

            features = extract_features(record)
            row = np.asarray(
                [features.get(name, 0.0) for name in feature_names],
                dtype=np.float32,
            )
            row = np.nan_to_num(
                row, nan=0.0, posinf=0.0, neginf=0.0
            )

            bucket = samples[label]
            if len(bucket) < SAMPLES_PER_CLASS:
                bucket.append(row)
            else:
                replacement = int(
                    rng.integers(0, eligible_counts[label])
                )
                if replacement < SAMPLES_PER_CLASS:
                    bucket[replacement] = row

    if any(len(samples[label]) < SAMPLES_PER_CLASS for label in (0, 1)):
        raise RuntimeError(
            "The official test set does not contain enough examples "
            "for both classes."
        )

    X_test = np.asarray(
        samples[0] + samples[1], dtype=np.float32
    )
    y_test = np.asarray(
        [0] * len(samples[0]) + [1] * len(samples[1]),
        dtype=np.int8,
    )

    print("\nOfficial test-set evaluation")
    print(f"Eligible benign records: {eligible_counts[0]:,}")
    print(f"Eligible malicious records: {eligible_counts[1]:,}")
    print(f"Evaluation records: {len(y_test):,}")
    print(f"Features: {X_test.shape[1]:,}")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"\nAccuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"F1-score:  {f1_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob):.4f}")

    print("\nConfusion matrix (rows=actual, columns=predicted):")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Benign", "Malicious"],
            zero_division=0,
        )
    )


if __name__ == "__main__":
    main()
