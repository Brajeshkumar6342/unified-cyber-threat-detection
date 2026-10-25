import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]

FEATURE_FILE = ROOT / "data" / "processed" / "malware" / "ember_sample.npz"
TEST_FILE = (
    ROOT / "data" / "raw" / "malware"
    / "ember_dataset_2018_2.tar.bz2" / "test_features.jsonl"
)
BASELINE_MODEL = ROOT / "models" / "malware" / "ember_random_forest_baseline.joblib"
FULL_MODEL = ROOT / "models" / "malware" / "ember_random_forest_full.joblib"
OUTPUT_FILE = ROOT / "reports" / "ember_same_test_comparison.json"

BATCH_SIZE = 2000


def extract_features(record, feature_names):
    general = record.get("general") or {}
    strings = record.get("strings") or {}
    histogram = record.get("histogram") or []
    byteentropy = record.get("byteentropy") or []

    values = {}

    for i in range(256):
        values[f"histogram_{i}"] = (
            float(histogram[i])
            if i < len(histogram) and isinstance(histogram[i], (int, float))
            else 0.0
        )
        values[f"byteentropy_{i}"] = (
            float(byteentropy[i])
            if i < len(byteentropy) and isinstance(byteentropy[i], (int, float))
            else 0.0
        )

    for key, value in general.items():
        if isinstance(value, (int, float, bool)):
            values[f"general_{key}"] = float(value)

    for key, value in strings.items():
        if isinstance(value, (int, float, bool)):
            values[f"strings_{key}"] = float(value)

    return [
        values.get(name, 0.0)
        for name in feature_names
    ]


def calculate_metrics(y_true, y_pred, y_prob):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_malicious": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall_malicious": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "f1_malicious": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).tolist(),
    }


def main():
    start = time.time()

    print("Loading feature names...")
    with np.load(FEATURE_FILE, allow_pickle=True) as data:
        feature_names = data["feature_names"].tolist()

    print(f"Number of features: {len(feature_names)}")

    print("Loading baseline model...")
    baseline = joblib.load(BASELINE_MODEL)

    print("Loading full-data model...")
    full_model = joblib.load(FULL_MODEL)

    y_true = []
    baseline_predictions = []
    baseline_probabilities = []
    full_predictions = []
    full_probabilities = []

    class_counts = {"benign": 0, "malicious": 0, "skipped": 0}
    batch_X = []
    batch_y = []
    processed = 0

    def evaluate_batch():
        if not batch_X:
            return

        X = np.asarray(batch_X, dtype=np.float32)
        y = np.asarray(batch_y, dtype=np.int8)

        base_prob = baseline.predict_proba(X)[:, 1]
        full_prob = full_model.predict_proba(X)[:, 1]

        baseline_predictions.extend((base_prob >= 0.5).astype(np.int8))
        baseline_probabilities.extend(base_prob.tolist())
        full_predictions.extend((full_prob >= 0.5).astype(np.int8))
        full_probabilities.extend(full_prob.tolist())
        y_true.extend(y.tolist())

    print("Reading the complete official test file...")
    with TEST_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            label = record.get("label")

            if label not in (0, 1):
                class_counts["skipped"] += 1
                continue

            class_counts["benign" if label == 0 else "malicious"] += 1
            batch_X.append(extract_features(record, feature_names))
            batch_y.append(label)
            processed += 1

            if len(batch_X) >= BATCH_SIZE:
                evaluate_batch()
                batch_X.clear()
                batch_y.clear()

            if processed and processed % 20000 == 0:
                print(f"Processed {processed:,} labeled records...")

        evaluate_batch()

    if set(y_true) != {0, 1}:
        raise ValueError("Expected both benign (0) and malicious (1) labels.")

    print("Calculating metrics...")
    results = {
        "dataset": "EMBER 2018 official test_features.jsonl",
        "evaluation_records": len(y_true),
        "class_counts": class_counts,
        "features": len(feature_names),
        "decision_threshold": 0.5,
        "baseline_model": str(BASELINE_MODEL.relative_to(ROOT)),
        "full_model": str(FULL_MODEL.relative_to(ROOT)),
        "baseline": calculate_metrics(
            y_true, baseline_predictions, baseline_probabilities
        ),
        "full_data_random_forest": calculate_metrics(
            y_true, full_predictions, full_probabilities
        ),
        "elapsed_seconds": round(time.time() - start, 2),
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\nSAME-TEST-SET COMPARISON")
    print(json.dumps(results, indent=2))
    print(f"\nResults saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
