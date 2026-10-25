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
    classification_report,
)

ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    ROOT / "data" / "raw" / "malware"
    / "ember_dataset_2018_2.tar.bz2"
    / "test_features.jsonl"
)

SAMPLE_FILE = ROOT / "data" / "processed" / "malware" / "ember_sample.npz"
MODEL_FILE = ROOT / "models" / "malware" / "ember_random_forest_full.joblib"
REPORT_FILE = ROOT / "reports" / "ember_full_random_forest_full_test.json"

BATCH_SIZE = 2000


def extract_features(record):
    features = {}

    histogram = record.get("histogram") or []
    byteentropy = record.get("byteentropy") or []

    for i in range(256):
        features[f"histogram_{i}"] = (
            float(histogram[i]) if i < len(histogram) else 0.0
        )
        features[f"byteentropy_{i}"] = (
            float(byteentropy[i]) if i < len(byteentropy) else 0.0
        )

    for section in ("general", "strings"):
        values = record.get(section) or {}

        for key, value in values.items():
            if isinstance(value, (int, float, bool)):
                features[f"{section}_{key}"] = float(value)
            else:
                features[f"{section}_{key}"] = 0.0

    return features


def main():
    start = time.time()

    for path in (DATA_FILE, SAMPLE_FILE, MODEL_FILE):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    print("Loading feature names...")
    sample = np.load(SAMPLE_FILE, allow_pickle=True)
    feature_names = list(sample["feature_names"])
    sample.close()

    print(f"Number of features: {len(feature_names)}")
    print("Loading trained Random Forest model...")
    model = joblib.load(MODEL_FILE)

    all_y_true = []
    all_y_pred = []
    all_y_prob = []

    batch_X = []
    batch_y = []
    class_counts = {0: 0, 1: 0}
    total = 0

    def evaluate_batch():
        nonlocal total, batch_X, batch_y

        if not batch_X:
            return

        X_batch = np.asarray(batch_X, dtype=np.float32)
        y_batch = np.asarray(batch_y, dtype=np.int8)

        probabilities = model.predict_proba(X_batch)[:, 1]
        predictions = (probabilities >= 0.5).astype(np.int8)

        all_y_true.extend(y_batch.tolist())
        all_y_pred.extend(predictions.tolist())
        all_y_prob.extend(probabilities.tolist())

        total += len(y_batch)

        if total % 10000 == 0 or total == 200000:
            print(f"Evaluated {total:,} records...")

        batch_X = []
        batch_y = []

    print("Starting full official test-dataset evaluation...")

    with DATA_FILE.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            record = json.loads(line)
            label = record.get("label")

            if label not in (0, 1):
                continue

            class_counts[label] += 1
            extracted = extract_features(record)

            row = [
                extracted.get(name, 0.0)
                for name in feature_names
            ]

            batch_X.append(row)
            batch_y.append(label)

            if len(batch_X) >= BATCH_SIZE:
                evaluate_batch()

    evaluate_batch()

    if total == 0:
        raise RuntimeError("No labeled test records were evaluated.")

    if set(all_y_true) != {0, 1}:
        raise RuntimeError(
            f"Expected both classes in the test dataset; found {set(all_y_true)}"
        )

    results = {
        "dataset": "EMBER 2018 official test_features.jsonl",
        "model": str(MODEL_FILE.relative_to(ROOT)),
        "features": len(feature_names),
        "evaluation_records": total,
        "class_counts": {
            "benign": class_counts[0],
            "malicious": class_counts[1],
        },
        "decision_threshold": 0.5,
        "accuracy": float(accuracy_score(all_y_true, all_y_pred)),
        "precision_malicious": float(
            precision_score(all_y_true, all_y_pred, zero_division=0)
        ),
        "recall_malicious": float(
            recall_score(all_y_true, all_y_pred, zero_division=0)
        ),
        "f1_malicious": float(
            f1_score(all_y_true, all_y_pred, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(all_y_true, all_y_prob)),
        "confusion_matrix": confusion_matrix(
            all_y_true, all_y_pred, labels=[0, 1]
        ).tolist(),
        "classification_report": classification_report(
            all_y_true,
            all_y_pred,
            labels=[0, 1],
            target_names=["Benign", "Malicious"],
            zero_division=0,
        ),
        "elapsed_seconds": round(time.time() - start, 2),
    }

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_FILE.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print("\nFULL OFFICIAL TEST RESULTS")
    print(json.dumps(results, indent=2))
    print(f"\nResults saved to: {REPORT_FILE}")


if __name__ == "__main__":
    main()
