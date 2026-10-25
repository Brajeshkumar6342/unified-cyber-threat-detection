import gc
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
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed" / "malware" / "full"
MODEL_PATH = ROOT / "models" / "malware" / "ember_random_forest_full.joblib"
OUTPUT_PATH = ROOT / "reports" / "ember_threshold_analysis.json"
CSV_PATH = ROOT / "reports" / "ember_threshold_analysis.csv"

RANDOM_STATE = 42
VALIDATION_SIZE = 0.20
BATCH_SIZE = 2000

THRESHOLDS = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]


def main():
    start_time = time.time()

    print("Loading memory-mapped dataset...")
    X = np.load(DATA_DIR / "X.npy", mmap_mode="r")
    y = np.load(DATA_DIR / "y.npy", mmap_mode="r")

    print(f"Dataset shape: {X.shape}")
    print(f"Label shape: {y.shape}")

    indices = np.arange(len(y), dtype=np.int32)
    _, val_idx = train_test_split(
        indices,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    del indices
    gc.collect()

    y_true = np.asarray(y[val_idx], dtype=np.int8)

    print(f"Reconstructed validation records: {len(val_idx):,}")
    print("Loading trained model...")
    model = joblib.load(MODEL_PATH)

    y_prob = np.empty(len(val_idx), dtype=np.float32)

    print("Predicting validation probabilities in batches...")
    for start in range(0, len(val_idx), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(val_idx))
        X_batch = np.asarray(X[val_idx[start:end]], dtype=np.float32)
        y_prob[start:end] = model.predict_proba(X_batch)[:, 1]
        del X_batch

        if end % 10000 == 0 or end == len(val_idx):
            print(f"  Predicted {end:,}/{len(val_idx):,}")

    del X
    gc.collect()

    results = []

    for threshold in THRESHOLDS:
        y_pred = (y_prob >= threshold).astype(np.int8)

        tn, fp, fn, tp = confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).ravel()

        results.append({
            "threshold": threshold,
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
            "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
            "false_negative_rate": float(fn / (fn + tp)) if fn + tp else 0.0,
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        })

    report = {
        "dataset": "EMBER 2018 labeled training files; reconstructed validation split",
        "validation_records": int(len(y_true)),
        "features": int(X.shape[1]) if "X" in locals() else 531,
        "random_state": RANDOM_STATE,
        "validation_fraction": VALIDATION_SIZE,
        "thresholds": results,
        "validation_roc_auc": float(roc_auc_score(y_true, y_prob)),
        "elapsed_seconds": round(time.time() - start_time, 2),
        "note": (
            "Validation split reconstructed using the original random_state=42 "
            "and stratified 80/20 split. Threshold selection must use validation "
            "data only; the official test set remains reserved for final evaluation."
        ),
    }

    OUTPUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    import csv
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print("\nTHRESHOLD ANALYSIS")
    print(
        f"{'Threshold':>9} {'Accuracy':>9} {'Precision':>10} "
        f"{'Recall':>9} {'F1':>9} {'FPR':>9} {'FNR':>9}"
    )

    for row in results:
        print(
            f"{row['threshold']:>9.2f} "
            f"{row['accuracy']:>9.4f} "
            f"{row['precision_malicious']:>10.4f} "
            f"{row['recall_malicious']:>9.4f} "
            f"{row['f1_malicious']:>9.4f} "
            f"{row['false_positive_rate']:>9.4f} "
            f"{row['false_negative_rate']:>9.4f}"
        )

    print(f"\nValidation ROC-AUC: {report['validation_roc_auc']:.4f}")
    print(f"JSON report: {OUTPUT_PATH}")
    print(f"CSV report: {CSV_PATH}")
    print(f"Elapsed seconds: {report['elapsed_seconds']}")


if __name__ == "__main__":
    main()
