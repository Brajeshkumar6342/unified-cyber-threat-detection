import gc
import json
import time
from pathlib import Path

import numpy as np
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split

DATA_DIR = Path("data/processed/malware/full")
MODEL_DIR = Path("models/malware")
REPORT_DIR = Path("reports")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_ESTIMATORS = 100
MAX_DEPTH = 20
MIN_SAMPLES_LEAF = 2
MAX_FEATURES = "sqrt"

def main():
    start_time = time.time()

    X = np.load(DATA_DIR / "X.npy", mmap_mode="r")
    y = np.load(DATA_DIR / "y.npy", mmap_mode="r")

    print("Dataset:", X.shape, flush=True)
    print("Labels:", y.shape, flush=True)

    # Store only the split indices in RAM; do not copy the entire matrix.
    indices = np.arange(len(y), dtype=np.int32)
    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    del indices
    gc.collect()

    print(f"Training records: {len(train_idx):,}", flush=True)
    print(f"Validation records: {len(val_idx):,}", flush=True)

    # Materialize only the training partition. This may still require
    # several GB of RAM; close memory-intensive applications first.
    X_train = np.asarray(X[train_idx], dtype=np.float32)
    y_train = np.asarray(y[train_idx], dtype=np.int8)
    del train_idx
    gc.collect()

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        max_features=MAX_FEATURES,
        random_state=RANDOM_STATE,
        n_jobs=2,
        class_weight="balanced_subsample",
    )

    print("Training Random Forest...", flush=True)
    model.fit(X_train, y_train)

    del X_train, y_train
    gc.collect()

    # Predict validation data in chunks to reduce peak memory usage.
    y_pred = np.empty(len(val_idx), dtype=np.int8)
    y_prob = np.empty(len(val_idx), dtype=np.float32)
    batch_size = 2000

    print("Predicting validation records in batches...", flush=True)
    for start in range(0, len(val_idx), batch_size):
        end = min(start + batch_size, len(val_idx))
        batch_idx = val_idx[start:end]
        X_batch = np.asarray(X[batch_idx], dtype=np.float32)
        y_prob[start:end] = model.predict_proba(X_batch)[:, 1]
        y_pred[start:end] = model.predict(X_batch)
        del X_batch

        if end % 10000 == 0 or end == len(val_idx):
            print(f"  Validated {end:,}/{len(val_idx):,}", flush=True)

    y_true = np.asarray(y[val_idx], dtype=np.int8)

    results = {
        "dataset": "EMBER 2018 labeled training files",
        "records": int(len(y)),
        "features": int(X.shape[1]),
        "train_records": int(len(y) - len(val_idx)),
        "validation_records": int(len(val_idx)),
        "n_estimators": N_ESTIMATORS,
        "max_depth": MAX_DEPTH,
        "min_samples_leaf": MIN_SAMPLES_LEAF,
        "max_features": MAX_FEATURES,
        "random_state": RANDOM_STATE,
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
            y_true, y_pred,
            target_names=["Benign", "Malicious"],
            zero_division=0,
        ),
        "elapsed_seconds": round(time.time() - start_time, 2),
    }

    model_path = MODEL_DIR / "ember_random_forest_full.joblib"
    report_path = REPORT_DIR / "ember_full_random_forest_results.json"

    print("Saving model...", flush=True)
    dump(model, model_path, compress=3)

    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\nValidation results")
    print("------------------")
    for key in (
        "accuracy", "precision_malicious", "recall_malicious",
        "f1_malicious", "roc_auc",
    ):
        print(f"{key}: {results[key]:.4f}")
    print("Confusion matrix:", results["confusion_matrix"])
    print(results["classification_report"])
    print(f"Model saved: {model_path}")
    print(f"Results saved: {report_path}")
    print(f"Elapsed seconds: {results['elapsed_seconds']}")

if __name__ == "__main__":
    main()
