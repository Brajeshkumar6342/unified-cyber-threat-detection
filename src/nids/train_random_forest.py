
from pathlib import Path
import json
import time

import pandas as pd
from scipy.sparse import load_npz
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
import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "nids"
MODEL_DIR = PROJECT_ROOT / "models" / "nids"
REPORT_DIR = PROJECT_ROOT / "reports" / "tables"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("Loading processed datasets...")

    X_train = load_npz(DATA_DIR / "X_train.npz")
    X_test = load_npz(DATA_DIR / "X_test.npz")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv").iloc[:, 0]
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").iloc[:, 0]

    with open(DATA_DIR / "feature_names.json", encoding="utf-8") as file:
        feature_names = json.load(file)

    assert X_train.shape[0] == len(y_train)
    assert X_test.shape[0] == len(y_test)
    assert X_train.shape[1] == X_test.shape[1] == len(feature_names)

    print("Training shape:", X_train.shape)
    print("Testing shape:", X_test.shape)
    print("\nTraining Random Forest...")

    model = RandomForestClassifier(
        n_estimators=100,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )

    start_time = time.time()
    model.fit(X_train, y_train)
    training_seconds = time.time() - start_time

    print(f"Training completed in {training_seconds:.2f} seconds.")

    print("\nEvaluating on the official test set...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "training_seconds": training_seconds,
    }

    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

    print("\nConfusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["BENIGN (0)", "ATTACK (1)"],
            zero_division=0,
        )
    )

    # Save model and metrics.
    joblib.dump(model, MODEL_DIR / "random_forest_baseline.joblib")

    pd.DataFrame([results]).to_csv(
        REPORT_DIR / "random_forest_baseline_metrics.csv",
        index=False,
    )

    print("\nSaved model to:", MODEL_DIR)
    print("Saved metrics to:", REPORT_DIR)


if __name__ == "__main__":
    main()
