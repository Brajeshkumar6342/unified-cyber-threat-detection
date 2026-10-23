
"""Run a validation-based Random Forest experiment for NIDS.

The official test set is deliberately not loaded or evaluated here.
Use validation results for model selection; reserve the official test
set for the final evaluation after decisions are complete.
"""

from pathlib import Path
import json
import time

import joblib
import pandas as pd
from scipy.sparse import load_npz, vstack
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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "nids"
MODEL_DIR = PROJECT_ROOT / "models" / "nids"
REPORT_DIR = PROJECT_ROOT / "reports" / "tables"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("Loading the original training data...")

    X = load_npz(DATA_DIR / "X_train.npz")
    y = pd.read_csv(DATA_DIR / "y_train.csv").iloc[:, 0].to_numpy()

    with open(DATA_DIR / "feature_names.json", encoding="utf-8") as file:
        feature_names = json.load(file)

    if X.shape[0] != len(y):
        raise ValueError("Feature and label counts do not match.")

    if X.shape[1] != len(feature_names):
        raise ValueError("Feature names do not match the feature matrix.")

    # Create a reproducible 80% training / 20% validation split.
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print("\nData split")
    print("----------")
    print(f"Training samples:   {X_train.shape[0]:,}")
    print(f"Validation samples: {X_val.shape[0]:,}")
    print(f"Feature count:      {X_train.shape[1]}")

    print("\nTraining Random Forest on the training split...")

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

    print("\nEvaluating on the validation split...")
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    results = {
        "experiment": "random_forest_validation_baseline",
        "random_state": 42,
        "validation_fraction": 0.20,
        "n_estimators": 100,
        "max_features": "sqrt",
        "accuracy": accuracy_score(y_val, y_pred),
        "precision_attack": precision_score(
            y_val, y_pred, zero_division=0
        ),
        "recall_attack": recall_score(
            y_val, y_pred, zero_division=0
        ),
        "f1_attack": f1_score(
            y_val, y_pred, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_val, y_prob),
        "training_seconds": training_seconds,
    }

    print("\nValidation metrics")
    print("------------------")
    for metric, value in results.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")

    print("\nConfusion matrix (rows=true, columns=predicted)")
    print("Label order: [BENIGN (0), ATTACK (1)]")
    print(confusion_matrix(y_val, y_pred, labels=[0, 1]))

    print("\nClassification report")
    print(
        classification_report(
            y_val,
            y_pred,
            labels=[0, 1],
            target_names=["BENIGN (0)", "ATTACK (1)"],
            zero_division=0,
        )
    )

    metrics_path = REPORT_DIR / "random_forest_validation_metrics.csv"
    pd.DataFrame([results]).to_csv(metrics_path, index=False)

    # Save validation predictions for later error analysis.
    predictions_path = REPORT_DIR / "random_forest_validation_predictions.csv"
    pd.DataFrame(
        {
            "actual_label": y_val,
            "predicted_label": y_pred,
            "attack_probability": y_prob,
        }
    ).to_csv(predictions_path, index=False)

    print("\nSaved validation metrics to:", metrics_path.relative_to(PROJECT_ROOT))
    print(
        "Saved validation predictions to:",
        predictions_path.relative_to(PROJECT_ROOT),
    )
    print("\nThe official test set was not loaded or evaluated.")


if __name__ == "__main__":
    main()
