
"""Compare Random Forest configurations using a fixed validation split.

The official test set is not loaded or used in this experiment.
"""

from pathlib import Path
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
)
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "nids"
REPORT_DIR = PROJECT_ROOT / "reports" / "tables"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


CONFIGURATIONS = [
    {
        "name": "RF_100_sqrt",
        "n_estimators": 100,
        "max_features": "sqrt",
        "min_samples_leaf": 1,
        "class_weight": None,
    },
    {
        "name": "RF_200_sqrt_leaf2",
        "n_estimators": 200,
        "max_features": "sqrt",
        "min_samples_leaf": 2,
        "class_weight": None,
    },
    {
        "name": "RF_100_sqrt_balanced",
        "n_estimators": 100,
        "max_features": "sqrt",
        "min_samples_leaf": 1,
        "class_weight": "balanced",
    },
]


def main():
    print("Loading original training data...")
    X = load_npz(DATA_DIR / "X_train.npz")
    y = pd.read_csv(DATA_DIR / "y_train.csv").iloc[:, 0].to_numpy()

    if X.shape[0] != len(y):
        raise ValueError("Feature and label counts do not match.")

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print(f"Training samples:   {X_train.shape[0]:,}")
    print(f"Validation samples: {X_val.shape[0]:,}")
    print("\nComparing configurations...")
    print("The official test set is not used.\n")

    results = []

    for config in CONFIGURATIONS:
        name = config["name"]
        parameters = {key: value for key, value in config.items() if key != "name"}

        print(f"Training {name}...")

        model = RandomForestClassifier(
            **parameters,
            random_state=42,
            n_jobs=-1,
        )

        start = time.time()
        model.fit(X_train, y_train)
        training_seconds = time.time() - start

        predictions = model.predict(X_val)
        probabilities = model.predict_proba(X_val)[:, 1]

        matrix = confusion_matrix(y_val, predictions, labels=[0, 1])
        tn, fp, fn, tp = matrix.ravel()

        result = {
            "experiment": name,
            **parameters,
            "accuracy": accuracy_score(y_val, predictions),
            "precision_attack": precision_score(
                y_val, predictions, zero_division=0
            ),
            "recall_attack": recall_score(
                y_val, predictions, zero_division=0
            ),
            "f1_attack": f1_score(
                y_val, predictions, zero_division=0
            ),
            "roc_auc": roc_auc_score(y_val, probabilities),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "false_positive_rate": fp / (fp + tn) if (fp + tn) else 0.0,
            "training_seconds": training_seconds,
        }
        results.append(result)

        print(
            f"  F1={result['f1_attack']:.4f} | "
            f"Recall={result['recall_attack']:.4f} | "
            f"FPR={result['false_positive_rate']:.4f} | "
            f"ROC-AUC={result['roc_auc']:.4f} | "
            f"Time={training_seconds:.2f}s"
        )

    results_df = pd.DataFrame(results)
    output_path = REPORT_DIR / "random_forest_configuration_comparison.csv"
    results_df.to_csv(output_path, index=False)

    print("\nFull validation comparison")
    print("--------------------------")
    display_columns = [
        "experiment",
        "accuracy",
        "precision_attack",
        "recall_attack",
        "f1_attack",
        "roc_auc",
        "false_positive_rate",
        "false_positives",
        "false_negatives",
        "training_seconds",
    ]
    print(results_df[display_columns].round(4).to_string(index=False))

    print(f"\nResults saved to: {output_path.relative_to(PROJECT_ROOT)}")
    print("\nUse validation results for model selection, not final performance claims.")


if __name__ == "__main__":
    main()
