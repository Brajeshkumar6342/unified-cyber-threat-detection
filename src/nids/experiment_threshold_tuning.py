
"""Tune Random Forest decision thresholds using validation data only."""

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

# Test a range of decision thresholds on validation data.
THRESHOLDS = [
    0.10, 0.15, 0.20, 0.25, 0.30,
    0.35, 0.40, 0.45, 0.50, 0.55,
    0.60, 0.65, 0.70, 0.75, 0.80,
]


def main():
    print("Loading original training data...")
    X = load_npz(DATA_DIR / "X_train.npz")
    y = pd.read_csv(DATA_DIR / "y_train.csv").iloc[:, 0].to_numpy()

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print(f"Training samples:   {X_train.shape[0]:,}")
    print(f"Validation samples: {X_val.shape[0]:,}")

    print("\nTraining baseline Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )

    start = time.time()
    model.fit(X_train, y_train)
    print(f"Training time: {time.time() - start:.2f} seconds")

    # Probability of class 1 (ATTACK).
    attack_probabilities = model.predict_proba(X_val)[:, 1]

    results = []

    print("\nEvaluating thresholds on validation data...")
    for threshold in THRESHOLDS:
        predictions = (attack_probabilities >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_val, predictions, labels=[0, 1]
        ).ravel()

        results.append(
            {
                "threshold": threshold,
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
                "roc_auc": roc_auc_score(y_val, attack_probabilities),
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
                "false_positive_rate": fp / (fp + tn) if (fp + tn) else 0.0,
            }
        )

    results_df = pd.DataFrame(results)

    output_path = REPORT_DIR / "random_forest_threshold_comparison.csv"
    results_df.to_csv(output_path, index=False)

    print("\nThreshold comparison")
    print("--------------------")
    columns = [
        "threshold",
        "precision_attack",
        "recall_attack",
        "f1_attack",
        "false_positive_rate",
        "false_positives",
        "false_negatives",
    ]
    print(results_df[columns].round(4).to_string(index=False))

    # Identify thresholds meeting both research targets.
    meets_targets = results_df[
        (results_df["recall_attack"] >= 0.98)
        & (results_df["false_positive_rate"] <= 0.05)
    ]

    print("\nResearch target check")
    print("---------------------")
    print("Target attack recall: at least 98%")
    print("Target false-positive rate: at most 5%")

    if meets_targets.empty:
        print("No tested threshold met both targets.")
        print("We may need to investigate model improvements.")
    else:
        print("Thresholds meeting both targets:")
        print(
            meets_targets[
                [
                    "threshold",
                    "recall_attack",
                    "false_positive_rate",
                    "f1_attack",
                ]
            ].round(4).to_string(index=False)
        )

    print(f"\nResults saved to: {output_path.relative_to(PROJECT_ROOT)}")
    print("The official test set was not loaded or evaluated.")


if __name__ == "__main__":
    main()
