
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "nids"
    / "UNSW_NB15_training-set.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "reports" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# Proposed research targets, not universal security standards.
TARGET_RECALL = 0.98
TARGET_FPR = 0.05


def main():
    print("Loading raw training dataset...")
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["label", "id", "attack_cat"])
    y = df["label"].astype(int)

    numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = X.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    # Split before fitting preprocessing.
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"Training rows: {len(X_train):,}")
    print(f"Validation rows: {len(X_val):,}")

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ]
    )

    print("Fitting preprocessing on training fold...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)

    model = RandomForestClassifier(
        n_estimators=200,
        max_features="sqrt",
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    print("\nTraining balanced Random Forest...")
    start_time = time.time()
    model.fit(X_train_processed, y_train)
    training_seconds = time.time() - start_time

    # Calculate probabilities once, then evaluate different thresholds.
    attack_probabilities = model.predict_proba(X_val_processed)[:, 1]

    thresholds = np.round(np.arange(0.10, 0.901, 0.025), 3)
    results = []

    for threshold in thresholds:
        predictions = (attack_probabilities >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_val,
            predictions,
            labels=[0, 1],
        ).ravel()

        fpr = fp / (fp + tn) if (fp + tn) else 0.0

        results.append(
            {
                "threshold": float(threshold),
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
                "false_positive_rate": fpr,
                "meets_recall_target": (
                    recall_score(y_val, predictions, zero_division=0)
                    >= TARGET_RECALL
                ),
                "meets_fpr_target": fpr <= TARGET_FPR,
            }
        )

    results_df = pd.DataFrame(results)

    output_path = OUTPUT_DIR / "balanced_rf_threshold_results.csv"
    results_df.to_csv(output_path, index=False)

    print(f"\nTraining time: {training_seconds:.2f} seconds")

    print("\n=== THRESHOLD RESULTS ===")
    print(
        results_df[
            [
                "threshold",
                "precision_attack",
                "recall_attack",
                "f1_attack",
                "false_positive_rate",
                "false_positives",
                "false_negatives",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    eligible = results_df[
        (results_df["recall_attack"] >= TARGET_RECALL)
        & (results_df["false_positive_rate"] <= TARGET_FPR)
    ]

    if eligible.empty:
        print(
            "\nNo tested threshold met both proposed targets: "
            f"recall >= {TARGET_RECALL:.0%} and "
            f"FPR <= {TARGET_FPR:.0%}."
        )
    else:
        best = eligible.sort_values(
            ["f1_attack", "false_positive_rate"],
            ascending=[False, True],
        ).iloc[0]

        print("\nBest eligible threshold by F1-score:")
        print(best.to_string())

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
