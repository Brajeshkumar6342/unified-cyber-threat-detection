
"""Evaluate Random Forest thresholds with leakage-aware preprocessing.

Uses the original UNSW-NB15 training CSV only.
The official test set is never loaded or evaluated.
"""

from pathlib import Path
import time

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
RAW_TRAIN_PATH = (
    PROJECT_ROOT / "data" / "raw" / "nids" / "UNSW_NB15_training-set.csv"
)
REPORT_DIR = PROJECT_ROOT / "reports" / "tables"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "label"
EXCLUDED_COLUMNS = ["id", "label", "attack_cat"]

# Test thresholds against validation probabilities.
THRESHOLDS = [
    0.10, 0.15, 0.20, 0.25, 0.30,
    0.35, 0.40, 0.45, 0.50, 0.55,
    0.60, 0.65, 0.70, 0.75, 0.80,
]


def make_preprocessor(X_train):
    numeric_columns = X_train.select_dtypes(include="number").columns.tolist()
    categorical_columns = X_train.select_dtypes(
        exclude="number"
    ).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )


def main():
    print("Loading original training CSV only...")
    data = pd.read_csv(RAW_TRAIN_PATH, low_memory=False)

    y = data[TARGET].astype(int)
    X = data.drop(
        columns=[column for column in EXCLUDED_COLUMNS if column in data.columns]
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print(f"Training records:   {len(X_train):,}")
    print(f"Validation records: {len(X_val):,}")

    print("\nFitting preprocessing on training fold only...")
    preprocessor = make_preprocessor(X_train)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)

    print("Training Random Forest with 200 trees and leaf size 2...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_features="sqrt",
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    start = time.time()
    model.fit(X_train_processed, y_train)
    print(f"Training time: {time.time() - start:.2f} seconds")

    probabilities = model.predict_proba(X_val_processed)[:, 1]
    results = []

    for threshold in THRESHOLDS:
        predictions = (probabilities >= threshold).astype(int)
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
                "f1_attack": f1_score(y_val, predictions, zero_division=0),
                "roc_auc": roc_auc_score(y_val, probabilities),
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
                "false_positive_rate": fp / (fp + tn) if (fp + tn) else 0.0,
            }
        )

    results_df = pd.DataFrame(results)
    output_path = REPORT_DIR / "random_forest_clean_thresholds.csv"
    results_df.to_csv(output_path, index=False)

    print("\nClean threshold comparison")
    print("--------------------------")
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

    meets_targets = results_df[
        (results_df["recall_attack"] >= 0.98)
        & (results_df["false_positive_rate"] <= 0.05)
    ]

    print("\nResearch target check")
    print("---------------------")
    print("Attack recall >= 98%")
    print("False-positive rate <= 5%")

    if meets_targets.empty:
        print("No tested threshold met both targets.")
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
