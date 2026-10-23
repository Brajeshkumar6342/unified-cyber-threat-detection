
"""Compare Random Forest configurations with leakage-aware preprocessing.

Uses only the original UNSW-NB15 training CSV.
The official test set is never loaded in this experiment.
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

CONFIGURATIONS = [
    {
        "name": "RF_100_default",
        "n_estimators": 100,
        "max_features": "sqrt",
        "min_samples_leaf": 1,
        "class_weight": None,
    },
    {
        "name": "RF_200_leaf2",
        "n_estimators": 200,
        "max_features": "sqrt",
        "min_samples_leaf": 2,
        "class_weight": None,
    },
    {
        "name": "RF_100_leaf2_balanced",
        "n_estimators": 100,
        "max_features": "sqrt",
        "min_samples_leaf": 2,
        "class_weight": "balanced",
    },
]


def make_preprocessor(X_train):
    """Build a preprocessor using columns from the training fold."""
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

    if TARGET not in data.columns:
        raise ValueError(f"Missing target column: {TARGET}")

    y = data[TARGET].astype(int)
    X = data.drop(
        columns=[column for column in EXCLUDED_COLUMNS if column in data.columns]
    )

    # Split before fitting preprocessing.
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print(f"Training records:   {len(X_train):,}")
    print(f"Validation records: {len(X_val):,}")
    print("The official test set is not used.\n")

    results = []

    for config in CONFIGURATIONS:
        name = config["name"]
        parameters = {
            key: value for key, value in config.items() if key != "name"
        }

        print(f"Running {name}...")

        # Fit a fresh preprocessor on the training fold for each experiment.
        preprocessor = make_preprocessor(X_train)
        X_train_processed = preprocessor.fit_transform(X_train)
        X_val_processed = preprocessor.transform(X_val)

        model = RandomForestClassifier(
            **parameters,
            random_state=42,
            n_jobs=-1,
        )

        start = time.time()
        model.fit(X_train_processed, y_train)
        training_seconds = time.time() - start

        predictions = model.predict(X_val_processed)
        probabilities = model.predict_proba(X_val_processed)[:, 1]

        tn, fp, fn, tp = confusion_matrix(
            y_val, predictions, labels=[0, 1]
        ).ravel()

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
            "f1_attack": f1_score(y_val, predictions, zero_division=0),
            "roc_auc": roc_auc_score(y_val, probabilities),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "false_positive_rate": fp / (fp + tn) if (fp + tn) else 0.0,
            "training_seconds": training_seconds,
            "processed_feature_count": X_train_processed.shape[1],
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
    output_path = REPORT_DIR / "random_forest_clean_comparison.csv"
    results_df.to_csv(output_path, index=False)

    print("\nClean validation comparison")
    print("---------------------------")
    columns = [
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
    print(results_df[columns].round(4).to_string(index=False))

    targets = results_df[
        (results_df["recall_attack"] >= 0.98)
        & (results_df["false_positive_rate"] <= 0.05)
    ]

    print("\nResearch targets:")
    print("Attack recall >= 98%")
    print("False-positive rate <= 5%")

    if targets.empty:
        print("No tested configuration met both targets.")
    else:
        print("Configurations meeting both targets:")
        print(targets[["experiment", "recall_attack", "false_positive_rate"]])

    print(f"\nResults saved to: {output_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
