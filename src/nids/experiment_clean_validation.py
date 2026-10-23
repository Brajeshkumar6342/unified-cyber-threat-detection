
"""Leakage-aware Random Forest experiment on raw UNSW-NB15 training data.

The official test set is never loaded in this experiment.
Preprocessing is fitted only on the training fold.
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


def main():
    print("Loading the original training CSV only...")
    data = pd.read_csv(RAW_TRAIN_PATH, low_memory=False)

    if TARGET not in data.columns:
        raise ValueError(f"Missing target column: {TARGET}")

    y = data[TARGET].astype(int)
    X = data.drop(
        columns=[column for column in EXCLUDED_COLUMNS if column in data.columns]
    )

    # Split raw records BEFORE fitting any preprocessing.
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    numeric_columns = X_train.select_dtypes(include="number").columns.tolist()
    categorical_columns = X_train.select_dtypes(
        exclude="number"
    ).columns.tolist()

    print("\nRaw data split")
    print("--------------")
    print(f"Training records:   {len(X_train):,}")
    print(f"Validation records: {len(X_val):,}")
    print(f"Numeric columns:    {len(numeric_columns)}")
    print(f"Categorical columns:{len(categorical_columns)}")

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
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )

    print("\nFitting preprocessing on the training fold only...")
    X_train_processed = preprocessor.fit_transform(X_train)

    print("Transforming validation data...")
    X_val_processed = preprocessor.transform(X_val)

    print("Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )

    start_time = time.time()
    model.fit(X_train_processed, y_train)
    training_seconds = time.time() - start_time

    predictions = model.predict(X_val_processed)
    probabilities = model.predict_proba(X_val_processed)[:, 1]

    tn, fp, fn, tp = confusion_matrix(
        y_val, predictions, labels=[0, 1]
    ).ravel()

    results = {
        "experiment": "clean_validation_rf_100_sqrt",
        "random_state": 42,
        "validation_fraction": 0.20,
        "preprocessing_fit_on_training_fold_only": True,
        "n_estimators": 100,
        "max_features": "sqrt",
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

    print("\nClean validation results")
    print("------------------------")
    for metric, value in results.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")

    print("\nConfusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(y_val, predictions, labels=[0, 1]))

    output_path = REPORT_DIR / "random_forest_clean_validation_metrics.csv"
    pd.DataFrame([results]).to_csv(output_path, index=False)

    print(f"\nSaved metrics to: {output_path.relative_to(PROJECT_ROOT)}")
    print("The official test set was not loaded or evaluated.")


if __name__ == "__main__":
    main()
