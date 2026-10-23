
"""Compare Random Forest and Extra Trees prediction thresholds.

Uses the raw UNSW-NB15 training CSV only.
Splits before preprocessing and never loads the official test set.
"""

from pathlib import Path
import time

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = (
    PROJECT_ROOT / "data" / "raw" / "nids" / "UNSW_NB15_training-set.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "reports" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70]
RANDOM_STATE = 42


def main():
    print("Loading raw training data only...")
    data = pd.read_csv(DATA_PATH, low_memory=False)

    y = data["label"].astype(int)
    X = data.drop(columns=["label", "id", "attack_cat"])

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    numeric_columns = X_train.select_dtypes(include="number").columns.tolist()
    categorical_columns = X_train.select_dtypes(
        exclude="number"
    ).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True,
        )),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns),
    ])

    print("Fitting preprocessing on training fold only...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=200,
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    results = []

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        start = time.time()
        model.fit(X_train_processed, y_train)
        print(f"Training time: {time.time() - start:.2f} seconds")

        probabilities = model.predict_proba(X_val_processed)[:, 1]

        for threshold in THRESHOLDS:
            predictions = (probabilities >= threshold).astype(int)

            tn, fp, fn, tp = confusion_matrix(
                y_val, predictions, labels=[0, 1]
            ).ravel()

            results.append({
                "model": model_name,
                "threshold": threshold,
                "precision_attack": precision_score(
                    y_val, predictions, zero_division=0
                ),
                "recall_attack": recall_score(
                    y_val, predictions, zero_division=0
                ),
                "f1_attack": f1_score(
                    y_val, predictions, zero_division=0
                ),
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
                "false_positive_rate": fp / (fp + tn),
            })

    results_df = pd.DataFrame(results)
    output_path = OUTPUT_DIR / "tree_ensemble_threshold_comparison.csv"
    results_df.to_csv(output_path, index=False)

    columns = [
        "model",
        "threshold",
        "precision_attack",
        "recall_attack",
        "f1_attack",
        "false_positive_rate",
        "false_positives",
        "false_negatives",
    ]

    print("\n=== THRESHOLD COMPARISON ===")
    print(results_df[columns].to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    ))

    qualifying = results_df[
        (results_df["recall_attack"] >= 0.98)
        & (results_df["false_positive_rate"] <= 0.05)
    ]

    print("\n=== RESEARCH TARGETS ===")
    print("Required: attack recall >= 98% and FPR <= 5%")

    if qualifying.empty:
        print("No tested model-threshold combination met both targets.")
    else:
        print(qualifying[columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        ))

    print(f"\nSaved results to: {output_path}")
    print("Official test data was not loaded.")


if __name__ == "__main__":
    main()

