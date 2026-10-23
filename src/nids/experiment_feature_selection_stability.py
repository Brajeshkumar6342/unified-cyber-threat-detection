
"""Test feature-selection stability across three validation splits.

Feature ranking and preprocessing are fitted independently within
each training fold. The official test set is never loaded.
"""

from pathlib import Path
import time

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
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
    PROJECT_ROOT / "data" / "raw" / "nids" / "UNSW_NB15_training-set.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "reports" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 21, 7]
TOP_FEATURES = 50


def make_preprocessor(X):
    numeric_columns = X.select_dtypes(include="number").columns.tolist()
    categorical_columns = X.select_dtypes(
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

    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns),
    ])


def evaluate(name, seed, X_train, X_val, y_train, y_val):
    model = RandomForestClassifier(
        n_estimators=200,
        max_features="sqrt",
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=-1,
    )

    start = time.time()
    model.fit(X_train, y_train)
    training_seconds = time.time() - start

    probabilities = model.predict_proba(X_val)[:, 1]
    predictions = (probabilities >= 0.50).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_val, predictions, labels=[0, 1]
    ).ravel()

    return {
        "seed": seed,
        "experiment": name,
        "feature_count": X_train.shape[1],
        "precision_attack": precision_score(
            y_val, predictions, zero_division=0
        ),
        "recall_attack": recall_score(
            y_val, predictions, zero_division=0
        ),
        "f1_attack": f1_score(y_val, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_val, probabilities),
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "training_seconds": training_seconds,
    }


def main():
    print("Loading raw training CSV only...")
    data = pd.read_csv(DATA_PATH, low_memory=False)

    y = data["label"].astype(int)
    X = data.drop(columns=["label", "id", "attack_cat"])

    results = []

    for seed in SEEDS:
        print(f"\n{'=' * 50}")
        print(f"Validation split seed: {seed}")
        print("=" * 50)

        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=seed,
            stratify=y,
        )

        preprocessor = make_preprocessor(X_train)
        X_train_processed = preprocessor.fit_transform(X_train)
        X_val_processed = preprocessor.transform(X_val)

        print(f"Training rows: {len(X_train):,}")
        print(f"Validation rows: {len(X_val):,}")
        print(f"Encoded features: {X_train_processed.shape[1]}")

        # Baseline: all encoded features.
        results.append(evaluate(
            "All_features",
            seed,
            X_train_processed,
            X_val_processed,
            y_train,
            y_val,
        ))

        # Learn feature importance from this split's training data only.
        ranking_model = RandomForestClassifier(
            n_estimators=150,
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        )
        ranking_model.fit(X_train_processed, y_train)

        keep_count = min(TOP_FEATURES, X_train_processed.shape[1])
        selector = SelectFromModel(
            ranking_model,
            threshold=-float("inf"),
            max_features=keep_count,
            prefit=True,
        )

        X_train_selected = selector.transform(X_train_processed)
        X_val_selected = selector.transform(X_val_processed)

        results.append(evaluate(
            "Top_50_features",
            seed,
            X_train_selected,
            X_val_selected,
            y_train,
            y_val,
        ))

    results_df = pd.DataFrame(results)
    results_path = OUTPUT_DIR / "feature_selection_stability.csv"
    results_df.to_csv(results_path, index=False)

    print("\n=== RESULTS BY SPLIT ===")
    print(results_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    ))

    metrics = [
        "precision_attack",
        "recall_attack",
        "f1_attack",
        "roc_auc",
        "false_positive_rate",
        "false_positives",
        "false_negatives",
        "training_seconds",
    ]

    summary = results_df.groupby("experiment")[metrics].agg(
        ["mean", "std"]
    )

    summary_path = OUTPUT_DIR / "feature_selection_stability_summary.csv"
    summary.to_csv(summary_path)

    print("\n=== MEAN AND STANDARD DEVIATION ACROSS SPLITS ===")
    print(summary.round(4).to_string())

    print(f"\nDetailed results: {results_path}")
    print(f"Summary results:  {summary_path}")
    print("Official test set was not loaded.")


if __name__ == "__main__":
    main()
