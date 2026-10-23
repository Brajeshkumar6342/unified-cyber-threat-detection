
"""Compare feature subsets using a leakage-aware validation split.

Uses raw UNSW-NB15 training data only.
Feature ranking and preprocessing are fitted on the training fold.
The official test set is never loaded.
"""

from pathlib import Path
import time

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
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
    PROJECT_ROOT / "data" / "raw" / "nids" / "UNSW_NB15_training-set.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "reports" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


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


def evaluate_model(name, model, X_train, X_val, y_train, y_val):
    start = time.time()
    model.fit(X_train, y_train)
    training_seconds = time.time() - start

    probabilities = model.predict_proba(X_val)[:, 1]
    predictions = (probabilities >= 0.50).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_val, predictions, labels=[0, 1]
    ).ravel()

    return {
        "experiment": name,
        "feature_count": X_train.shape[1],
        "accuracy": accuracy_score(y_val, predictions),
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
    print("Loading raw training dataset only...")
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

    print(f"Training rows: {len(X_train):,}")
    print(f"Validation rows: {len(X_val):,}")

    # Fit preprocessing only on the training fold.
    preprocessor = make_preprocessor(X_train)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)

    baseline = RandomForestClassifier(
        n_estimators=200,
        max_features="sqrt",
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    results = []
    results.append(evaluate_model(
        "All_features",
        baseline,
        X_train_processed,
        X_val_processed,
        y_train,
        y_val,
    ))

    # Learn feature ranking from training-fold data only.
    print("\nLearning feature ranking from training fold...")
    selector_model = RandomForestClassifier(
        n_estimators=150,
        max_features="sqrt",
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    selector_model.fit(X_train_processed, y_train)

    importances = selector_model.feature_importances_

    feature_names = preprocessor.get_feature_names_out()
    importance_table = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    ranking_path = OUTPUT_DIR / "feature_selection_training_importance.csv"
    importance_table.to_csv(ranking_path, index=False)

    for keep_count in [50, 100]:
        keep_count = min(keep_count, X_train_processed.shape[1])

        selector = SelectFromModel(
            selector_model,
            threshold=-float("inf"),
            max_features=keep_count,
            prefit=True,
        )

        X_train_selected = selector.transform(X_train_processed)
        X_val_selected = selector.transform(X_val_processed)

        selected_model = RandomForestClassifier(
            n_estimators=200,
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        results.append(evaluate_model(
            f"Top_{keep_count}_features",
            selected_model,
            X_train_selected,
            X_val_selected,
            y_train,
            y_val,
        ))

    results_df = pd.DataFrame(results)
    output_path = OUTPUT_DIR / "feature_selection_comparison.csv"
    results_df.to_csv(output_path, index=False)

    print("\n=== FEATURE SELECTION COMPARISON ===")
    print(results_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    ))

    print(f"\nComparison saved to: {output_path}")
    print(f"Training-fold feature ranking saved to: {ranking_path}")
    print("Official test set was not loaded.")


if __name__ == "__main__":
    main()
