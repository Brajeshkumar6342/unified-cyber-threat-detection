
import time
from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
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


def main():
    print("Loading raw training dataset...")
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["label", "id", "attack_cat"])
    y = df["label"].astype(int)

    numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = X.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    # Split raw data before fitting preprocessing.
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

    # HistGradientBoosting requires dense input.
    print("Converting preprocessed matrices to dense arrays...")
    X_train_dense = X_train_processed.toarray()
    X_val_dense = X_val_processed.toarray()

    print(f"Processed training shape: {X_train_dense.shape}")
    print(f"Processed validation shape: {X_val_dense.shape}")

    models = {
        "RandomForest_200_leaf2": RandomForestClassifier(
            n_estimators=200,
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "HistGradientBoosting_default": HistGradientBoostingClassifier(
            max_iter=100,
            learning_rate=0.1,
            max_leaf_nodes=31,
            l2_regularization=0.0,
            random_state=RANDOM_STATE,
        ),
    }

    results = []

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        start_time = time.time()

        if model_name == "RandomForest_200_leaf2":
            model.fit(X_train_processed, y_train)
            predictions = model.predict(X_val_processed)
            probabilities = model.predict_proba(X_val_processed)[:, 1]
        else:
            model.fit(X_train_dense, y_train)
            predictions = model.predict(X_val_dense)
            probabilities = model.predict_proba(X_val_dense)[:, 1]

        training_seconds = time.time() - start_time

        tn, fp, fn, tp = confusion_matrix(
            y_val,
            predictions,
            labels=[0, 1],
        ).ravel()

        fpr = fp / (fp + tn) if (fp + tn) else 0.0

        result = {
            "model": model_name,
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
            "false_positive_rate": fpr,
            "training_seconds": training_seconds,
        }

        results.append(result)

        print(f"Accuracy: {result['accuracy']:.4f}")
        print(f"Attack precision: {result['precision_attack']:.4f}")
        print(f"Attack recall: {result['recall_attack']:.4f}")
        print(f"Attack F1-score: {result['f1_attack']:.4f}")
        print(f"ROC-AUC: {result['roc_auc']:.4f}")
        print(f"False-positive rate: {fpr:.4f}")
        print(f"False positives: {fp:,}")
        print(f"False negatives: {fn:,}")
        print(f"Training time: {training_seconds:.2f} seconds")

    results_df = pd.DataFrame(results)
    output_path = OUTPUT_DIR / "clean_gradient_boosting_comparison.csv"
    results_df.to_csv(output_path, index=False)

    print("\n=== MODEL COMPARISON ===")
    columns = [
        "model",
        "precision_attack",
        "recall_attack",
        "f1_attack",
        "roc_auc",
        "false_positive_rate",
        "false_positives",
        "false_negatives",
        "training_seconds",
    ]
    print(
        results_df[columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
