
from pathlib import Path
import json

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = PROJECT_ROOT / "data" / "raw" / "nids" / "UNSW_NB15_training-set.csv"
TEST_PATH = PROJECT_ROOT / "data" / "raw" / "nids" / "UNSW_NB15_testing-set.csv"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "nids"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "label"
EXCLUDED_COLUMNS = ["id", "label", "attack_cat"]


def main():
    print("Loading UNSW-NB15 datasets...")

    train_df = pd.read_csv(TRAIN_PATH, low_memory=False)
    test_df = pd.read_csv(TEST_PATH, low_memory=False)

    print(f"Training shape: {train_df.shape}")
    print(f"Testing shape: {test_df.shape}")

    # Verify the required target and feature columns.
    if TARGET not in train_df.columns or TARGET not in test_df.columns:
        raise ValueError("The label column must exist in both datasets.")

    if set(train_df.columns) != set(test_df.columns):
        raise ValueError("Training and testing columns do not match.")

    # Separate labels from input features.
    y_train = train_df[TARGET].astype(int)
    y_test = test_df[TARGET].astype(int)

    X_train = train_df.drop(
        columns=[c for c in EXCLUDED_COLUMNS if c in train_df.columns]
    )
    X_test = test_df.drop(
        columns=[c for c in EXCLUDED_COLUMNS if c in test_df.columns]
    )

    # Detect numeric and categorical columns from the training set.
    numeric_columns = X_train.select_dtypes(include="number").columns.tolist()
    categorical_columns = X_train.select_dtypes(
        exclude="number"
    ).columns.tolist()

    print(f"Numeric features: {len(numeric_columns)}")
    print(f"Categorical features: {len(categorical_columns)}")
    print("Categorical columns:", categorical_columns)

    # Impute missing values and scale numeric features.
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Impute missing categories and one-hot encode them.
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
        ],
        remainder="drop",
    )

    print("\nFitting preprocessing on training data only...")
    X_train_processed = preprocessor.fit_transform(X_train)

    print("Transforming testing data...")
    X_test_processed = preprocessor.transform(X_test)

    # Save as sparse matrices to avoid unnecessary memory usage.
    from scipy.sparse import save_npz

    save_npz(
        OUTPUT_DIR / "X_train.npz",
        X_train_processed,
    )
    save_npz(
        OUTPUT_DIR / "X_test.npz",
        X_test_processed,
    )

    y_train.to_csv(OUTPUT_DIR / "y_train.csv", index=False)
    y_test.to_csv(OUTPUT_DIR / "y_test.csv", index=False)

    feature_names = preprocessor.get_feature_names_out().tolist()
    with open(OUTPUT_DIR / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)

    print("\nPreprocessing complete.")
    print("Processed training shape:", X_train_processed.shape)
    print("Processed testing shape:", X_test_processed.shape)
    print("Training labels:", y_train.value_counts().to_dict())
    print("Testing labels:", y_test.value_counts().to_dict())
    print("Saved processed files to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
