
from pathlib import Path

import pandas as pd
import joblib
from scipy.sparse import load_npz
from sklearn.metrics import confusion_matrix


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "nids"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "nids"
MODEL_PATH = PROJECT_ROOT / "models" / "nids" / "random_forest_baseline.joblib"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "tables"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("Loading model and test data...")

    model = joblib.load(MODEL_PATH)
    X_test = load_npz(DATA_DIR / "X_test.npz")
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").iloc[:, 0]

    raw_test = pd.read_csv(
        RAW_DIR / "UNSW_NB15_testing-set.csv",
        low_memory=False,
    )

    if len(raw_test) != len(y_test):
        raise ValueError("Raw test rows do not match processed test labels.")

    y_pred = model.predict(X_test)

    tn, fp, fn, tp = confusion_matrix(
        y_test, y_pred, labels=[0, 1]
    ).ravel()

    print("\nConfusion matrix counts:")
    print(f"True negatives:  {tn:,}")
    print(f"False positives: {fp:,}")
    print(f"False negatives: {fn:,}")
    print(f"True positives:  {tp:,}")

    # Keep raw features and actual/predicted labels for interpretation.
    errors = raw_test.copy()
    errors["actual_label"] = y_test.to_numpy()
    errors["predicted_label"] = y_pred
    errors["error_type"] = ""

    errors.loc[
        (errors["actual_label"] == 0)
        & (errors["predicted_label"] == 1),
        "error_type",
    ] = "false_positive"

    errors.loc[
        (errors["actual_label"] == 1)
        & (errors["predicted_label"] == 0),
        "error_type",
    ] = "false_negative"

    errors = errors[errors["error_type"] != ""].copy()

    output_path = OUTPUT_DIR / "random_forest_misclassified_records.csv"
    errors.to_csv(output_path, index=False)

    print("\nError counts by type:")
    print(errors["error_type"].value_counts().to_string())

    print("\nError counts by actual attack category:")
    print(
        pd.crosstab(
            errors["attack_cat"],
            errors["error_type"],
            dropna=False,
        ).to_string()
    )

    print("\nSaved misclassified records to:")
    print(output_path)


if __name__ == "__main__":
    main()
