
"""Explore correctly detected and missed Fuzzers attacks.

This is descriptive error analysis of the official test set.
Do not use these results to tune the model.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from scipy.sparse import load_npz


ROOT = Path(__file__).resolve().parents[2]

RAW_TEST_PATH = ROOT / "data" / "raw" / "nids" / "UNSW_NB15_testing-set.csv"
X_TEST_PATH = ROOT / "data" / "processed" / "nids" / "X_test.npz"
Y_TEST_PATH = ROOT / "data" / "processed" / "nids" / "y_test.csv"
MODEL_PATH = ROOT / "models" / "nids" / "random_forest_baseline.joblib"
OUTPUT_PATH = ROOT / "reports" / "tables" / "fuzzer_error_summary.csv"


def main():
    print("Loading test data and trained model...")

    raw_test = pd.read_csv(RAW_TEST_PATH)
    X_test = load_npz(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH).iloc[:, 0].to_numpy()
    model = joblib.load(MODEL_PATH)

    predictions = model.predict(X_test)

    if not (
        len(raw_test) == X_test.shape[0] == len(y_test) == len(predictions)
    ):
        raise ValueError("Raw test data, features, labels, and predictions do not align.")

    # Fuzzers records are attack records in the original dataset.
    fuzzer_mask = raw_test["attack_cat"].astype(str).str.strip().eq("Fuzzers")
    actual_attack = y_test == 1
    missed_mask = fuzzer_mask & actual_attack & (predictions == 0)
    detected_mask = fuzzer_mask & actual_attack & (predictions == 1)

    missed_count = int(missed_mask.sum())
    detected_count = int(detected_mask.sum())
    total_fuzzers = missed_count + detected_count

    print("\nFuzzers detection summary")
    print("-------------------------")
    print(f"Total Fuzzers records:       {total_fuzzers:,}")
    print(f"Correctly detected:          {detected_count:,}")
    print(f"Missed (false negatives):    {missed_count:,}")

    if total_fuzzers:
        print(f"Detection rate:              {detected_count / total_fuzzers:.2%}")
        print(f"Miss rate:                   {missed_count / total_fuzzers:.2%}")

    # Compare original numeric network features for missed and detected records.
    excluded = {"id", "label"}
    numeric_columns = [
        col
        for col in raw_test.select_dtypes(include=np.number).columns
        if col not in excluded
    ]

    missed_features = raw_test.loc[missed_mask, numeric_columns]
    detected_features = raw_test.loc[detected_mask, numeric_columns]

    summary = pd.DataFrame(
        {
            "missed_fuzzers_median": missed_features.median(),
            "detected_fuzzers_median": detected_features.median(),
            "missed_fuzzers_mean": missed_features.mean(),
            "detected_fuzzers_mean": detected_features.mean(),
        }
    )
    summary["absolute_median_difference"] = (
        summary["missed_fuzzers_median"]
        - summary["detected_fuzzers_median"]
    ).abs()

    summary = summary.sort_values(
        "absolute_median_difference", ascending=False
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_PATH, index_label="feature")

    print("\nLargest median differences (descriptive only)")
    print("---------------------------------------------")
    print(summary.head(15).round(4).to_string())

    print(f"\nSummary saved to: {OUTPUT_PATH.relative_to(ROOT)}")
    print("\nAnalysis complete. Do not use official test results to tune the model.")


if __name__ == "__main__":
    main()
