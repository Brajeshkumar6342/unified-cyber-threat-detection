
from pathlib import Path

import numpy as np
import pandas as pd


def audit_dataset(
    file_path: str,
    label_column: str | None = None,
) -> None:
    """Print a basic data-quality report for a CSV dataset."""

    path = Path(file_path)

    if not path.exists():
        print(f"ERROR: File not found: {path}")
        print("Check the path and try again.")
        return

    print("=" * 70)
    print("DATASET AUDIT REPORT")
    print("=" * 70)
    print(f"File: {path}")

    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        print(f"ERROR: Could not read the CSV: {exc}")
        return

    rows, columns = df.shape

    print(f"\nRows: {rows:,}")
    print(f"Columns: {columns:,}")
    print(f"Duplicate rows: {df.duplicated().sum():,}")

    print("\n--- COLUMN DATA TYPES ---")
    print(df.dtypes.to_string())

    print("\n--- MISSING VALUES ---")
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if missing.empty:
        print("No missing values found.")
    else:
        print(missing.to_string())

    print("\n--- INFINITE VALUES (NUMERIC COLUMNS) ---")
    numeric_df = df.select_dtypes(include=np.number)
    infinite = pd.Series(
        np.isinf(numeric_df.to_numpy()).sum(axis=0),
        index=numeric_df.columns,
    )
    infinite = infinite[infinite > 0].sort_values(ascending=False)

    if infinite.empty:
        print("No infinite values found in numeric columns.")
    else:
        print(infinite.to_string())

    print("\n--- CLASS DISTRIBUTION ---")

    if label_column is None:
        possible_labels = [
            col for col in df.columns
            if col.strip().lower() in {"label", "class", "attack_cat", "target"}
        ]

        if len(possible_labels) == 1:
            label_column = possible_labels[0]
            print(f"Detected label column: {label_column}")
        elif len(possible_labels) > 1:
            print(
                "Multiple possible label columns found: "
                f"{possible_labels}"
            )
            print("Pass the intended label column explicitly.")
        else:
            print("No common label column detected.")
            print("Pass the label column explicitly if known.")

    if label_column is not None:
        if label_column not in df.columns:
            print(f"ERROR: Label column '{label_column}' does not exist.")
        else:
            counts = df[label_column].value_counts(dropna=False)
            percentages = df[label_column].value_counts(
                normalize=True, dropna=False
            ).mul(100)

            distribution = pd.DataFrame({
                "count": counts,
                "percentage": percentages.round(2),
            })
            print(distribution.to_string())

    print("\n--- NUMERIC SUMMARY ---")
    if numeric_df.empty:
        print("No numeric columns found.")
    else:
        print(numeric_df.describe().T.to_string())

    print("\nAudit complete.")
    print("=" * 70)


if __name__ == "__main__":
    audit_dataset(
        "data/processed/nids/sample_nids.csv",
        label_column="label",
    )