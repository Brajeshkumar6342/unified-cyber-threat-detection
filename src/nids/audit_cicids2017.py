
"""Memory-conscious audit of the CICIDS2017 CSV files."""

from pathlib import Path
import csv

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "nids" / "cicids2017"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    files = sorted(DATA_DIR.glob("*.csv"))

    if not files:
        raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")

    summaries = []

    for path in files:
        print(f"\n{'=' * 70}")
        print(f"File: {path.name}")
        print(f"Size: {path.stat().st_size / (1024 * 1024):.1f} MB")

        # Read the header without loading the full dataset.
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
            header = next(csv.reader(f))

        print(f"Columns: {len(header)}")
        print("First 12 columns:", header[:12])
        print("Last 5 columns:", header[-5:])

        # Process rows in chunks to limit memory use.
        row_count = 0
        label_counts = {}
        missing_cells = 0
        infinity_cells = 0
        duplicate_rows = 0
        first_chunk_columns = None

        try:
            for chunk in pd.read_csv(
                path,
                encoding="utf-8-sig",
                low_memory=False,
                chunksize=100_000,
            ):
                first_chunk_columns = chunk.columns.tolist()
                row_count += len(chunk)

                # CICIDS2017 normally names its target column " Label".
                label_col = next(
                    (c for c in chunk.columns if c.strip().lower() == "label"),
                    None,
                )

                if label_col is not None:
                    counts = chunk[label_col].astype("string").str.strip().value_counts()
                    for label, count in counts.items():
                        label_counts[str(label)] = (
                            label_counts.get(str(label), 0) + int(count)
                        )

                missing_cells += int(chunk.isna().sum().sum())

                numeric = chunk.select_dtypes(include="number")
                if not numeric.empty:
                    infinity_cells += int(
                        (~numeric.map(lambda x: abs(x) != float("inf"))).sum().sum()
                    )

                duplicate_rows += int(chunk.duplicated().sum())

        except Exception as exc:
            print(f"ERROR while reading {path.name}: {exc}")
            raise

        print(f"Rows: {row_count:,}")
        print(f"Missing cells: {missing_cells:,}")
        print(f"Infinite numeric cells: {infinity_cells:,}")
        print(f"Duplicate rows within chunks: {duplicate_rows:,}")
        print(f"Label counts: {label_counts}")

        summaries.append({
            "file": path.name,
            "rows": row_count,
            "columns": len(first_chunk_columns or header),
            "missing_cells": missing_cells,
            "infinite_numeric_cells": infinity_cells,
            "duplicate_rows_within_chunks": duplicate_rows,
            "label_counts": str(label_counts),
        })

    summary_df = pd.DataFrame(summaries)
    output_path = OUTPUT_DIR / "cicids2017_dataset_audit.csv"
    summary_df.to_csv(output_path, index=False)

    print("\n=== DATASET AUDIT SUMMARY ===")
    print(summary_df.to_string(index=False))
    print(f"\nAudit saved to: {output_path}")


if __name__ == "__main__":
    main()
