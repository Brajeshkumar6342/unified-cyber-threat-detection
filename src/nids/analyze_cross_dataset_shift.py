
"""Compare candidate common-feature distributions across NIDS datasets."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

SOURCE_PATH = ROOT / "data/raw/nids/UNSW_NB15_training-set.csv"
TARGET_PATH = (
    ROOT
    / "data/raw/nids/cicids2017/"
    / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
)
OUTPUT_PATH = ROOT / "reports/tables/cross_dataset_feature_shift.csv"

FEATURE_MAP = {
    "duration_seconds": ("dur", "Flow Duration"),
    "forward_packets": ("spkts", "Total Fwd Packets"),
    "backward_packets": ("dpkts", "Total Backward Packets"),
    "forward_bytes": ("sbytes", "Total Length of Fwd Packets"),
    "backward_bytes": ("dbytes", "Total Length of Bwd Packets"),
}


def load_source():
    columns = sorted({source for source, _ in FEATURE_MAP.values()})
    data = pd.read_csv(SOURCE_PATH, usecols=columns)

    result = pd.DataFrame()
    for common_name, (source_column, _) in FEATURE_MAP.items():
        result[common_name] = pd.to_numeric(
            data[source_column], errors="coerce"
        )

    return result.replace([np.inf, -np.inf], np.nan)


def load_target():
    target_columns = [target for _, target in FEATURE_MAP.values()]
    data = pd.read_csv(
        TARGET_PATH,
        usecols=lambda column: column.strip() in target_columns,
    )
    data.columns = data.columns.str.strip()

    result = pd.DataFrame()
    for common_name, (_, target_column) in FEATURE_MAP.items():
        result[common_name] = pd.to_numeric(
            data[target_column], errors="coerce"
        )

    # CICIDS2017 duration is in microseconds; convert to seconds.
    result["duration_seconds"] = result["duration_seconds"] / 1_000_000

    # Invalid negative durations become missing in the derived analysis data.
    result.loc[result["duration_seconds"] < 0, "duration_seconds"] = np.nan

    return result.replace([np.inf, -np.inf], np.nan)


def summarize(name, series):
    valid = series.dropna()
    return {
        "dataset": name,
        "count": int(valid.count()),
        "missing": int(series.isna().sum()),
        "mean": valid.mean(),
        "std": valid.std(),
        "min": valid.min(),
        "median": valid.median(),
        "p95": valid.quantile(0.95),
        "p99": valid.quantile(0.99),
        "max": valid.max(),
    }


def main():
    source = load_source()
    target = load_target()

    records = []

    for feature in FEATURE_MAP:
        records.append(summarize("UNSW-NB15 training", source[feature]))
        records.append(summarize("CICIDS2017 Friday DDoS", target[feature]))

    summary = pd.DataFrame(records)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_PATH, index=False)

    pd.set_option("display.float_format", lambda value: f"{value:.6g}")

    print("\n=== Cross-Dataset Feature Distribution Comparison ===")
    print(summary.to_string(index=False))
    print(f"\nSaved report: {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
