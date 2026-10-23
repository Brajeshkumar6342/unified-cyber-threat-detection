
"""Compare candidate NIDS features by dataset and true class."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

SOURCE_PATH = ROOT / "data" / "raw" / "nids" / "UNSW_NB15_training-set.csv"
TARGET_PATH = (
    ROOT
    / "data"
    / "raw"
    / "nids"
    / "cicids2017"
    / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
)

OUTPUT_PATH = (
    ROOT
    / "reports"
    / "tables"
    / "cross_dataset_class_conditional_shift.csv"
)


def load_source():
    columns = ["dur", "spkts", "dpkts", "sbytes", "dbytes", "label"]
    data = pd.read_csv(SOURCE_PATH, usecols=columns)

    result = pd.DataFrame({
        "duration_seconds": pd.to_numeric(data["dur"], errors="coerce"),
        "forward_packets": pd.to_numeric(data["spkts"], errors="coerce"),
        "backward_packets": pd.to_numeric(data["dpkts"], errors="coerce"),
        "forward_bytes": pd.to_numeric(data["sbytes"], errors="coerce"),
        "backward_bytes": pd.to_numeric(data["dbytes"], errors="coerce"),
        "label": pd.to_numeric(data["label"], errors="coerce"),
    })

    if result["label"].isna().any():
        raise ValueError("Missing source labels.")

    if not set(result["label"].unique()).issubset({0, 1}):
        raise ValueError("Unexpected source labels.")

    result["label"] = result["label"].astype(int)
    result["dataset"] = "UNSW-NB15"

    return result


def load_target():
    data = pd.read_csv(TARGET_PATH)
    data.columns = data.columns.str.strip()

    duration = pd.to_numeric(data["Flow Duration"], errors="coerce")

    result = pd.DataFrame({
        "duration_seconds": duration.where(duration >= 0) / 1_000_000,
        "forward_packets": pd.to_numeric(
            data["Total Fwd Packets"], errors="coerce"
        ),
        "backward_packets": pd.to_numeric(
            data["Total Backward Packets"], errors="coerce"
        ),
        "forward_bytes": pd.to_numeric(
            data["Total Length of Fwd Packets"], errors="coerce"
        ),
        "backward_bytes": pd.to_numeric(
            data["Total Length of Bwd Packets"], errors="coerce"
        ),
    })

    labels = data["Label"].astype(str).str.strip().str.upper()

    unexpected = sorted(set(labels.unique()) - {"BENIGN", "DDOS"})
    if unexpected:
        raise ValueError(f"Unexpected target labels: {unexpected}")

    result["label"] = labels.map({"BENIGN": 0, "DDOS": 1}).astype(int)
    result["dataset"] = "CICIDS2017"

    return result


def main():
    source = load_source()
    target = load_target()

    combined = pd.concat([source, target], ignore_index=True)

    feature_columns = [
        "duration_seconds",
        "forward_packets",
        "backward_packets",
        "forward_bytes",
        "backward_bytes",
    ]

    records = []

    for (dataset, label), group in combined.groupby(["dataset", "label"]):
        for feature in feature_columns:
            values = pd.to_numeric(group[feature], errors="coerce")
            valid = values.replace([np.inf, -np.inf], np.nan).dropna()

            records.append({
                "dataset": dataset,
                "class_label": int(label),
                "class_name": "Benign" if label == 0 else "Attack",
                "feature": feature,
                "rows_in_class": len(group),
                "missing_or_nonfinite": int(
                    (values.isna() | ~np.isfinite(values)).sum()
                ),
                "mean": valid.mean(),
                "std": valid.std(),
                "median": valid.median(),
                "p95": valid.quantile(0.95),
                "p99": valid.quantile(0.99),
                "max": valid.max(),
            })

    summary = pd.DataFrame(records)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_PATH, index=False)

    print("=== Class-conditional feature summary ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nSaved report to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
