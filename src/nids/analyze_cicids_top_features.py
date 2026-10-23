"""Compare top CICIDS2017 features across BENIGN and DDoS traffic."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    ROOT
    / "data"
    / "raw"
    / "nids"
    / "cicids2017"
    / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
)

FEATURES = {
    "forward_bytes": "Total Length of Fwd Packets",
    "forward_packets": "Total Fwd Packets",
}


def main():
    data = pd.read_csv(DATA_PATH)
    data.columns = data.columns.str.strip()

    labels = data["Label"].astype(str).str.strip().str.upper()

    unexpected = sorted(set(labels.unique()) - {"BENIGN", "DDOS"})
    if unexpected:
        raise ValueError(f"Unexpected labels found: {unexpected}")

    data["traffic_class"] = labels.map({
        "BENIGN": "BENIGN",
        "DDOS": "DDoS",
    })

    for feature_name, source_column in FEATURES.items():
        data[feature_name] = pd.to_numeric(
            data[source_column], errors="coerce"
        )

    rows = []

    for feature_name in FEATURES:
        for traffic_class in ["BENIGN", "DDoS"]:
            values = data.loc[
                data["traffic_class"] == traffic_class,
                feature_name,
            ].dropna()

            rows.append({
                "feature": feature_name,
                "traffic_class": traffic_class,
                "count": len(values),
                "mean": values.mean(),
                "median": values.median(),
                "p95": values.quantile(0.95),
                "p99": values.quantile(0.99),
                "maximum": values.max(),
                "zero_count": int((values == 0).sum()),
            })

    summary = pd.DataFrame(rows)

    output_path = (
        ROOT
        / "reports"
        / "tables"
        / "cicids_top_feature_class_summary.csv"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)

    print("=== CICIDS2017: Top Feature Comparison ===")
    print(summary.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    ))
    print(f"\nSaved report to: {output_path}")


if __name__ == "__main__":
    main()
    