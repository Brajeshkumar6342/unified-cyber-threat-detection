
"""Compare feature names between UNSW-NB15 and CICIDS2017."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

UNSW_PATH = ROOT / "data" / "raw" / "nids" / "UNSW_NB15_training-set.csv"
CICIDS_DIR = ROOT / "data" / "raw" / "nids" / "cicids2017"
OUTPUT_DIR = ROOT / "reports" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def normalized_feature_names(path):
    # Read only a small sample; do not load the full datasets.
    sample = pd.read_csv(path, nrows=1000, low_memory=False)
    sample.columns = [str(c).strip() for c in sample.columns]

    label_names = {"label", "attack_cat"}
    excluded = {"id", *label_names}

    features = [
        c for c in sample.columns
        if c.strip().lower() not in excluded
    ]

    numeric_features = set(
        sample[features].select_dtypes(include="number").columns
    )

    return set(features), numeric_features


def main():
    unsw_features, unsw_numeric = normalized_feature_names(UNSW_PATH)

    print(f"UNSW-NB15 features: {len(unsw_features)}")
    print(f"UNSW-NB15 numeric features: {len(unsw_numeric)}")

    rows = []

    for path in sorted(CICIDS_DIR.glob("*.csv")):
        cic_features, cic_numeric = normalized_feature_names(path)

        common = sorted(unsw_features & cic_features)
        common_numeric = sorted(unsw_numeric & cic_numeric)

        print(f"\n{'=' * 60}")
        print(f"CICIDS2017 file: {path.name}")
        print(f"CICIDS2017 features: {len(cic_features)}")
        print(f"Exact shared feature names: {len(common)}")
        print(f"Shared numeric feature names: {len(common_numeric)}")

        if common:
            print("Shared features:", common)

        rows.append({
            "cicids_file": path.name,
            "unsw_feature_count": len(unsw_features),
            "cicids_feature_count": len(cic_features),
            "shared_feature_count": len(common),
            "shared_numeric_feature_count": len(common_numeric),
            "shared_features": "; ".join(common),
        })

    output_path = OUTPUT_DIR / "dataset_feature_compatibility.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)

    print(f"\nCompatibility report saved to: {output_path}")


if __name__ == "__main__":
    main()
