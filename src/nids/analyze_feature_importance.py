
from pathlib import Path
import json

import pandas as pd
import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "nids" / "random_forest_baseline.joblib"
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "nids"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "tables"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("Loading trained Random Forest model...")

    model = joblib.load(MODEL_PATH)

    with open(DATA_DIR / "feature_names.json", encoding="utf-8") as file:
        feature_names = json.load(file)

    importances = model.feature_importances_

    if len(feature_names) != len(importances):
        raise ValueError("Feature names and model importances do not match.")

    results = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    })

    results = results.sort_values(
        "importance", ascending=False
    ).reset_index(drop=True)

    output_path = OUTPUT_DIR / "random_forest_feature_importance.csv"
    results.to_csv(output_path, index=False)

    print("\nTop 20 features:")
    print(results.head(20).to_string(index=False))

    print("\nSaved complete feature ranking to:")
    print(output_path)


if __name__ == "__main__":
    main()
