from pathlib import Path

import numpy as np
import pandas as pd
import joblib

DATA_FILE = Path("data/processed/malware/ember_sample.npz")
MODEL_FILE = Path("models/malware/ember_random_forest_baseline.joblib")
OUTPUT_FILE = Path("reports/ember_feature_importance.csv")

with np.load(DATA_FILE, allow_pickle=False) as data:
    feature_names = data["feature_names"].astype(str)

model = joblib.load(MODEL_FILE)

if len(feature_names) != len(model.feature_importances_):
    raise ValueError("Feature names and model feature count do not match.")

importance = model.feature_importances_
top_indices = np.argsort(importance)[::-1][:20]

results = pd.DataFrame({
    "Rank": np.arange(1, len(top_indices) + 1),
    "Feature_Index": top_indices,
    "Feature_Name": feature_names[top_indices],
    "Importance": importance[top_indices],
})

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
results.to_csv(OUTPUT_FILE, index=False)

print("\nTop 20 EMBER Feature Importances")
print(results.to_string(index=False))
print(f"\nSaved to: {OUTPUT_FILE}")
