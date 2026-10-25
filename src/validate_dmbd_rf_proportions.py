
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score

DATA = "data/processed/process_attribution/dmbd_all_events_experiment_features.csv"

df = pd.read_csv(DATA)
train = df[df["split"] == "train"].reset_index(drop=True)

event_cols = [
    "process_create", "process_term", "image_load",
    "file_events", "network_events", "registry_events"
]

X_counts = train[event_cols].astype(float)
denominator = X_counts.sum(axis=1).replace(0, 1)
X_combined = pd.concat(
    [
        X_counts,
        X_counts.div(denominator, axis=0).rename(
            columns={c: f"{c}_proportion" for c in event_cols}
        )
    ],
    axis=1
)
y = train["label"].astype(int)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
results = {"counts_only": [], "counts_plus_proportions": []}

for fold, (fit_idx, val_idx) in enumerate(cv.split(X_counts, y), start=1):
    for name, X in [
        ("counts_only", X_counts),
        ("counts_plus_proportions", X_combined)
    ]:
        model = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
        model.fit(X.iloc[fit_idx], y.iloc[fit_idx])
        prob = model.predict_proba(X.iloc[val_idx])[:, 1]
        pred = (prob >= 0.5).astype(int)

        results[name].append({
            "fold": fold,
            "f1": float(f1_score(y.iloc[val_idx], pred)),
            "roc_auc": float(roc_auc_score(y.iloc[val_idx], prob))
        })

        print(
            f"Fold {fold} | {name} | "
            f"F1={results[name][-1]['f1']:.5f} | "
            f"ROC-AUC={results[name][-1]['roc_auc']:.5f}"
        )

print("\nSummary: mean and standard deviation across validation folds")
for name, values in results.items():
    table = pd.DataFrame(values)
    print(f"\n{name}")
    print(table[["f1", "roc_auc"]].agg(["mean", "std"]).round(5).to_string())

Path("reports/phase3_rf_proportion_cv.json").write_text(
    __import__("json").dumps(results, indent=2),
    encoding="utf-8"
)
print("\nSaved: reports/phase3_rf_proportion_cv.json")
