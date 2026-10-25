
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

DATA = Path("data/processed/process_attribution/dmbd_all_events_experiment_features.csv")
REPORT = Path("reports/phase3_rf_count_proportion_comparison.json")

df = pd.read_csv(DATA)

event_cols = [
    "process_create", "process_term", "image_load",
    "file_events", "network_events", "registry_events"
]

train = df[df["split"] == "train"].copy()
test = df[df["split"] == "test"].copy()

# Split the official training partition into training and validation.
fit_df, val_df = train_test_split(
    train, test_size=0.20, random_state=42, stratify=train["label"]
)

def make_features(frame, use_proportions):
    counts = frame[event_cols].astype(float)
    if not use_proportions:
        return counts
    denominator = counts.sum(axis=1).replace(0, 1)
    proportions = counts.div(denominator, axis=0)
    proportions.columns = [f"{c}_proportion" for c in proportions.columns]
    return pd.concat([counts, proportions], axis=1)

def metrics(y, pred, prob):
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "malicious_precision": float(precision_score(y, pred, zero_division=0)),
        "malicious_recall": float(recall_score(y, pred, zero_division=0)),
        "malicious_f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, prob)),
        "confusion_matrix": confusion_matrix(y, pred).tolist()
    }

results = {}

for name, use_proportions in [
    ("counts_only", False),
    ("counts_plus_proportions", True),
]:
    X_fit = make_features(fit_df, use_proportions)
    X_val = make_features(val_df, use_proportions)
    y_fit = fit_df["label"].astype(int)
    y_val = val_df["label"].astype(int)

    best = None
    for leaf in [1, 2, 5, 10]:
        for depth in [None, 20, 40]:
            model = RandomForestClassifier(
                n_estimators=300,
                max_depth=depth,
                min_samples_leaf=leaf,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_fit, y_fit)
            prob = model.predict_proba(X_val)[:, 1]
            pred = (prob >= 0.5).astype(int)
            score = f1_score(y_val, pred, zero_division=0)
            candidate = (score, leaf, depth)
            if best is None or candidate[0] > best[0]:
                best = candidate

    best_val_f1, best_leaf, best_depth = best

    # Refit on the entire official training partition.
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=best_depth,
        min_samples_leaf=best_leaf,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    X_train = make_features(train, use_proportions)
    X_test = make_features(test, use_proportions)
    model.fit(X_train, train["label"].astype(int))

    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)

    results[name] = {
        "selected_min_samples_leaf": best_leaf,
        "selected_max_depth": best_depth,
        "validation_f1": float(best_val_f1),
        "test_metrics": metrics(test["label"].astype(int), pred, prob),
        "feature_importance": {
            feature: float(value)
            for feature, value in sorted(
                zip(X_train.columns, model.feature_importances_),
                key=lambda item: -item[1]
            )
        }
    }
    print(f"\n{name}")
    print(json.dumps(results[name], indent=2))

REPORT.parent.mkdir(parents=True, exist_ok=True)
REPORT.write_text(json.dumps(results, indent=2))
print(f"\nSaved comparison report: {REPORT}")
