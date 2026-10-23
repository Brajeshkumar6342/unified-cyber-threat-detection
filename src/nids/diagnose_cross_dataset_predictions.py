
"""Diagnose prediction behaviour in cross-dataset NIDS evaluation.

Uses the UNSW-NB15 training CSV for source training/validation.
Evaluates transfer to the CICIDS2017 Friday DDoS subset.
Does not use the official UNSW-NB15 test set or tune the model.
"""

from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from experiment_cross_dataset import (
    FEATURES,
    RandomForestClassifier,
    SimpleImputer,
    Pipeline,
    load_source,
    load_target,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = ROOT / "reports" / "tables" / "cross_dataset_prediction_diagnostics.csv"


def main():
    X_source, y_source = load_source()
    X_target, y_target = load_target()

    X_train, X_val, y_train, y_val = train_test_split(
        X_source,
        y_source,
        test_size=0.2,
        random_state=42,
        stratify=y_source,
    )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )),
    ])

    model.fit(X_train[FEATURES], y_train)

    datasets = [
        ("UNSW_NB15_validation", X_val[FEATURES], y_val),
        ("CICIDS2017_transfer", X_target[FEATURES], y_target),
    ]

    rows = []

    for dataset_name, X, y in datasets:
        probabilities = model.predict_proba(X)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y, predictions, labels=[0, 1]
        ).ravel()

        print(f"\n=== {dataset_name} ===")
        print(f"Rows: {len(y):,}")
        print(f"Actual attack rate: {y.mean():.4f}")
        print(f"Predicted attack rate: {predictions.mean():.4f}")
        print(f"Mean attack probability: {probabilities.mean():.4f}")
        print(f"Median attack probability: {pd.Series(probabilities).median():.4f}")
        print(f"True negatives: {tn:,}")
        print(f"False positives: {fp:,}")
        print(f"False negatives: {fn:,}")
        print(f"True positives: {tp:,}")
        print(f"False-positive rate: {fp / (tn + fp):.4f}")
        print(f"False-negative rate: {fn / (fn + tp):.4f}")

        for actual_label in (0, 1):
            class_probabilities = probabilities[y.to_numpy() == actual_label]

            print(
                f"Actual class {actual_label}: "
                f"n={len(class_probabilities):,}, "
                f"mean_probability={class_probabilities.mean():.4f}, "
                f"median_probability={pd.Series(class_probabilities).median():.4f}"
            )

        rows.append({
            "dataset": dataset_name,
            "rows": len(y),
            "actual_attack_rate": y.mean(),
            "predicted_attack_rate": predictions.mean(),
            "mean_attack_probability": probabilities.mean(),
            "median_attack_probability": pd.Series(probabilities).median(),
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
            "false_positive_rate": fp / (tn + fp),
            "false_negative_rate": fn / (fn + tp),
            "probability_below_0_1": (probabilities < 0.1).mean(),
            "probability_0_4_to_0_6": (
                (probabilities >= 0.4) & (probabilities <= 0.6)
            ).mean(),
            "probability_above_0_9": (probabilities > 0.9).mean(),
        })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT_PATH, index=False)

    print("\nSaved diagnostic summary to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
