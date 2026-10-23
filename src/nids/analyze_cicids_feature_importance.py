
"""Analyze feature importance for the within-CICIDS2017 Random Forest."""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    ROOT
    / "data"
    / "raw"
    / "nids"
    / "cicids2017"
    / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
)

FEATURES = [
    "duration_seconds",
    "forward_packets",
    "backward_packets",
    "forward_bytes",
    "backward_bytes",
]


def main():
    data = pd.read_csv(DATA_PATH)
    data.columns = data.columns.str.strip()

    duration = pd.to_numeric(data["Flow Duration"], errors="coerce")

    X = pd.DataFrame({
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
        raise ValueError(f"Unexpected labels: {unexpected}")

    y = labels.map({"BENIGN": 0, "DDOS": 1}).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
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

    model.fit(X_train, y_train)

    classifier = model.named_steps["classifier"]

    importance = pd.DataFrame({
        "feature": FEATURES,
        "importance": classifier.feature_importances_,
    }).sort_values("importance", ascending=False)

    output_path = ROOT / "reports" / "tables" / "cicids_feature_importance.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    importance.to_csv(output_path, index=False)

    print("=== CICIDS2017 Feature Importance ===")
    print(importance.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nSaved report to: {output_path}")


if __name__ == "__main__":
    main()
