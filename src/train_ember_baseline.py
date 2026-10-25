from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

DATA_FILE = Path("data/processed/malware/ember_sample.npz")
MODEL_DIR = Path("models/malware")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

with np.load(DATA_FILE) as data:
    X = data["X"]
    y = data["y"]

print(f"Dataset shape: {X.shape}")
print(f"Benign: {int(np.sum(y == 0)):,}")
print(f"Malicious: {int(np.sum(y == 1)):,}")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print(f"\nTraining samples: {len(y_train):,}")
print(f"Testing samples: {len(y_test):,}")
print("\nTraining Random Forest...", flush=True)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced",
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n=== EMBER Malware Classification: Baseline Results ===")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred, zero_division=0):.4f}")
print(f"F1-score:  {f1_score(y_test, y_pred, zero_division=0):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob):.4f}")

print("\nClassification report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Benign", "Malicious"],
    zero_division=0,
))

print("Confusion matrix (rows=actual, columns=predicted):")
print(confusion_matrix(y_test, y_pred))

model_path = MODEL_DIR / "ember_random_forest_baseline.joblib"
joblib.dump(model, model_path)
print(f"\nModel saved to: {model_path}")
