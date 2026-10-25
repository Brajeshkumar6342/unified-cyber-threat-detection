from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed" / "process_attribution"
REPORT_DIR = ROOT / "reports"
RF_PATH = REPORT_DIR / "dmbd_experiment_random_forest.joblib"
LSTM_PATH = ROOT / "models" / "dmbd_lstm_experiment_classifier.pt"
FEATURES_PATH = DATA_DIR / "dmbd_all_events_experiment_features.csv"
NPZ_PATH = DATA_DIR / "dmbd_lstm_sequences_512.npz"
OUTPUT_JSON = REPORT_DIR / "phase3_validation_ensemble_comparison.json"
OUTPUT_CSV = REPORT_DIR / "phase3_validation_ensemble_predictions.csv"

class EventLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)
        return self.classifier(self.dropout(hidden[-1])).squeeze(1)

def calculate_metrics(y, probabilities, predictions):
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "confusion_matrix": confusion_matrix(y, predictions).tolist(),
    }

def main():
    torch.set_num_threads(min(4, torch.get_num_threads()))

    with np.load(NPZ_PATH, allow_pickle=False) as data:
        X = data["X"].copy()
        y = data["y"].astype(np.int64)
        splits = data["split"].copy()
        ids = data["experiment_ids"].astype(str)
        original_lengths = data["original_lengths"].astype(np.int64)

    train_indices = np.flatnonzero(splits == "train")
    _, val_indices = train_test_split(
        train_indices,
        test_size=0.15,
        random_state=42,
        stratify=y[train_indices],
    )

    val_ids = ids[val_indices]
    y_val = y[val_indices]

    # Load the saved LSTM architecture and weights.
    checkpoint = torch.load(LSTM_PATH, map_location="cpu", weights_only=False)
    vocabulary = checkpoint["vocabulary"]
    lstm = EventLSTM(
        vocab_size=max(vocabulary.values()) + 1,
        embed_dim=checkpoint["embedding_dim"],
        hidden_dim=checkpoint["hidden_dim"],
    )
    lstm.load_state_dict(checkpoint["model_state_dict"])
    lstm.eval()

    val_X = torch.from_numpy(X[val_indices].astype(np.int64))
    val_lengths = torch.from_numpy(
        np.minimum(original_lengths[val_indices], X.shape[1]).astype(np.int64)
    )

    # Get LSTM probabilities for the validation experiments.
    lstm_probs = []
    with torch.no_grad():
        for start in range(0, len(val_indices), 128):
            xb = val_X[start:start + 128]
            lb = val_lengths[start:start + 128]
            logits = lstm(xb, lb)
            lstm_probs.extend(torch.sigmoid(logits).numpy().tolist())
    lstm_probs = np.asarray(lstm_probs)

    # Get Random Forest probabilities using its exact saved feature schema.
    features_df = pd.read_csv(FEATURES_PATH)
    val_features = features_df.set_index(features_df["experiment"].astype(str))
    missing_ids = [exp_id for exp_id in val_ids if exp_id not in val_features.index]
    if missing_ids:
        raise ValueError(f"{len(missing_ids)} validation IDs missing from feature table")

    rf = joblib.load(RF_PATH)
    feature_names = list(rf.feature_names_in_)
    X_rf = val_features.loc[val_ids, feature_names].astype(float)
    rf_probs = rf.predict_proba(X_rf)[:, 1]

    if len(rf_probs) != len(y_val) or len(lstm_probs) != len(y_val):
        raise ValueError("Prediction lengths do not match validation labels")

    rf_pred = (rf_probs >= 0.5).astype(int)
    lstm_pred = (lstm_probs >= 0.5).astype(int)
    avg_probs = (rf_probs + lstm_probs) / 2.0
    avg_pred = (avg_probs >= 0.5).astype(int)
    or_pred = ((rf_probs >= 0.5) | (lstm_probs >= 0.5)).astype(int)
    # For the OR rule, use the maximum probability as a ranking score.
    or_probs = np.maximum(rf_probs, lstm_probs)

    results = {
        "validation_size": int(len(val_indices)),
        "validation_benign": int((y_val == 0).sum()),
        "validation_malicious": int((y_val == 1).sum()),
        "note": (
            "Exploratory validation results only. The saved LSTM checkpoint "
            "was selected using this same validation split, so its results "
            "are not an independent estimate. Official test data was not used."
        ),
        "models": {
            "random_forest": calculate_metrics(y_val, rf_probs, rf_pred),
            "lstm": calculate_metrics(y_val, lstm_probs, lstm_pred),
            "probability_average": calculate_metrics(y_val, avg_probs, avg_pred),
            "or_rule": calculate_metrics(y_val, or_probs, or_pred),
        },
    }

    prediction_df = pd.DataFrame({
        "experiment": val_ids,
        "true_label": y_val,
        "rf_probability": rf_probs,
        "lstm_probability": lstm_probs,
        "rf_prediction": rf_pred,
        "lstm_prediction": lstm_pred,
        "average_probability": avg_probs,
        "average_prediction": avg_pred,
        "or_prediction": or_pred,
    })

    OUTPUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    prediction_df.to_csv(OUTPUT_CSV, index=False)

    print("Validation experiments:", len(val_indices))
    print("Validation IDs unique:", len(set(val_ids)) == len(val_ids))
    print("Validation/test overlap:", bool(
        set(val_ids) & set(ids[splits == "test"])
    ))
    print("\nValidation metrics:")
    for name, metrics in results["models"].items():
        print(
            f"{name}: F1={metrics['f1']:.4f}, "
            f"precision={metrics['precision']:.4f}, "
            f"recall={metrics['recall']:.4f}, "
            f"ROC-AUC={metrics['roc_auc']:.4f}"
        )
    print("\nSaved:", OUTPUT_JSON)
    print("Saved:", OUTPUT_CSV)

if __name__ == "__main__":
    main()
