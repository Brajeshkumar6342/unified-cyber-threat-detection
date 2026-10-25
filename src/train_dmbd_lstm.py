from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed" / "process_attribution"
REPORT_DIR = ROOT / "reports"
MODEL_DIR = ROOT / "models"

NPZ_PATH = DATA_DIR / "dmbd_lstm_sequences_512.npz"
META_PATH = DATA_DIR / "dmbd_lstm_sequences_512_metadata.csv"
VOCAB_PATH = DATA_DIR / "dmbd_lstm_event_vocabulary.json"

SEED = 42
BATCH_SIZE = 128
EPOCHS = 12
PATIENCE = 3
EMBED_DIM = 16
HIDDEN_DIM = 32
LEARNING_RATE = 0.001
VALIDATION_SIZE = 0.15


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class EventLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=0
        )
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden, _) = self.lstm(packed)
        return self.classifier(
            self.dropout(hidden[-1])
        ).squeeze(1)


def main():
    set_seed(SEED)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device("cpu")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for path in (NPZ_PATH, META_PATH, VOCAB_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Missing required input: {path}")

    metadata = pd.read_csv(META_PATH)
    with VOCAB_PATH.open("r", encoding="utf-8") as f:
        vocabulary = json.load(f)

    with np.load(NPZ_PATH, allow_pickle=False) as data:
        X = data["X"].copy()
        y = data["y"].astype(np.int64)
        splits = data["split"].copy()
        experiment_ids = data["experiment_ids"].copy()
        original_lengths = data["original_lengths"].astype(np.int64)

    if len(metadata) != len(X):
        raise ValueError("Metadata row count does not match sequence array.")
    if not np.array_equal(metadata["experiment"].astype(str).to_numpy(),
                          experiment_ids.astype(str)):
        raise ValueError("Metadata experiment order differs from NPZ.")
    if not np.array_equal(metadata["label"].to_numpy(), y):
        raise ValueError("Metadata labels differ from NPZ.")
    if not np.array_equal(metadata["split"].to_numpy(), splits):
        raise ValueError("Metadata split labels differ from NPZ.")

    train_indices = np.flatnonzero(splits == "train")
    test_indices = np.flatnonzero(splits == "test")

    if len(train_indices) != 40000 or len(test_indices) != 24747:
        raise ValueError("Unexpected train/test sizes.")
    if set(experiment_ids[train_indices]) & set(experiment_ids[test_indices]):
        raise ValueError("Train/test experiment overlap detected.")

    # Validation is drawn only from the official training split.
    fit_indices, val_indices = train_test_split(
        train_indices,
        test_size=VALIDATION_SIZE,
        random_state=SEED,
        stratify=y[train_indices],
    )

    X_tensor = torch.from_numpy(X.astype(np.int64))
    y_tensor = torch.from_numpy(y.astype(np.float32))
    lengths_tensor = torch.from_numpy(
        np.minimum(original_lengths, X.shape[1]).astype(np.int64)
    )

    def make_loader(indices, shuffle):
        ds = TensorDataset(
            X_tensor[indices],
            lengths_tensor[indices],
            y_tensor[indices],
        )
        return DataLoader(
            ds,
            batch_size=BATCH_SIZE,
            shuffle=shuffle,
            num_workers=0,
        )

    train_loader = make_loader(fit_indices, True)
    val_loader = make_loader(val_indices, False)
    test_loader = make_loader(test_indices, False)

    model = EventLSTM(
        vocab_size=max(int(v) for v in vocabulary.values()) + 1,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE
    )

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    patience_counter = 0
    history = []

    print("Device:", device)
    print("Vocabulary entries:", len(vocabulary))
    print("Training subset:", len(fit_indices))
    print("Validation subset:", len(val_indices))
    print("Official test subset:", len(test_indices))
    print("Model parameters:", sum(p.numel() for p in model.parameters()))
    print("Starting training...", flush=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for xb, lb, yb in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb.to(device), lb)
            loss = criterion(logits, yb.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss_sum += loss.item() * len(yb)
            train_count += len(yb)

        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        val_probs = []
        val_true = []

        with torch.no_grad():
            for xb, lb, yb in val_loader:
                logits = model(xb.to(device), lb)
                loss = criterion(logits, yb.to(device))
                val_loss_sum += loss.item() * len(yb)
                val_count += len(yb)
                val_probs.extend(torch.sigmoid(logits).cpu().numpy())
                val_true.extend(yb.numpy())

        train_loss = train_loss_sum / train_count
        val_loss = val_loss_sum / val_count
        val_pred = (np.asarray(val_probs) >= 0.5).astype(int)
        val_f1 = f1_score(val_true, val_pred, zero_division=0)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "validation_f1": val_f1,
        })

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_f1={val_f1:.4f}",
            flush=True,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print("Early stopping triggered.", flush=True)
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid model.")

    model.load_state_dict(best_state)
    model.eval()

    torch.save(
        {
            "model_state_dict": best_state,
            "vocabulary": vocabulary,
            "embedding_dim": EMBED_DIM,
            "hidden_dim": HIDDEN_DIM,
            "max_sequence_length": X.shape[1],
            "best_epoch": best_epoch,
            "seed": SEED,
        },
        MODEL_DIR / "dmbd_lstm_experiment_classifier.pt",
    )

    pd.DataFrame(history).to_csv(
        REPORT_DIR / "phase3_lstm_training_history.csv",
        index=False,
    )

    # The official test set is evaluated only after model selection.
    test_probs = []
    test_true = []

    with torch.no_grad():
        for xb, lb, yb in test_loader:
            logits = model(xb.to(device), lb)
            test_probs.extend(torch.sigmoid(logits).cpu().numpy())
            test_true.extend(yb.numpy().astype(int))

    test_probs = np.asarray(test_probs)
    test_true = np.asarray(test_true)
    test_pred = (test_probs >= 0.5).astype(int)
    cm = confusion_matrix(test_true, test_pred, labels=[0, 1])

    metrics = {
        "model": "PyTorch LSTM on event-type sequences",
        "prediction_target": "experiment-level benign/malicious label",
        "best_epoch": int(best_epoch),
        "threshold": 0.5,
        "test_samples": int(len(test_true)),
        "test_benign": int(np.sum(test_true == 0)),
        "test_malicious": int(np.sum(test_true == 1)),
        "accuracy": float(accuracy_score(test_true, test_pred)),
        "malicious_precision": float(
            precision_score(test_true, test_pred, pos_label=1, zero_division=0)
        ),
        "malicious_recall": float(
            recall_score(test_true, test_pred, pos_label=1, zero_division=0)
        ),
        "malicious_f1": float(
            f1_score(test_true, test_pred, pos_label=1, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(test_true, test_probs)),
        "confusion_matrix_labels_0_1": cm.tolist(),
        "class_0": "benign",
        "class_1": "malicious",
        "truncated_test_sequences": int(
            metadata.loc[test_indices, "truncated"].sum()
        ),
        "note": (
            "Labels are experiment-level, not individual-process ground truth. "
            "Sequences retain the first 512 events; longer sequences are truncated."
        ),
    }

    predictions = pd.DataFrame({
        "experiment": experiment_ids[test_indices],
        "true_label": test_true,
        "predicted_label": test_pred,
        "malicious_probability": test_probs,
        "original_length": original_lengths[test_indices],
    })
    predictions.to_csv(
        REPORT_DIR / "phase3_lstm_test_predictions.csv",
        index=False,
    )

    metrics_path = REPORT_DIR / "phase3_lstm_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    report_lines = [
        "# Phase 3 — LSTM Experiment-Level Classification",
        "",
        "## Experimental setup",
        "",
        "- Input: ordered event-type sequences, capped at 512 events.",
        "- Architecture: embedding + LSTM + dropout + binary classifier.",
        "- Validation: stratified 15% subset of the official training split.",
        "- Test: official test split, evaluated after validation-based early stopping.",
        "- Class 0: benign; class 1: malicious.",
        "- Labels describe experiments, not individual processes.",
        "- Longer sequences are truncated to the first 512 events.",
        "",
        "## Test metrics",
        "",
        f"- Accuracy: {metrics['accuracy']:.4f}",
        f"- Malicious precision: {metrics['malicious_precision']:.4f}",
        f"- Malicious recall: {metrics['malicious_recall']:.4f}",
        f"- Malicious F1: {metrics['malicious_f1']:.4f}",
        f"- ROC-AUC: {metrics['roc_auc']:.4f}",
        f"- Best epoch: {best_epoch}",
        "",
        "## Confusion matrix",
        "",
        "Rows are true labels; columns are predicted labels.",
        "",
        "| True label | Predicted benign (0) | Predicted malicious (1) |",
        "|---|---:|---:|",
        f"| Benign (0) | {cm[0,0]} | {cm[0,1]} |",
        f"| Malicious (1) | {cm[1,0]} | {cm[1,1]} |",
        "",
        "## Classification report",
        "",
        "```text",
        classification_report(
            test_true,
            test_pred,
            labels=[0, 1],
            target_names=["benign", "malicious"],
            zero_division=0,
        ).rstrip(),
        "```",
        "",
        "## Limitations",
        "",
        "- The ground truth is experiment-level, not process-level attribution.",
        "- The test set is imbalanced; accuracy should not be interpreted alone.",
        "- Only the first 512 events are retained for long sequences.",
        "- This is an experimental classification result, not evidence of deployment readiness.",
        "",
        "Metrics JSON: `reports/phase3_lstm_metrics.json`",
        "Training history: `reports/phase3_lstm_training_history.csv`",
        "Test predictions: `reports/phase3_lstm_test_predictions.csv`",
    ]

    report_path = REPORT_DIR / "phase3_lstm_experiment_classification_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("\nTest results:", flush=True)
    print(json.dumps(metrics, indent=2), flush=True)
    print("\nSaved model:", MODEL_DIR / "dmbd_lstm_experiment_classifier.pt")
    print("Saved report:", report_path)
    print("Saved metrics:", metrics_path)
    print("Saved history:", REPORT_DIR / "phase3_lstm_training_history.csv")
    print("Saved predictions:", REPORT_DIR / "phase3_lstm_test_predictions.csv")


if __name__ == "__main__":
    main()
