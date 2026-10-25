# Phase 3 — LSTM Experiment-Level Classification

## Experimental setup

- Input: ordered event-type sequences, capped at 512 events.
- Architecture: embedding + LSTM + dropout + binary classifier.
- Validation: stratified 15% subset of the official training split.
- Test: official test split, evaluated after validation-based early stopping.
- Class 0: benign; class 1: malicious.
- Labels describe experiments, not individual processes.
- Longer sequences are truncated to the first 512 events.

## Test metrics

- Accuracy: 0.8630
- Malicious precision: 0.9421
- Malicious recall: 0.8802
- Malicious F1: 0.9101
- ROC-AUC: 0.8975
- Best epoch: 12

## Confusion matrix

Rows are true labels; columns are predicted labels.

| True label | Predicted benign (0) | Predicted malicious (1) |
|---|---:|---:|
| Benign (0) | 4195 | 1054 |
| Malicious (1) | 2336 | 17162 |

## Classification report

```text
              precision    recall  f1-score   support

      benign       0.64      0.80      0.71      5249
   malicious       0.94      0.88      0.91     19498

    accuracy                           0.86     24747
   macro avg       0.79      0.84      0.81     24747
weighted avg       0.88      0.86      0.87     24747
```

## Limitations

- The ground truth is experiment-level, not process-level attribution.
- The test set is imbalanced; accuracy should not be interpreted alone.
- Only the first 512 events are retained for long sequences.
- This is an experimental classification result, not evidence of deployment readiness.

Metrics JSON: `reports/phase3_lstm_metrics.json`
Training history: `reports/phase3_lstm_training_history.csv`
Test predictions: `reports/phase3_lstm_test_predictions.csv`
