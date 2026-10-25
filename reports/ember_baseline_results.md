# EMBER Malware Classification — Baseline Experiment

## Dataset
- Dataset: EMBER 2018 static PE features
- Pilot sample size: 2,000
- Benign samples: 1,000
- Malicious samples: 1,000
- Numerical features: 531
- Training samples: 1,600
- Test samples: 400
- Train/test split: 80/20
- Random seed: 42

## Model
- Algorithm: Random Forest Classifier
- Number of trees: 100
- Class weighting: balanced

## Test Results
- Accuracy: 0.8950
- Precision: 0.9202
- Recall: 0.8650
- F1-score: 0.8918
- ROC-AUC: 0.9577

## Confusion Matrix
Rows are actual labels; columns are predicted labels.

| Actual / Predicted | Benign | Malicious |
|---|---:|---:|
| Benign | 185 | 15 |
| Malicious | 27 | 173 |

## Limitations
- This is a small pilot experiment, not a final evaluation.
- Samples were taken from one training file.
- Features were restricted to selected numeric values from histogram,
  byteentropy, general, and strings fields.
- The test split is drawn from the same source file as the training split.
- No independent test dataset has been evaluated.
- The results do not establish real-world malware detection performance.
