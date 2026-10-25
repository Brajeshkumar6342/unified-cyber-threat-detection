# EMBER 2018 — Full Official Test Evaluation

## Experiment details
- Model: Random Forest
- Model file: `models/malware/ember_random_forest_full.joblib`
- Dataset: EMBER 2018 `test_features.jsonl`
- Evaluation records: 200,000
- Benign records: 100,000
- Malicious records: 100,000
- Features: 531
- Decision threshold: 0.5
- Evaluation time: 39.85 seconds

## Metrics
| Metric | Result |
|---|---:|
| Accuracy | 0.924075 |
| Malicious precision | 0.923157 |
| Malicious recall | 0.925160 |
| Malicious F1-score | 0.924157 |
| ROC-AUC | 0.977696 |

## Confusion matrix
Rows are actual labels; columns are predicted labels.

| Actual class | Predicted benign | Predicted malicious |
|---|---:|---:|
| Benign | 92,299 | 7,701 |
| Malicious | 7,484 | 92,516 |

## Interpretation
The model achieved 92.41% accuracy and a malicious-class F1-score of 92.42% on all 200,000 labeled records in the official EMBER 2018 test file. ROC-AUC was 0.9777.

The model detected 92,516 of 100,000 malicious records and missed 7,484. It also incorrectly flagged 7,701 benign records as malicious.

## Methodological limitations
- This experiment uses a selected 531-feature representation: 256 histogram bins, 256 byte-entropy bins, and 19 numeric fields from the `general` and `strings` sections. It is not the complete EMBER feature representation.
- Non-numeric fields within the selected sections were represented as 0.0.
- These results evaluate malware classification on EMBER 2018; they do not establish real-world deployment performance.
- Comparisons against earlier test samples must account for differences in sample selection.
- The classification threshold was 0.5; alternative thresholds have not yet been evaluated.

## Result source
Machine-readable results: `reports/ember_full_random_forest_full_test.json`
