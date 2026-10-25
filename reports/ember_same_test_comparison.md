# EMBER 2018: Baseline vs Full-Data Random Forest

## Objective
Compare the baseline Random Forest and the full-data Random Forest
on the same complete, labeled EMBER 2018 official test dataset.

## Dataset
- Test dataset: EMBER 2018 official test_features.jsonl
- Total labeled test records: 200,000
- Benign records: 100,000
- Malicious records: 100,000
- Features: 531
- Decision threshold: 0.5

## Model comparison

| Metric | Baseline | Full-data Random Forest |
|---|---:|---:|
| Accuracy | 0.826890 | 0.924075 |
| Malicious precision | 0.901782 | 0.923157 |
| Malicious recall | 0.733690 | 0.925160 |
| Malicious F1-score | 0.809098 | 0.924157 |
| ROC-AUC | 0.898862 | 0.977696 |

## Confusion matrices

Rows represent actual labels; columns represent predicted labels.
Label 0 = benign; label 1 = malicious.

### Baseline Random Forest
| Actual class | Predicted benign | Predicted malicious |
|---|---:|---:|
| Benign | 92,009 | 7,991 |
| Malicious | 26,631 | 73,369 |

### Full-data Random Forest
| Actual class | Predicted benign | Predicted malicious |
|---|---:|---:|
| Benign | 92,299 | 7,701 |
| Malicious | 7,484 | 92,516 |

## Findings
- Accuracy improved by 9.7185 percentage points.
- Malicious precision improved by approximately 2.14 percentage points.
- Malicious recall improved by approximately 19.15 percentage points.
- Malicious F1-score improved by approximately 11.51 percentage points.
- ROC-AUC improved by approximately 7.88 percentage points.
- False negatives decreased from 26,631 to 7,484, a reduction of approximately 71.9%.
- False positives decreased from 7,991 to 7,701.

## Interpretation
The full-data Random Forest performed substantially better than the baseline
on the same official test records, particularly in malicious recall and F1-score.

The baseline and full-data models differ in training-data quantity and training
configuration. Consequently, the experiment compares the two complete model
setups and does not isolate training-data size as the only cause of improvement.

## Limitations
- The experiment uses a selected 531-feature representation rather than the
  complete EMBER feature representation.
- Non-numeric values within the selected general and strings fields were
  represented as 0.0.
- The test set is balanced, with equal numbers of benign and malicious records.
  Real-world class prevalence may differ.
- Test-set performance does not establish deployment readiness or generalization
  to malware families and environments outside the evaluation dataset.
- Threshold tuning and analysis of individual errors have not yet been completed.

## Reproducibility
Machine-readable comparison:
reports/ember_same_test_comparison.json

Full official test results:
reports/ember_full_random_forest_full_test.json

Earlier baseline and full-model results are retained separately.
