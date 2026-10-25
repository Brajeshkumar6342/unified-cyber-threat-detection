# EMBER 2018 — Classification Threshold Analysis

## Objective
Evaluate the effect of classification thresholds on the full-data
Random Forest malware classifier using the reconstructed validation set.

## Experimental setup
- Dataset: EMBER 2018 labeled training records
- Total labeled records: 600,000
- Training partition: 480,000
- Validation partition: 120,000
- Features: 531
- Split: stratified 80/20
- Random state: 42
- Model: ember_random_forest_full.joblib
- Validation ROC-AUC: 0.9903

## Threshold results

| Threshold | Accuracy | Precision | Recall | F1-score | FPR | FNR |
|---:|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.9016 | 0.8412 | 0.9901 | 0.9096 | 0.1869 | 0.0099 |
| 0.30 | 0.9313 | 0.8943 | 0.9784 | 0.9344 | 0.1157 | 0.0216 |
| 0.40 | 0.9437 | 0.9276 | 0.9626 | 0.9448 | 0.0752 | 0.0374 |
| 0.50 | 0.9478 | 0.9533 | 0.9417 | 0.9475 | 0.0461 | 0.0583 |
| 0.60 | 0.9433 | 0.9728 | 0.9121 | 0.9415 | 0.0255 | 0.0879 |
| 0.70 | 0.9289 | 0.9872 | 0.8691 | 0.9244 | 0.0113 | 0.1309 |
| 0.80 | 0.9004 | 0.9958 | 0.8041 | 0.8897 | 0.0034 | 0.1959 |

## Findings
- Threshold 0.50 achieved the highest validation accuracy (94.78%)
  and F1-score (94.75%) among the tested thresholds.
- Threshold 0.20 achieved the highest malware recall (99.01%),
  but had a false-positive rate of 18.69%.
- Threshold 0.40 achieved 96.26% malware recall with a 7.52%
  false-positive rate.
- Threshold 0.80 achieved 99.58% precision but only 80.41% recall.
- Increasing the threshold generally reduced false positives while
  increasing false negatives.

## Provisional decision
Retain threshold 0.50 as the default for subsequent experiments.
This choice is based on its validation accuracy and F1-score.
Alternative thresholds may be preferred when the relative costs of
false negatives and false positives differ.

## Limitations
- Only seven thresholds were tested.
- Threshold selection and reported metrics use the same validation set;
  a separate validation set would be preferable for unbiased assessment
  after threshold selection.
- The official EMBER test set remains reserved for final evaluation.
- Results apply to the selected 531-feature representation, not the
  complete EMBER feature representation.
- Validation results do not establish deployment readiness.

## Machine-readable results
- reports/ember_threshold_analysis.json
- reports/ember_threshold_analysis.csv
