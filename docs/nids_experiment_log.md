
# NIDS Experiment Log

## 1. Project Objective

Develop a machine-learning-based Network Intrusion Detection System (NIDS) that classifies network traffic as benign or attack traffic. This experiment establishes a reproducible Random Forest baseline and compares alternative configurations.

## 2. Dataset

- **Dataset:** UNSW-NB15
- **Training records:** 175,341
- **Official testing records:** 82,332
- **Original columns:** 45
- **Processed feature count:** 194

The original dataset's `id`, `label`, and `attack_cat` columns are excluded from model inputs. The target is `label`, where `0` represents benign traffic and `1` represents attack traffic.

The official training/testing split is preserved during preprocessing.

## 3. Preprocessing

The preprocessing pipeline performs the following operations:

1. Separates features from the target label.
2. Excludes the identifier and attack-category columns from model inputs.
3. Imputes missing numeric values using the median.
4. Standardizes numeric features.
5. Imputes missing categorical values using the most frequent value.
6. One-hot encodes categorical features.
7. Fits preprocessing transformations on training data and applies them to the official test data.
8. Saves the transformed features and labels for reuse.

The processed feature matrix contains 194 features.

## 4. Baseline Model

**Algorithm:** Random Forest Classifier

| Parameter | Value |
|---|---|
| Number of trees | 100 |
| Maximum features | `sqrt` |
| Random state | 42 |
| Parallel jobs | -1 |

### Official test-set baseline results

These results were obtained from the initial baseline experiment.

| Metric | Result |
|---|---:|
| Accuracy | 87.05% |
| Attack precision | 81.74% |
| Attack recall | 98.49% |
| Attack F1-score | 89.33% |
| ROC-AUC | 97.80% |
| False positives | 9,975 |
| False negatives | 686 |

Confusion matrix (rows = actual, columns = predicted):

| Actual class | Predicted benign | Predicted attack |
|---|---:|---:|
| Benign | 27,025 | 9,975 |
| Attack | 686 | 44,646 |

The baseline detected most attacks but generated a substantial number of false positives.

## 5. Validation Methodology

To compare model configurations, the original training data was divided into:

- **Training subset:** 80% (140,272 records)
- **Validation subset:** 20% (35,069 records)

The split uses stratification and `random_state=42` for reproducibility.

The official test set is not loaded by the validation experiment scripts.

## 6. Validation Experiments

Three Random Forest configurations were evaluated on the same validation split.

| Configuration | Trees | Minimum samples per leaf | Class weighting |
|---|---:|---:|---|
| RF_100_sqrt | 100 | 1 | None |
| RF_200_sqrt_leaf2 | 200 | 2 | None |
| RF_100_sqrt_balanced | 100 | 1 | Balanced |

### Validation results

| Metric | RF_100_sqrt | RF_200_sqrt_leaf2 | RF_100_sqrt_balanced |
|---|---:|---:|---:|
| Accuracy | 95.91% | 95.91% | 95.70% |
| Attack precision | 96.36% | 95.98% | 97.43% |
| Attack recall | 97.68% | 98.11% | 96.22% |
| Attack F1-score | 97.01% | 97.03% | 96.82% |
| ROC-AUC | 99.36% | 99.38% | 99.37% |
| False-positive rate | 7.87% | 8.77% | 5.42% |
| False positives | 881 | 982 | 607 |
| False negatives | 554 | 451 | 902 |
| Training time | 9.58 s | 13.06 s | 9.14 s |

## 7. Interpretation

### RF_200_sqrt_leaf2

This configuration achieved the highest attack recall (98.11%) and attack F1-score (97.03%) among the three experiments. It produced fewer false negatives but more false positives than the original validation configuration.

### RF_100_sqrt_balanced

This configuration achieved the lowest false-positive rate (5.42%) and highest attack precision (97.43%). However, it produced more false negatives than the other configurations.

### Main observation

The experiments show a trade-off between detecting attacks and reducing false alarms. Model selection should consider the intended operational requirements instead of relying on accuracy alone.

These results are validation results. They should not be presented as final test-set performance.

## 8. Fuzzers Error Analysis

The initial official test-set error analysis reported:

- Total Fuzzers records: 6,062
- Correctly detected Fuzzers records: 5,431
- Missed Fuzzers records: 631
- Fuzzers detection rate: 89.59%
- Fuzzers miss rate: 10.41%

Fuzzers represented 631 of the baseline's 686 false negatives.

Descriptive comparisons showed differences in median values for several features, including `dtcpb`, `stcpb`, `sload`, `dload`, and `smean`.

These differences are exploratory observations. They do not establish causation, and the official test set must not be used for iterative model tuning.

## 9. Current Limitations

1. Only Random Forest configurations have been compared in these experiments.
2. Validation performance may differ from performance on unseen external traffic.
3. The UNSW-NB15 dataset may not represent all modern network environments.
4. Feature importance and median differences do not establish causal relationships.
5. No final tuned-model evaluation has yet been completed.
6. The current experiments cover binary benign-versus-attack classification, not the complete unified framework.

## 10. Planned Next Steps

1. Define model-selection criteria based on attack recall, false-positive rate, F1-score, and operational requirements.
2. Investigate additional configurations using training/validation data only.
3. Select a candidate model using validation results.
4. Retrain the selected configuration on the full official training set.
5. Evaluate the final candidate on the official test set once the selection decisions are complete.
6. Compare the final candidate with the original baseline using the same official test set.
7. Extend the project to static malware classification and process-level threat attribution.
8. Integrate the validated components into the unified threat detection framework.


## 11. Additional Tree-Ensemble Experiments

### Random Forest vs. Extra Trees

Both models were trained using the raw UNSW-NB15 training CSV. Preprocessing was fitted only on the training fold, with a stratified 80/20 split using random state 42.

| Metric | Random Forest | Extra Trees |
|---|---:|---:|
| Attack precision | 96.06% | 92.97% |
| Attack recall | 98.09% | 99.71% |
| Attack F1-score | 97.06% | 96.22% |
| ROC-AUC | 99.39% | 99.25% |
| False-positive rate | 8.57% | 16.08% |
| False positives | 960 | 1,801 |
| False negatives | 456 | 69 |

Extra Trees achieved higher attack recall but generated substantially more false positives. Random Forest provided the better overall F1-score.

### Threshold Comparison

Thresholds from 0.30 to 0.70 were evaluated for both models on the same validation split.

Neither model met both experimental targets: attack recall of at least 98% and false-positive rate of at most 5%.

For Random Forest, increasing the threshold reduced false positives but also reduced attack recall. These results demonstrate the trade-off between detection sensitivity and false alarms.

## 12. Feature-Selection Experiments

### Single-Split Comparison

Feature selection was performed using feature importance learned from the training fold.

| Configuration | Features | Attack F1-score | Attack recall | False-positive rate | False positives |
|---|---:|---:|---:|---:|---:|
| All features | 192 | 97.06% | 98.09% | 8.57% | 960 |
| Top 100 | 100 | 97.09% | 97.99% | 8.23% | 922 |
| Top 50 | 50 | 97.18% | 98.04% | 7.97% | 893 |

The top-50 configuration improved the validation F1-score slightly and reduced false positives compared with the full feature set.

### Three-Split Stability Experiment

Three stratified 80/20 validation splits were evaluated using random states 42, 21, and 7. Feature ranking was learned separately from each training fold.

| Mean metric | All features | Top 50 features |
|---|---:|---:|
| Attack precision | 95.90% | 96.19% |
| Attack recall | 98.26% | 98.08% |
| Attack F1-score | 97.07% | 97.13% |
| ROC-AUC | 99.39% | 99.41% |
| False-positive rate | 8.96% | 8.29% |
| Mean false positives | 1,004 | 928 |
| Mean false negatives | 415 | 458 |

Across these three splits, selecting 50 features produced a small average improvement in F1-score and fewer false positives, but also slightly lower attack recall.

These splits share the same underlying dataset and are not independent external tests. The results support further investigation rather than a definitive claim of improved generalization.

## 13. Updated Research Limitations

1. The current experiments use UNSW-NB15 and do not establish cross-dataset generalization.
2. The feature-selection improvement is modest and has been assessed on only three validation splits.
3. The proposed recall and false-positive-rate targets have not been achieved simultaneously by the tested configurations.
4. Results from different datasets and research papers may not be directly comparable because their tasks, splits, and evaluation protocols differ.
5. Static malware classification and process-level threat attribution remain to be implemented and evaluated.

## 14. Next Research Priorities

1. Document the validation findings and preserve experiment reproducibility.
2. Investigate cross-dataset generalization using a second NIDS dataset.
3. Select and freeze the NIDS candidate using training and validation evidence.
4. Evaluate the selected candidate on the reserved official UNSW-NB15 test set once.
5. Develop the static malware classification component.
6. Investigate process-level threat attribution and integrate the three components.
