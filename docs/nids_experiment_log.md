
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

## 15. Initial Five-Feature Cross-Dataset Transfer Experiment

### Objective

Evaluate whether a Random Forest trained on UNSW-NB15 can generalize to the CICIDS2017 Friday DDoS subset using five candidate common features.

### Method

- Source: `UNSW_NB15_training-set.csv`
- Source split: stratified 80/20 training/validation split, random state 42
- Source training rows: 140,272
- Source validation rows: 35,069
- Target: `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`
- Target evaluation rows: 225,745
- Features: duration, forward packets, backward packets, forward bytes, backward bytes
- Model: Random Forest with 200 estimators, `max_features="sqrt"`, `min_samples_leaf=2`, random state 42
- Preprocessing: median imputation fitted within a pipeline on source training data only
- Target duration: converted from microseconds to seconds; the two negative duration values were marked missing and imputed using the source-fitted median
- The official UNSW-NB15 test set was not used.

### Results

| Metric | UNSW-NB15 validation | CICIDS2017 transfer |
|---|---:|---:|
| Accuracy | 0.9460 | 0.5160 |
| Attack precision | 0.9502 | 0.5651 |
| Attack recall | 0.9717 | 0.6357 |
| Attack F1-score | 0.9608 | 0.5983 |
| ROC-AUC | 0.9889 | 0.5548 |
| True negatives | 9,984 | 35,095 |
| False positives | 1,216 | 62,623 |
| False negatives | 676 | 46,642 |
| True positives | 23,193 | 81,385 |

### Interpretation

The model performed substantially worse on CICIDS2017 than on the held-out UNSW-NB15 validation split. The CICIDS2017 false-positive rate was approximately 64.12%, and the ROC-AUC was 0.5548.

Possible explanations include differences in feature distributions, flow-direction conventions, feature definitions, attack composition, and the restricted five-feature representation. This experiment does not isolate which explanation is responsible.

### Limitations

The four packet/byte mappings remain candidate mappings, not fully verified equivalences. The target evaluation is restricted to the Friday DDoS subset and does not establish generalization to all CICIDS2017 attack categories.

This result is an initial cross-dataset baseline. CICIDS2017 results must not be used to tune this source-trained model in subsequent transfer experiments; any changes should be evaluated using a clearly separated experimental protocol.

### 16. Cross-Dataset Feature Distribution Shift Analysis

**Objective:** Investigate differences in candidate common-feature distributions between UNSW-NB15 and CICIDS2017 to help explain the poor cross-dataset transfer performance.

**Analysis script:** `src/nids/analyze_cross_dataset_shift.py`

**Output report:** `reports/tables/cross_dataset_feature_shift.csv`

**Methodology:**
- Compared five candidate common features: duration, forward packets, backward packets, forward bytes, and backward bytes.
- Used the UNSW-NB15 training CSV and the CICIDS2017 Friday DDoS CSV.
- Converted CICIDS2017 flow duration from microseconds to seconds.
- Treated negative CICIDS2017 durations as missing in the derived analysis.
- Calculated count, missing values, mean, standard deviation, minimum, median, 95th percentile, 99th percentile, and maximum.
- This was a descriptive analysis; no model tuning was performed using target evaluation labels.

**Key findings:**

| Feature | UNSW-NB15 median | CICIDS2017 median |
|---|---:|---:|
| Duration (seconds) | 0.001582 | 1.45234 |
| Forward packets | 2 | 3 |
| Backward packets | 2 | 4 |
| Forward bytes | 430 | 30 |
| Backward bytes | 164 | 164 |

Duration and forward-byte distributions show substantial differences between the datasets. Packet-count medians also differ, while equal backward-byte medians do not imply equivalent distributions.

**Interpretation:** The observed differences are consistent with substantial feature-distribution shift and may contribute to poor cross-dataset generalization. However, this descriptive comparison does not establish the cause of the transfer failure. Differences in feature definitions, traffic characteristics, class composition, and dataset collection procedures require further investigation.

**Limitations:**
- The analysis compares overall dataset distributions rather than distributions within each class.
- Candidate semantic feature mappings have not been fully validated as equivalent.
- The results do not establish which features are responsible for the model's errors.
- The CICIDS2017 target evaluation results remain diagnostic and must not be used to tune the model if they are to serve as a held-out evaluation.

**Next step:** Investigate feature definitions and compare source-training and source-validation distributions. Design any further model-selection procedure using source training and validation data, keeping the target evaluation set separate from model selection.