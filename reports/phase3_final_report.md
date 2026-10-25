# Phase 3: Dynamic Malware Behavior Classification Using Machine Learning

## 1. Executive Summary

Phase 3 investigates whether dynamic system-event telemetry can distinguish benign and malicious experiments using aggregated event-count features and ordered event-type sequences.

The study uses the Dynamic Malware Behavior Dataset (DMBD). Eight event files containing approximately 44.1 million telemetry events were processed and aggregated into 64,747 experiment-level records.

Two primary approaches were evaluated:
1. Random Forest classification using aggregated event-count features.
2. LSTM classification using ordered event-type sequences.

The existing seven-feature Random Forest achieved a test F1-score of 0.9503 and ROC-AUC of 0.9738. The LSTM achieved a test F1-score of 0.9101 and ROC-AUC of 0.8975.

Additional experiments investigated normalized event proportions, five-fold cross-validation, classification errors, and exploratory validation-set ensembles.

**Scope clarification:** The prediction target is the experiment-level benign/malicious label. These experiments do not establish ground-truth attribution of malicious behavior to individual processes.

## 2. Research Objectives

The objectives of Phase 3 are:

- Process dynamic telemetry events into structured experiment-level representations.
- Evaluate Random Forest classification using aggregated event counts.
- Evaluate an LSTM using ordered event-type sequences.
- Compare event counts with normalized event proportions.
- Analyze model errors and identify challenging experiment patterns.
- Document limitations and reproducibility considerations.

## 3. Dataset and Data Preparation

### 3.1 Dataset overview

The experiment used the Dynamic Malware Behavior Dataset (DMBD).

The processing pipeline covered eight event files, `trees_0.json` through `trees_7.json`.

| Dataset property | Observed value |
|---|---:|
| Telemetry event rows | 44,109,386 |
| Unique experiments | 64,747 |
| Event categories represented | 10 event types |
| Experiment-level feature records | 64,747 |
| Duplicate experiment identifiers in feature table | 0 |
| Missing values in feature table | 1 |

The feature table contains 12 columns: the experiment identifier, total event count, seven event-category counts, label, year, and split.

The one missing value is the `year` metadata for experiment `E42472`. Its event counts and class label are present. The missing year is not an input feature to either of the primary models described in this report.

### 3.2 Labels and data splits

The class mapping is:

- Class 0: benign
- Class 1: malicious

The official experiment splits contain:

| Split | Benign | Malicious | Total |
|---|---:|---:|---:|
| Training | 20,000 | 20,000 | 40,000 |
| Test | 5,249 | 19,498 | 24,747 |
| Total | 25,249 | 39,498 | 64,747 |

The training split is balanced, while the official test split is substantially more heavily weighted toward malicious experiments.

Consequently, accuracy alone is insufficient to characterize model performance. Precision, recall, F1-score, ROC-AUC, and confusion matrices are also reported.

### 3.3 Event categorization

The feature-extraction script maps event names into the following categories:

| Feature | Categorization rule |
|---|---|
| `process_create` | Exact event name `process_create` |
| `process_term` | Exact event name `process_term` |
| `image_load` | Exact event name `image_load` |
| `file_events` | Event names beginning with `file_` |
| `network_events` | Event names beginning with `net_` |
| `registry_events` | Event names beginning with `reg_` |
| `other_events` | Event names not covered by the rules above |

The total event count is also retained as `total_events`.

The extraction implementation validates experiment coverage, detects experiments repeated across event files, checks event-category totals against total event counts, rejects zero-event experiments, and validates a reference experiment.

The feature table contains one record per experiment.

## 4. Methodology

### 4.1 Random Forest

The saved primary Random Forest model uses seven features:

- `total_events`
- `process_create`
- `process_term`
- `image_load`
- `file_events`
- `network_events`
- `registry_events`

The saved estimator configuration is:

| Parameter | Value |
|---|---|
| Estimator | RandomForestClassifier |
| Number of trees | 300 |
| Minimum samples per leaf | 2 |
| Maximum depth | Unlimited |
| Number of input features | 7 |

The `other_events` category is excluded from this model because it does not provide useful variation in the current dataset.

### 4.2 LSTM sequence classifier

The LSTM uses ordered event-type sequences extracted from the original telemetry.

The sequence representation retains at most the first 512 events per experiment. Longer sequences are truncated.

The architecture consists of:

1. An embedding layer with dimension 16.
2. An LSTM layer with hidden size 32.
3. Dropout with probability 0.3.
4. A linear binary-classification output layer.

The model was trained using PyTorch, Adam optimization, a learning rate of 0.001, batch size 128, and binary cross-entropy with logits.

A stratified 15% validation subset was selected from the official training partition. The selected checkpoint was saved at epoch 12. The official test set was evaluated after validation-based model selection.

The test set contains 5,529 sequences longer than the 512-event limit that were truncated.

### 4.3 Evaluation metrics

The experiments use the following metrics:

- Accuracy: fraction of predictions that are correct.
- Precision: fraction of predicted malicious experiments that are actually malicious.
- Recall: fraction of malicious experiments that are detected.
- F1-score: harmonic mean of precision and recall.
- ROC-AUC: ranking performance across classification thresholds.
- Confusion matrix: counts of true negatives, false positives, false negatives, and true positives.

For the confusion matrices in this report, rows are true labels and columns are predicted labels, ordered as benign (0) followed by malicious (1).

## 5. Primary Test Results

### 5.1 Random Forest

The saved seven-feature Random Forest was evaluated on 24,747 official test experiments.

| Metric | Test result |
|---|---:|
| Accuracy | 0.9240 |
| Malicious precision | 0.9791 |
| Malicious recall | 0.9233 |
| Malicious F1-score | 0.9503 |
| ROC-AUC | 0.9738 |

Confusion matrix:

| True label | Predicted benign | Predicted malicious |
|---|---:|---:|
| Benign | 4,864 | 385 |
| Malicious | 1,496 | 18,002 |

The model correctly classified 22,866 of the 24,747 test experiments. It generated 385 false positives and 1,496 false negatives.

The high malicious precision indicates that most experiments flagged as malicious were malicious in this test set. However, the false negatives show that the model did not detect every malicious experiment.

### 5.2 LSTM

The LSTM was evaluated on the same 24,747 official test experiments.

| Metric | Test result |
|---|---:|
| Accuracy | 0.8630 |
| Malicious precision | 0.9421 |
| Malicious recall | 0.8802 |
| Malicious F1-score | 0.9101 |
| ROC-AUC | 0.8975 |
| Selected epoch | 12 |

Confusion matrix:

| True label | Predicted benign | Predicted malicious |
|---|---:|---:|
| Benign | 4,195 | 1,054 |
| Malicious | 2,336 | 17,162 |

The LSTM correctly classified 21,357 test experiments, with 1,054 false positives and 2,336 false negatives.

### 5.3 Primary model comparison

| Metric | Random Forest | LSTM |
|---|---:|---:|
| Accuracy | 0.9240 | 0.8630 |
| Precision | 0.9791 | 0.9421 |
| Recall | 0.9233 | 0.8802 |
| F1-score | 0.9503 | 0.9101 |
| ROC-AUC | 0.9738 | 0.8975 |
| False positives | 385 | 1,054 |
| False negatives | 1,496 | 2,336 |

The saved Random Forest outperformed the LSTM on all five reported metrics in this evaluation.

This comparison does not establish that Random Forest is universally superior to sequence models. The two approaches use different representations, and the LSTM has a 512-event truncation limit.

## 6. Event Counts Versus Event Proportions

A separate experiment compared two Random Forest feature representations:

1. Six event-category counts.
2. The same six counts plus normalized proportions for each category.

These experiments are distinct from the saved seven-feature primary model.

### 6.1 Official test comparison

| Metric | Counts only | Counts plus proportions |
|---|---:|---:|
| Accuracy | 0.9240 | 0.9246 |
| Precision | 0.9771 | 0.9770 |
| Recall | 0.9252 | 0.9261 |
| F1-score | 0.9504 | 0.9509 |
| ROC-AUC | 0.9721 | 0.9713 |

The observed test F1-score difference is very small. The combined representation has slightly higher test F1 and recall but slightly lower precision and ROC-AUC.

These test results should be interpreted descriptively, not used alone to select the feature representation.

### 6.2 Five-fold cross-validation

The separate cross-validation experiment compared the two feature representations across five folds.

| Metric | Counts only: mean | Counts only: standard deviation | Counts plus proportions: mean | Counts plus proportions: standard deviation |
|---|---:|---:|---:|---:|
| F1-score | 0.91357 | 0.00195 | 0.91256 | 0.00264 |
| ROC-AUC | 0.96773 | 0.00137 | 0.96733 | 0.00142 |

The counts-only representation performed slightly better on average in this cross-validation experiment.

The observed differences are small. The results do not demonstrate that normalized proportions are generally harmful, but they do not provide evidence of a meaningful improvement from adding them either.

## 7. Exploratory Validation Ensemble Comparison

Four approaches were compared on the same 6,000 validation experiments, comprising 3,000 benign and 3,000 malicious examples.

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.9382 | 0.9525 | 0.9223 | 0.9372 | 0.9860 |
| LSTM | 0.8447 | 0.8566 | 0.8280 | 0.8420 | 0.8982 |
| Probability average | 0.9140 | 0.9442 | 0.8800 | 0.9110 | 0.9710 |
| OR rule | 0.8830 | 0.8516 | 0.9277 | 0.8880 | 0.9636 |

The probability-average method averages the Random Forest and LSTM probabilities before applying a 0.5 threshold.

The OR rule predicts malicious when either model's probability reaches 0.5. Its ranking score uses the maximum of the two model probabilities.

The OR rule achieved the highest recall among these validation approaches, but its precision was lower. Probability averaging did not outperform Random Forest alone.

**Important limitation:** The saved LSTM checkpoint was selected using this same validation split. Therefore, these ensemble comparisons are exploratory and are not independent estimates of generalization performance. The official test data was not used to select the ensemble rule.

## 8. Error Analysis

### 8.1 Random Forest errors

The primary Random Forest produced:

- 1,496 false negatives.
- 385 false positives.

Analysis by original experiment length showed that the false negatives were concentrated in short sequences:

- 1,104 false negatives had 1–64 events.
- 231 had 65–128 events.
- 83 had 129–256 events.
- 78 had more than 256 events.

Thus, 1,335 of 1,496 false negatives, approximately 89.2%, had no more than 128 events.

Among malicious experiments, false negatives also tended to have lower average event counts than correctly detected malicious experiments.

These observations suggest that sparse telemetry is associated with more classification errors in this dataset. They do not establish that sequence length alone causes the errors.

### 8.2 LSTM errors

The LSTM produced:

- 2,336 false negatives.
- 1,054 false positives.

Of its false negatives, 1,564 had original sequences of 64 events or fewer.

The LSTM therefore also struggled with a subset of short malicious experiments. This is consistent with the broader observation that sparse experiments are challenging, although the two models do not make identical errors.

### 8.3 Error complementarity

Comparing the saved Random Forest and LSTM test predictions showed that their errors were not identical.

Among malicious experiments:

- 1,247 were missed by both models.
- 249 were missed only by Random Forest.
- 1,089 were missed only by LSTM.
- 16,913 were detected by both.

Among benign experiments:

- 214 were incorrectly flagged by both models.
- 171 were false positives only for Random Forest.
- 840 were false positives only for LSTM.

These observations suggest that the models capture partly different patterns. They do not establish that an ensemble would improve performance. Ensemble selection would require a prespecified decision rule and validation-only evaluation.

## 9. Data Quality and Threats to Validity

### 9.1 Experiment-level labels

The ground truth describes experiments, not individual processes.

Consequently, the current results support experiment-level benign/malicious classification only. They do not validate which specific process initiated or performed malicious behavior.

Individual-process attribution would require suitable process-level ground truth and a separately validated methodology.

### 9.2 Class imbalance

The official test set contains 19,498 malicious experiments and 5,249 benign experiments.

Performance estimates depend on the dataset's class distribution and may change under different deployment prevalence. Precision in particular can change substantially when the prevalence of malicious activity changes.

### 9.3 Sequence truncation

The LSTM retains at most the first 512 events of each experiment. This removes later events from longer experiments and may omit behavior relevant to classification.

### 9.4 Generalization

High test performance on this dataset does not demonstrate robustness to new malware families, unseen environments, different telemetry collection policies, or changing attack behavior.

Additional independent datasets and carefully designed distribution-shift experiments would be needed to support broader generalization claims.

### 9.5 Feature representation

The Random Forest uses aggregate counts rather than the original event order. The LSTM uses ordered event types but does not incorporate the full range of event attributes in the current sequence representation.

The results therefore compare two particular representations and implementations, not every possible tree-based or recurrent model.

### 9.6 Missing metadata

One experiment has a missing `year` value. The missing field is metadata rather than a feature used by the two primary classifiers.

### 9.7 Validation-based model selection

The LSTM checkpoint was selected using the same validation split used in the exploratory ensemble comparison. The resulting validation metrics should not be interpreted as independent unbiased estimates.

### 9.8 Operational readiness

These are experimental results. No claim of production readiness, real-time detection capability, or validated process-level attribution is established by these experiments.

## 10. Reproducibility and Project Artifacts

The following scripts and outputs support the current experiments.

### Feature extraction and auditing

- `src/audit_dmbd_experiments.py`
- `src/count_dmbd_events.py`
- `src/inspect_dmbd_events.py`
- `src/extract_dmbd_features.py`
- `src/extract_dmbd_all_features.py`
- `data/processed/process_attribution/dmbd_all_events_experiment_features.csv`

### LSTM

- `src/extract_dmbd_lstm_sequences.py`
- `src/train_dmbd_lstm.py`
- `data/processed/process_attribution/dmbd_lstm_sequences_512.npz`
- `data/processed/process_attribution/dmbd_lstm_sequences_512_metadata.csv`
- `data/processed/process_attribution/dmbd_lstm_event_vocabulary.json`
- `models/dmbd_lstm_experiment_classifier.pt`
- `reports/phase3_lstm_metrics.json`
- `reports/phase3_lstm_experiment_classification_report.md`
- `reports/phase3_lstm_training_history.csv`
- `reports/phase3_lstm_test_predictions.csv`

### Random Forest and comparisons

- `reports/dmbd_experiment_random_forest.joblib`
- `reports/dmbd_feature_importance.csv`
- `reports/phase3_rf_test_predictions.csv`
- `src/compare_dmbd_rf_proportions.py`
- `src/validate_dmbd_rf_proportions.py`
- `reports/phase3_rf_count_proportion_comparison.json`
- `reports/phase3_rf_proportion_cv.json`
- `src/compare_dmbd_validation_ensemble.py`
- `reports/phase3_validation_ensemble_comparison.json`
- `reports/phase3_validation_ensemble_predictions.csv`

The scripts rely on the local DMBD files under `D:\dmdb` and the project environment's Python packages. Reproduction should preserve the official train/test split and the validation split used by the scripts.

## 11. Conclusion

Phase 3 demonstrates that dynamic telemetry aggregated into experiment-level event counts can be effective for distinguishing benign and malicious experiments in the evaluated dataset.

The saved seven-feature Random Forest achieved an F1-score of 0.9503 and ROC-AUC of 0.9738 on the official test set. The LSTM sequence model achieved an F1-score of 0.9101 and ROC-AUC of 0.8975.

Five-fold cross-validation slightly favored event counts alone over adding normalized event proportions. Exploratory validation ensemble experiments did not show an improvement over Random Forest alone.

Error analysis found that many false negatives occurred in short or sparse experiments. This is a useful direction for further investigation, not proof of a causal relationship.

The primary limitation is the experiment-level ground truth. Phase 3 does not yet establish individual-process attribution. Further work should prioritize process-level labels, independent-dataset evaluation, robustness testing, and a carefully defined operational evaluation before any deployment claims are made.

---

## Final status

The primary Random Forest and LSTM experiments, feature-representation comparisons, cross-validation, exploratory ensemble evaluation, and error analyses have been completed.

This report consolidates the current findings. It does not claim that all possible robustness tests or individual-process attribution experiments have been completed.

Existing Phase 1 and Phase 2 artifacts are unchanged.
