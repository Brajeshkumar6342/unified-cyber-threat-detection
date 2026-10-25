# Unified Cyber Threat Detection

A cybersecurity research prototype investigating three related tasks:

1. Network Intrusion Detection (NIDS)
2. Static Malware Classification
3. Dynamic Malware Behavior Analysis

The project evaluates machine-learning approaches using established datasets, documents experimental results, and investigates the limitations of generalization and behavioral classification.

## Research Phases

### Phase 1: Network Intrusion Detection

**Objective:** Evaluate network intrusion detection and investigate cross-dataset generalization.

Key findings:
- Evaluates a four-feature approach across datasets.
- Examines feature mapping and the effects of dataset shift.
- Cross-dataset mean F1-score: 0.6987 +/- 0.0167.
- Cross-dataset mean ROC-AUC: 0.6373 +/- 0.0061.
- Official UNSW-NB15 test F1-score: 0.8536.
- Official UNSW-NB15 test ROC-AUC: 0.8719.

Documentation:
- Results: `reports/final_nids_results.md`
- Experiment log: `docs/nids_experiment_log.md`
- Feature mapping: `docs/cross_dataset_feature_mapping.md`

### Phase 2: Static Malware Classification

**Objective:** Classify executable files as benign or malicious using static features.

Dataset and approach:
- Dataset: EMBER 2018.
- Features: 531 numeric features.
- Model: Random Forest.
- Evaluation: 200,000 examples in the official test set.

Full Random Forest test results:

| Metric | Result |
|---|---:|
| Accuracy | 0.9241 |
| Precision | 0.9232 |
| Recall | 0.9252 |
| F1-score | 0.9242 |
| ROC-AUC | 0.9777 |

These results describe performance on the evaluated EMBER test set. They do not establish equivalent performance on other malware datasets or in a production environment.

Documentation:
- Main report: `reports/phase2_ember_malware_classification_report.md`
- Full-test results: `reports/ember_full_random_forest_full_test.md`
- Same-test model comparison: `reports/ember_same_test_comparison.md`
- Threshold analysis: `reports/ember_threshold_analysis.md`

### Phase 3: Dynamic Malware Behavior Analysis

**Objective:** Investigate experiment-level malware classification using dynamic process and system-event telemetry.

Dataset and approach:
- Dataset: DMBD.
- Unit of classification: experiment.
- Models: event-count Random Forest and sequence-based LSTM.
- The Random Forest uses event-count features.
- The LSTM processes event sequences with a maximum sequence length of 512 events.

#### Random Forest Results

| Metric | Result |
|---|---:|
| Test examples | 24,747 |
| Accuracy | 0.9240 |
| Malicious-class precision | 0.9791 |
| Malicious-class recall | 0.9233 |
| Malicious-class F1-score | 0.9503 |
| ROC-AUC | 0.9738 |

Confusion matrix, in `[[TN, FP], [FN, TP]]` order:

```text
[[4864,  385],
 [1496, 18002]]
```

#### LSTM Results

| Metric | Result |
|---|---:|
| Test examples | 24,747 |
| Accuracy | 0.8630 |
| Malicious-class precision | 0.9421 |
| Malicious-class recall | 0.8802 |
| Malicious-class F1-score | 0.9101 |
| ROC-AUC | 0.8975 |

Confusion matrix, in `[[TN, FP], [FN, TP]]` order:

```text
[[4195, 1054],
 [2336, 17162]]
```

On the reported test metrics, the Random Forest outperforms the LSTM. The results also show that shorter event sequences are a significant challenge for detection.

Documentation:
- Final report: `reports/phase3_final_report.md`
- LSTM report: `reports/phase3_lstm_experiment_classification_report.md`
- LSTM metrics: `reports/phase3_lstm_metrics.json`
- Validation ensemble comparison: `reports/phase3_validation_ensemble_comparison.json`
- Feature importance: `reports/dmbd_feature_importance.csv`

## Important Research Limitations

- **Dataset dependence:** Performance on one dataset does not guarantee performance on another.
- **Dataset shift:** Differences in features, collection methods, and data distributions can reduce cross-dataset generalization.
- **Experiment-level labels:** DMBD labels apply to experiments, not individual processes. Phase 3 results therefore do not establish ground-truth attribution of a specific malicious process.
- **Sequence truncation:** The LSTM uses sequences capped at 512 events, so some experiments have incomplete event histories.
- **Model comparison:** Random Forest and LSTM results should be interpreted in the context of their respective feature representations and experimental procedures.
- **Exploratory experiments:** Feature-proportion and ensemble comparisons are exploratory. Validation data should guide model selection; repeated test-set comparisons can lead to overfitting to the evaluation process.
- **Research prototype:** The reported metrics do not establish production readiness, real-world operational effectiveness, or suitability for automated security decisions.

## Repository Structure

```text
unified-cyber-threat-detection/
|-- data/
|   |-- raw/
|   `-- processed/
|-- docs/
|-- evaluation/
|-- experiments/
|-- models/
|-- reports/
|-- src/
|-- tests/
|-- requirements.txt
`-- README.md
```

Key directories:
- `src/` - data preparation, feature extraction, model training, evaluation, and analysis scripts.
- `data/` - raw and processed data locations; datasets may need to be obtained separately.
- `models/` - model artifact location.
- `reports/` - experiment reports, metrics, predictions, and analysis outputs.
- `docs/` - experiment logs and feature-mapping documentation.
- `tests/` - project tests.
- `requirements.txt` - Python dependencies specified for the project.

## Reproducibility

Before reproducing an experiment:

1. Use an appropriate Python environment.
2. Install the dependencies specified in `requirements.txt`.
3. For the Phase 3 LSTM, install a PyTorch build compatible with the Python version and platform. PyTorch is installed in the currently validated environment but is not yet specified in `requirements.txt`.
4. Obtain the required datasets and verify their locations. The DMBD raw data used in the experiments was stored externally at `D:\dmdb`; this path may need to be changed for another environment.
5. Review the relevant script and its expected inputs before running it.
6. Keep training, validation, and official test sets separate.
7. Preserve the original experiment reports and artifacts when rerunning experiments.
8. Record code changes, dependency versions, parameters, random seeds, and evaluation results.
9. Check that regenerated results are comparable to the documented experiments before reporting them.

The reports document completed experiments, but the presence of these files alone does not establish reproducibility in a clean environment.

## Project Status

The three research phases have documented experimental results and supporting artifacts. The repository is a research prototype for cybersecurity model evaluation, not a production-ready threat detection system.
