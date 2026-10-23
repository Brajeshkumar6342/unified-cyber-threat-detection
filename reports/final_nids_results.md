# Final NIDS Experimental Results

## 1. Research objective

Evaluate a Random Forest-based Network Intrusion Detection System (NIDS) using UNSW-NB15 and CICIDS2017, with particular emphasis on cross-dataset generalization.

The investigation compares within-dataset performance with cross-dataset transfer performance and studies whether reducing the feature set improves transfer.

## 2. Experimental configuration

**Model:** Random Forest classifier.

**Preprocessing:** Median imputation.

**Random Forest parameters:**
- Number of trees: 200
- Maximum features: sqrt
- Minimum samples per leaf: 2
- Random state: 42
- Parallel jobs: -1

**Datasets:**
- UNSW-NB15 training set: 175,341 records.
- Official UNSW-NB15 test set: 82,332 records.
- CICIDS2017 Friday afternoon DDoS subset: 225,745 records.

**E4 features:**
1. `duration_seconds` — UNSW `dur`; CICIDS2017 `Flow Duration` converted to seconds.
2. `forward_packets` — UNSW `spkts`; CICIDS2017 `Total Fwd Packets`.
3. `backward_packets` — UNSW `dpkts`; CICIDS2017 `Total Backward Packets`.
4. `backward_bytes` — UNSW `dbytes`; CICIDS2017 `Total Length of Bwd Packets`.

These mappings are candidate correspondences. Their semantic equivalence across datasets has not been fully established.

## 3. Official UNSW-NB15 test evaluation

E4 was trained on the complete UNSW-NB15 training file and evaluated on the official UNSW-NB15 test file.

| Metric | Result |
|---|---:|
| Accuracy | 81.68% |
| Precision | 76.19% |
| Recall | 97.04% |
| F1-score | 85.36% |
| ROC-AUC | 87.19% |
| True negatives | 23,254 |
| False positives | 13,746 |
| False negatives | 1,341 |
| True positives | 43,991 |
| False-positive rate | 37.15% |
| False-negative rate | 2.96% |

### Interpretation

The model achieved high attack recall, detecting 97.04% of attack records. However, its false-positive rate was 37.15%, indicating that many benign records were incorrectly classified as attacks. This limits its operational suitability without further investigation.

## 4. Cross-dataset feature ablation

The original five-feature configuration, E0, was compared with E4, which removes `forward_bytes`.

| Configuration | Mean transfer F1 | Mean ROC-AUC | Mean FPR | Mean FNR |
|---|---:|---:|---:|---:|
| E0: Five features | 0.5924 | 0.5567 | 0.6697 | 0.3641 |
| E4: Four features, excluding forward bytes | 0.6987 | 0.6373 | 0.4669 | 0.2716 |

Results are means across five source split seeds.

E4 improved mean transfer F1 by approximately 10.63 percentage points compared with E0. It also improved mean ROC-AUC and reduced both mean false-positive and false-negative rates.

E4 performed better than E0 in transfer F1 across all five tested splits. However, the runs share the same CICIDS2017 target dataset and are not independent external evaluations.

## 5. Log-transform experiment

| Configuration | Source-validation F1 | CICIDS2017 transfer F1 | Transfer ROC-AUC |
|---|---:|---:|---:|
| E4: Raw features | 0.9491 | 0.6846 | 0.6392 |
| E6: Log-transformed features | 0.9489 | 0.6383 | 0.6052 |

The log-transformed configuration did not improve transfer performance in the tested split. Its transfer F1 decreased by approximately 4.63 percentage points.

## 6. E7 feature-reduction experiment

E7 retained duration and packet counts while removing both byte-count features.

| Configuration | Mean transfer F1 | Mean ROC-AUC | Mean FPR | Mean FNR |
|---|---:|---:|---:|---:|
| E4: Four features | 0.6987 | 0.6373 | 0.4669 | 0.2716 |
| E7: Three features | 0.3689 | 0.3781 | 0.5100 | 0.6857 |

E7 performed substantially worse than E4. It is not the preferred configuration under the current experimental setup.

## 7. Within-CICIDS2017 results

The initial within-dataset five-feature experiment reported:

| Metric | Result |
|---|---:|
| Accuracy | 99.95% |
| Precision | 99.98% |
| Recall | 99.93% |
| F1-score | 99.96% |
| ROC-AUC | 1.0000 |

After removing exact duplicate rows, performance remained approximately unchanged. Removing `forward_bytes` also retained very high within-dataset performance.

These results should be interpreted cautiously. Random splitting does not establish performance on future traffic or independent network captures. Removing exact duplicates alone does not rule out near-duplicates, related flows, or other forms of record dependence.

## 8. Main findings

1. Strong within-dataset performance did not translate into equally strong cross-dataset performance.
2. Removing `forward_bytes` improved transfer metrics in the tested CICIDS2017 experiments.
3. The four-feature E4 configuration was the strongest cross-dataset candidate among the configurations evaluated here.
4. Applying `log1p` did not improve transfer performance in the tested experiment.
5. Removing both byte-count features substantially degraded transfer performance.
6. The official UNSW-NB15 test evaluation showed high attack recall but a substantial false-positive rate.

## 9. Limitations

1. The cross-dataset evaluation focuses on the CICIDS2017 Friday afternoon DDoS subset rather than all attack categories.
2. CICIDS2017 informed feature selection and experimentation; it is not an untouched external test set.
3. The five source split seeds share the same target dataset and do not constitute five independent target evaluations.
4. Feature mappings across datasets are candidate correspondences, not proven semantic equivalences.
5. Randomly split within-dataset results do not establish generalization to future traffic, unseen networks, or independent captures.
6. Removing exact duplicates does not rule out all possible data dependence or leakage.
7. The official UNSW-NB15 test results evaluate source-domain performance, not cross-dataset transfer.
8. E4's transfer performance and false-positive rate remain insufficient to claim operational readiness.
9. The causes of dataset shift have not been isolated experimentally.

## 10. Conclusion

The experiments identify E4, which excludes forward-byte volume while retaining duration, forward packet count, backward packet count, and backward-byte volume, as the strongest cross-dataset candidate tested so far.

Its mean CICIDS2017 transfer F1-score was 0.6987 across five source split seeds, compared with 0.5924 for the original five-feature configuration. However, its mean transfer ROC-AUC was 0.6373 and its mean false-positive rate was 0.4669, indicating substantial remaining limitations.

The official UNSW-NB15 test evaluation produced an F1-score of 0.8536 and ROC-AUC of 0.8719, with a false-positive rate of 0.3715.

Overall, the findings support further research into cross-dataset feature compatibility and generalization. They do not establish that the proposed configuration is ready for deployment or that a single identified mechanism explains the observed dataset shift.

## 11. Reproducibility artifacts

Relevant result tables:
- `reports/tables/cross_dataset_feature_ablation.csv`
- `reports/tables/cross_dataset_ablation_stability.csv`
- `reports/tables/cross_dataset_log_transform.csv`
- `reports/tables/cross_dataset_e7_comparison.csv`
- `reports/tables/e4_unsw_official_test_metrics.csv`

Relevant experiment log:
- `docs/nids_experiment_log.md`

Original cross-dataset baseline:
- `src/nids/experiment_cross_dataset.py`

Official UNSW-NB15 evaluation script:
- `src/nids/evaluate_e4_unsw_official_test.py`
