# Cross-Dataset Feature Mapping: UNSW-NB15 and CICIDS2017

## 1. Objective

Identify semantically compatible network-flow features before evaluating cross-dataset generalization.

## 2. Candidate Feature Mapping

| UNSW-NB15 | CICIDS2017 | Status | Notes |
|---|---|---|---|
| `dur` | `Flow Duration` | Candidate; unit conversion required | CICIDS2017 duration is in microseconds. UNSW-NB15 duration units must be verified from its feature-generation documentation before conversion. |
| `spkts` | `Total Fwd Packets` | Candidate | Both represent directional packet counts, but flow direction conventions may differ. |
| `dpkts` | `Total Backward Packets` | Candidate | Both represent the opposite-direction packet count, subject to flow construction conventions. |
| `sbytes` | `Total Length of Fwd Packets` | Candidate | Verify byte-count definitions and direction conventions. |
| `dbytes` | `Total Length of Bwd Packets` | Candidate | Verify byte-count definitions and direction conventions. |
| `sload` | No mapping selected | Unresolved | Requires verification of rate units and calculation. |
| `rate` | No mapping selected | Unresolved | Requires verification of the exact rate definition. |
| `sinpkt` | No mapping selected | Unresolved | Requires verification of inter-packet timing definitions and units. |
| `smean` | No mapping selected | Unresolved | Requires verification of packet-size averaging definitions. |

## 3. Evidence and Limitations

- UNSW-NB15 feature documentation is available from the [official UNSW-NB15 dataset page](https://research.unsw.edu.au/projects/unsw-nb15-dataset).
- CICFlowMeter documentation describes forward and backward directions relative to the first packet and specifies that flow duration is measured in microseconds.
- Similar feature names or meanings do not establish exact equivalence.
- Differences in flow construction, direction, measurement units, and aggregation can affect transfer performance.
- No mapping should be considered final until its definitions and units have been verified.

### Source CSV Inspection

The UNSW-NB15 training CSV was inspected to confirm that the candidate source columns `dur`, `spkts`, `dpkts`, `sbytes`, and `dbytes` are present. A three-row sample showed decimal values in `dur` and integer values in the packet and byte-count columns.

This inspection confirms column availability and illustrates the stored values, but it does not independently establish the official measurement units, flow-direction conventions, or byte-counting methodology. The five feature mappings therefore remain candidates pending documentation verification.

## 4. Planned Experimental Protocol

1. Verify the candidate feature definitions and units.
2. Finalize a common feature representation and document every conversion.
3. Split UNSW-NB15 training data into training and validation partitions.
4. Fit all preprocessing and the source model using only the source training partition.
5. Freeze the preprocessing and model before cross-dataset evaluation.
6. Evaluate on CICIDS2017 without using its labels to tune the transferred model.
7. Report precision, recall, F1-score, false-positive rate, and ROC-AUC where appropriate.
8. Train a separate CICIDS2017 reference model for context; report it separately from the cross-dataset transfer result.

The official UNSW-NB15 test set remains reserved for final evaluation and must not be used for model selection or tuning.

## 5. Duration Data-Quality Decision

Inspection of the CICIDS2017 Friday DDoS file found two records with `Flow Duration = -1`, both labelled BENIGN. Both records contain one forward packet, one backward packet, and six bytes in each direction.

For the cross-dataset experiment:

- Convert `Flow Duration` from microseconds to seconds.
- Treat negative duration values as invalid and convert them to missing values in derived experiment data.
- Preserve zero-duration records unless additional evidence indicates that they are invalid.
- Fit imputation parameters using source training data only.
- Keep the raw CICIDS2017 CSV files unchanged.
- Record invalid-duration counts and preprocessing decisions in the experiment report.

The two observed invalid-duration records are not sufficient evidence to establish why the original values occurred. Their cause remains unresolved.

## 6. Packet and Byte Feature Distribution Check

The UNSW-NB15 training dataset and CICIDS2017 Friday DDoS subset were compared using descriptive statistics.

| UNSW-NB15 feature | CICIDS2017 feature | UNSW median | CICIDS median | Decision |
|---|---|---:|---:|---|
| `spkts` | `Total Fwd Packets` | 2 | 3 | Retain as candidate |
| `dpkts` | `Total Backward Packets` | 2 | 4 | Retain as candidate |
| `sbytes` | `Total Length of Fwd Packets` | 430 | 30 | Retain as candidate; investigate distribution difference |
| `dbytes` | `Total Length of Bwd Packets` | 164 | 164 | Retain as candidate; matching medians do not prove equivalence |

The compared distributions differ in their means, upper percentiles, and maximum values. These differences may reflect traffic composition, collection conditions, or differences in feature-generation methodology.

The comparison is descriptive only. The rows belong to separate datasets and cannot be matched one-to-one.

No feature pair is considered fully validated based on distribution statistics alone. The four pairs remain candidates pending further definition and methodology checks.

## 7. Initial Cross-Dataset Feature Set

The first transfer experiment will use five candidate features:

| Common feature | UNSW-NB15 source column | CICIDS2017 target column | Transformation |
|---|---|---|---|
| `duration_seconds` | `dur` | `Flow Duration` | Convert CICIDS2017 microseconds to seconds; mark negative values missing |
| `forward_packets` | `spkts` | `Total Fwd Packets` | No unit conversion |
| `backward_packets` | `dpkts` | `Total Backward Packets` | No unit conversion |
| `forward_bytes` | `sbytes` | `Total Length of Fwd Packets` | No unit conversion |
| `backward_bytes` | `dbytes` | `Total Length of Bwd Packets` | No unit conversion |

These are candidate semantic mappings, not confirmed equivalences. Direction conventions, byte-counting definitions, and distribution differences remain limitations.

The experiment will use binary labels: benign traffic versus attack traffic. Source preprocessing and model fitting will use only the UNSW-NB15 training partition. CICIDS2017 labels will be used only for final target evaluation and not for model selection or threshold tuning.

The initial result will be interpreted as a limited cross-dataset baseline using five features, not as a complete or universally generalizable intrusion detection solution.
