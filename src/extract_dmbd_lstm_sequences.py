import json
from pathlib import Path

import ijson
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(r"D:\dmdb")
LABEL_DIR = ROOT / "data" / "raw" / "process_attribution"
OUT_DIR = ROOT / "data" / "processed" / "process_attribution"

MAX_LEN = 512
PAD_ID = 0

FEATURES_CSV = OUT_DIR / "dmbd_all_events_experiment_features.csv"
OUTPUT_NPZ = OUT_DIR / "dmbd_lstm_sequences_512.npz"
OUTPUT_META = OUT_DIR / "dmbd_lstm_sequences_512_metadata.csv"
OUTPUT_VOCAB = OUT_DIR / "dmbd_lstm_event_vocabulary.json"


def load_label_file(path, split):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not {"experiment", "label"}.issubset(data):
        raise ValueError(f"Unexpected label-file format: {path.name}")

    experiments = data["experiment"]
    labels = data["label"]

    if set(experiments) != set(labels):
        raise ValueError(f"Experiment and label IDs differ in {path.name}")

    label_map = {"benign": 0, "malicious": 1}

    normalized_labels = {}
    for index in experiments:
        raw_label = str(labels[index]).strip().lower()
        if raw_label not in label_map:
            raise ValueError(
                f"Unknown label {raw_label!r} in {path.name}"
            )
        normalized_labels[str(experiments[index])] = {
            "label": label_map[raw_label],
            "split": split,
        }

    return normalized_labels


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not FEATURES_CSV.exists():
        raise FileNotFoundError(f"Missing feature CSV: {FEATURES_CSV}")

    feature_df = pd.read_csv(FEATURES_CSV, usecols=["experiment", "split"])
    feature_split = dict(
        zip(feature_df["experiment"].astype(str),
            feature_df["split"].astype(str))
    )

    labels = {}
    labels.update(load_label_file(
        LABEL_DIR / "truth_labels_train.json", "train"
    ))
    labels.update(load_label_file(
        LABEL_DIR / "truth_labels_test.json", "test"
    ))

    if set(labels) != set(feature_split):
        raise ValueError(
            "Label IDs and feature CSV experiment IDs do not match."
        )

    for experiment, record in labels.items():
        if feature_split[experiment] != record["split"]:
            raise ValueError(f"Split mismatch for {experiment}")

    files = sorted(DATA_DIR.glob("trees_*.json"))
    if len(files) != 8:
        raise ValueError(f"Expected 8 event files, found {len(files)}")

    # Preallocate one matrix instead of retaining a dictionary of arrays.
    experiment_ids = sorted(labels)
    row_by_experiment = {
        experiment: row for row, experiment in enumerate(experiment_ids)
    }
    X = np.zeros((len(experiment_ids), MAX_LEN), dtype=np.uint16)
    sequence_lengths = np.zeros(len(experiment_ids), dtype=np.uint32)

    event_vocabulary = {}
    seen_experiments = set()
    total_events = 0

    for file_number, path in enumerate(files, start=1):
        print(f"\\n[{file_number}/{len(files)}] Streaming {path.name}",
              flush=True)

        # First pass: count event rows without retaining a large Python dictionary.
        row_count = 0
        max_index = -1

        with path.open("rb") as f:
            for index, _ in ijson.kvitems(f, "experiment"):
                numeric_index = int(index)
                if numeric_index < 0:
                    raise ValueError(f"Negative event index: {index}")
                row_count += 1
                max_index = max(max_index, numeric_index)

        if max_index + 1 != row_count:
            raise ValueError(
                f"Non-contiguous event indices in {path.name}: "
                f"{row_count:,} rows, maximum index {max_index:,}"
            )

        # Second pass: map each event index to a compact experiment-row ID.
        event_to_row = np.full(row_count, np.uint32(2**32 - 1), dtype=np.uint32)

        with path.open("rb") as f:
            for index, experiment in ijson.kvitems(f, "experiment"):
                experiment = str(experiment)
                row = row_by_experiment.get(experiment)

                if row is None:
                    raise ValueError(
                        f"Unlabeled experiment encountered: {experiment}"
                    )

                event_to_row[int(index)] = row

        print(f"  Rows indexed: {row_count:,}", flush=True)
        current_experiment = None
        current_position = 0
        file_events = 0

        with path.open("rb") as f:
            for index, event_type in ijson.kvitems(f, "RuleName"):
                numeric_index = int(index)
                if numeric_index < 0 or numeric_index >= len(event_to_row):
                    raise ValueError(f"Invalid RuleName row index: {index}")

                row = int(event_to_row[numeric_index])
                if row == 2**32 - 1:
                    raise ValueError(f"Missing experiment mapping for RuleName row {index}")

                experiment = experiment_ids[row]

                if experiment != current_experiment:
                    if experiment in seen_experiments:
                        raise ValueError(
                            f"Experiment repeated or non-contiguous: {experiment}"
                        )
                    seen_experiments.add(experiment)
                    current_experiment = experiment
                    current_position = 0

                event_type = str(event_type)
                if event_type not in event_vocabulary:
                    event_vocabulary[event_type] = len(event_vocabulary) + 1


                sequence_lengths[row] += 1
                if current_position < MAX_LEN:
                    event_id = event_vocabulary[event_type]
                    if event_id >= np.iinfo(np.uint16).max:
                        raise ValueError("Event vocabulary exceeds uint16 capacity.")
                    X[row, current_position] = event_id
                    current_position += 1

                total_events += 1
                file_events += 1

        del event_to_row
        print(f"  Events streamed: {file_events:,}", flush=True)
        print(f"  Experiments encountered: {len(seen_experiments):,}", flush=True)

    if seen_experiments != set(labels):
        missing_events = sorted(set(labels) - seen_experiments)[:10]
        extra_events = sorted(seen_experiments - set(labels))[:10]
        raise ValueError(
            f"Coverage mismatch. Missing experiments: {missing_events}; "
            f"unexpected experiments: {extra_events}"
        )

    # Preserve raw telemetry counts and validate them against the audited CSV.
    raw_event_lengths = sequence_lengths.copy()

    if len(event_vocabulary) >= np.iinfo(np.uint16).max:
        raise ValueError("Too many event types for uint16 encoding.")

    print(f"\\nTotal event rows: {total_events:,}", flush=True)
    print(f"Experiments: {len(experiment_ids):,}", flush=True)
    print(f"Event types: {len(event_vocabulary)}", flush=True)

    with OUTPUT_VOCAB.open("w", encoding="utf-8") as f:
        json.dump(
            {"PAD": PAD_ID, **event_vocabulary},
            f, indent=2, ensure_ascii=False
        )

    y = np.asarray([labels[e]["label"] for e in experiment_ids],
                  dtype=np.uint8)
    split = np.asarray([labels[e]["split"] for e in experiment_ids],
                       dtype="U5")
    # Use the already audited event totals to recover each original sequence length.
    length_df = pd.read_csv(
        FEATURES_CSV, usecols=["experiment", "total_events"]
    )
    length_map = dict(zip(
        length_df["experiment"].astype(str),
        length_df["total_events"].astype(int)
    ))
    if set(length_map) != set(experiment_ids):
        raise ValueError("Feature CSV experiment IDs do not match sequence IDs.")

    csv_lengths = np.asarray(
        [length_map[e] for e in experiment_ids], dtype=np.uint32
    )

    if not np.array_equal(raw_event_lengths, csv_lengths):
        mismatch_rows = np.flatnonzero(raw_event_lengths != csv_lengths)[:10]
        examples = [
            {
                "experiment": experiment_ids[i],
                "raw_events": int(raw_event_lengths[i]),
                "csv_events": int(csv_lengths[i]),
            }
            for i in mismatch_rows
        ]
        raise ValueError(f"Raw event counts disagree with feature CSV: {examples}")

    original_lengths = raw_event_lengths

    if len(event_vocabulary) >= np.iinfo(np.uint16).max:
        raise ValueError("Too many event types for uint16 encoding.")

    np.savez_compressed(
        OUTPUT_NPZ,
        X=X,
        y=y,
        experiment_ids=np.asarray(experiment_ids, dtype=str),
        split=split,
        original_lengths=original_lengths,
        max_len=np.asarray([MAX_LEN], dtype=np.uint16),
    )

    metadata = pd.DataFrame({
        "experiment": experiment_ids,
        "label": y,
        "split": split,
        "original_length": original_lengths,
        "retained_length": np.minimum(original_lengths, MAX_LEN),
        "truncated": original_lengths > MAX_LEN,
    })
    metadata.to_csv(OUTPUT_META, index=False)

    print("\nExtraction complete.", flush=True)
    print(f"Total event rows: {total_events:,}")
    print(f"Experiments: {len(experiment_ids):,}")
    print(f"Event types: {len(event_vocabulary)}")
    print(f"Sequence array shape: {X.shape}")
    print(f"Training experiments: {sum(split == 'train'):,}")
    print(f"Test experiments: {sum(split == 'test'):,}")
    print(f"Truncated experiments: {sum(original_lengths > MAX_LEN):,}")
    print(f"Padding ID: {PAD_ID}")
    print(f"Sequences: {OUTPUT_NPZ}")
    print(f"Metadata: {OUTPUT_META}")
    print(f"Vocabulary: {OUTPUT_VOCAB}")
    print("Labels are experiment-level, not process-level.")


if __name__ == "__main__":
    main()
