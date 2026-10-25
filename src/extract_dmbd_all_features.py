import json
from pathlib import Path
from collections import Counter

import ijson
import pandas as pd

DATA_DIR = Path(r"D:\dmdb")
LABEL_DIR = Path("data/raw/process_attribution")
OUTPUT_DIR = Path("data/processed/process_attribution")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EVENT_FILES = [DATA_DIR / f"trees_{i}.json" for i in range(8)]

FEATURES = [
    "process_create",
    "process_term",
    "image_load",
    "file_events",
    "network_events",
    "registry_events",
    "other_events",
]


def load_labels(path, split):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    result = {}

    for idx, experiment in data["experiment"].items():
        raw = str(data["label"][idx]).strip().lower()

        if raw in {"benign", "clean", "0"}:
            label = 0
        elif raw in {"malicious", "malware", "1"}:
            label = 1
        else:
            continue

        result[str(experiment)] = {
            "label": label,
            "year": data.get("year", {}).get(idx),
            "split": split,
        }

    return result


def category(rule):
    rule = str(rule).lower()

    if rule == "process_create":
        return "process_create"
    if rule == "process_term":
        return "process_term"
    if rule == "image_load":
        return "image_load"
    if rule.startswith("file_"):
        return "file_events"
    if rule.startswith("net_"):
        return "network_events"
    if rule.startswith("reg_"):
        return "registry_events"

    return "other_events"


def main():
    labels = {}
    labels.update(
        load_labels(LABEL_DIR / "truth_labels_train.json", "train")
    )
    labels.update(
        load_labels(LABEL_DIR / "truth_labels_test.json", "test")
    )

    if len(labels) != 64747:
        raise RuntimeError(
            f"Expected 64,747 labels, found {len(labels):,}"
        )

    print(f"Loaded labels: {len(labels):,}")

    # One statistics record per experiment.
    stats = {
        experiment: {
            "total_events": 0,
            **{feature: 0 for feature in FEATURES},
        }
        for experiment in labels
    }

    seen_experiments = set()
    total_events_all_files = 0

    for event_file in EVENT_FILES:
        if not event_file.exists():
            raise FileNotFoundError(event_file)

        print(f"\nProcessing {event_file.name}...")

        experiments_in_file = set()
        file_event_count = 0

        # Build the index-to-experiment mapping for this file.
        experiment_by_index = {}

        with event_file.open("rb") as f:
            for index_text, experiment in ijson.kvitems(
                f, "experiment"
            ):
                experiment_by_index[index_text] = str(experiment)

        # Each experiment must occur in only one event file.
        experiments_in_file.update(experiment_by_index.values())

        duplicates = seen_experiments.intersection(experiments_in_file)
        if duplicates:
            raise RuntimeError(
                f"Experiments repeated across files: "
                f"{sorted(duplicates)[:10]}"
            )

        unknown = experiments_in_file.difference(labels)
        if unknown:
            raise RuntimeError(
                f"Found experiments without labels: "
                f"{sorted(unknown)[:10]}"
            )

        # Aggregate event types by direct JSON index mapping.
        with event_file.open("rb") as f:
            for index_text, rule in ijson.kvitems(f, "RuleName"):
                experiment = experiment_by_index.get(index_text)

                if experiment is None:
                    raise RuntimeError(
                        f"Missing experiment mapping for index "
                        f"{index_text} in {event_file.name}"
                    )

                stats[experiment]["total_events"] += 1
                stats[experiment][category(rule)] += 1

                file_event_count += 1

                if file_event_count % 1_000_000 == 0:
                    print(
                        f"  Processed {file_event_count:,} events..."
                    )

        seen_experiments.update(experiments_in_file)
        total_events_all_files += file_event_count

        print(f"  Event rows: {file_event_count:,}")
        print(f"  Unique experiments: {len(experiments_in_file):,}")

    # Ensure every labeled experiment was found in the event data.
    missing_experiments = set(labels).difference(seen_experiments)

    if missing_experiments:
        raise RuntimeError(
            f"{len(missing_experiments)} labeled experiments "
            f"were not found in the event files."
        )

    rows = []

    for experiment, counts in stats.items():
        rows.append({
            "experiment": experiment,
            **counts,
            **labels[experiment],
        })

    df = pd.DataFrame(rows).sort_values("experiment").reset_index(drop=True)

    # Validate category sums before writing the combined CSV.
    category_total = df[FEATURES].sum(axis=1)
    mismatches = df.loc[
        category_total != df["total_events"]
    ]

    if not mismatches.empty:
        raise RuntimeError(
            f"{len(mismatches)} experiments have category totals "
            f"that do not match total_events. Output was not saved."
        )

    if (df["total_events"] <= 0).any():
        raise RuntimeError(
            "Some experiments have zero events in the combined dataset."
        )

    # Independent expected values for E100.
    e100 = df.loc[df["experiment"] == "E100"]

    if len(e100) != 1:
        raise RuntimeError("Expected exactly one E100 row.")

    expected_e100 = {
        "total_events": 926,
        "process_create": 4,
        "process_term": 5,
        "image_load": 354,
        "file_events": 194,
        "network_events": 9,
        "registry_events": 359,
        "other_events": 1,
    }

    for feature, expected in expected_e100.items():
        actual = int(e100.iloc[0][feature])
        if actual != expected:
            raise RuntimeError(
                f"E100 validation failed for {feature}: "
                f"expected {expected}, got {actual}"
            )

    output_file = (
        OUTPUT_DIR / "dmbd_all_events_experiment_features.csv"
    )
    df.to_csv(output_file, index=False)

    print("\n" + "=" * 60)
    print("COMBINED EXTRACTION SUCCESSFUL")
    print("=" * 60)
    print(f"Output: {output_file}")
    print(f"Experiments: {len(df):,}")
    print(f"Total event rows: {total_events_all_files:,}")
    print(f"Unique experiments: {len(seen_experiments):,}")
    print("Zero-event experiments: 0")
    print("Duplicate experiments across files: 0")
    print("Category totals: validated")
    print("E100 reference counts: validated")

    print("\nLabel distribution:")
    print(df.groupby(["split", "label"]).size().to_string())

    print("\nEvent-count summary:")
    print(df["total_events"].describe().round(2).to_string())


if __name__ == "__main__":
    main()
