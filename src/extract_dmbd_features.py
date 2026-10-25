import json
from pathlib import Path
from collections import Counter, defaultdict

import ijson
import pandas as pd

EVENT_FILE = Path(r"D:\dmdb\trees_0.json")
LABEL_DIR = Path("data/raw/process_attribution")
OUTPUT_DIR = Path("data/processed/process_attribution")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
    with open(path, "r", encoding="utf-8") as f:
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
    labels.update(load_labels(LABEL_DIR / "truth_labels_train.json", "train"))
    labels.update(load_labels(LABEL_DIR / "truth_labels_test.json", "test"))

    print(f"Loaded labels: {len(labels):,}")

    # Map the actual JSON row index directly to its experiment ID.
    # Do not rely on list position or experiment-range traversal.
    experiment_by_index = {}

    with EVENT_FILE.open("rb") as f:
        for index_text, experiment in ijson.kvitems(f, "experiment"):
            experiment_by_index[index_text] = str(experiment)

    print(f"Event rows mapped: {len(experiment_by_index):,}")
    print(f"Unique experiments: {len(set(experiment_by_index.values())):,}")

    stats = {
        experiment: {
            "total_events": 0,
            **{feature: 0 for feature in FEATURES},
        }
        for experiment in labels
    }

    # Count event types using the matching JSON index.
    # The event index is used as a string to look up the same JSON key.
    with EVENT_FILE.open("rb") as f:
        for index_text, rule in ijson.kvitems(f, "RuleName"):
            experiment = experiment_by_index.get(index_text)

            if experiment is None:
                raise RuntimeError(
                    f"Missing experiment mapping for event index {index_text}"
                )

            if experiment not in labels:
                continue

            stats[experiment]["total_events"] += 1
            stats[experiment][category(rule)] += 1

    rows = []

    for experiment, counts in stats.items():
        rows.append({
            "experiment": experiment,
            **counts,
            **labels[experiment],
        })

    df = pd.DataFrame(rows).sort_values("experiment").reset_index(drop=True)

    # Validate that category counts sum to the total event count.
    category_total = df[FEATURES].sum(axis=1)
    mismatches = df.loc[category_total != df["total_events"]]

    if not mismatches.empty:
        raise RuntimeError(
            f"{len(mismatches)} experiments have category totals that do not "
            "match total_events. CSV was not saved."
        )

    # Validate E100 against independently counted raw RuleName events.
    e100 = df.loc[df["experiment"] == "E100"]

    if len(e100) != 1:
        raise RuntimeError(f"Expected exactly one E100 row, found {len(e100)}")

    e100 = e100.iloc[0]

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

    print("\nE100 validation:")
    for feature, expected in expected_e100.items():
        actual = int(e100[feature])
        print(f"{feature}: expected={expected}, extracted={actual}")

        if actual != expected:
            raise RuntimeError(
                f"E100 validation failed for {feature}: "
                f"expected {expected}, got {actual}. CSV was not saved."
            )

    output_file = OUTPUT_DIR / "dmbd_trees0_event_count_features.csv"
    df.to_csv(output_file, index=False)

    print(f"\nFeature table saved: {output_file}")
    print(f"Labeled experiments: {len(df):,}")
    print("All category totals match total_events.")
    print("E100 validation passed.")

    print("\nLabel distribution:")
    print(df.groupby(["split", "label"]).size().to_string())


if __name__ == "__main__":
    main()
