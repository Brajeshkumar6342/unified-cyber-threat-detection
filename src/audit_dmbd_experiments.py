import ijson
import json
from pathlib import Path
import time

data_dir = Path(r"D:\dmdb")

with open(data_dir / "truth_labels_train.json", encoding="utf-8") as f:
    train = json.load(f)

with open(data_dir / "truth_labels_test.json", encoding="utf-8") as f:
    test = json.load(f)

train_ids = set(train["experiment"].values())
test_ids = set(test["experiment"].values())
labelled_ids = train_ids | test_ids

all_seen = set()
files = sorted(data_dir.glob("trees_*.json"))

for path in files:
    start = time.time()
    file_seen = set()

    print(f"\nScanning {path.name}...", flush=True)

    with path.open("rb") as f:
        for prefix, event, value in ijson.parse(f):
            if prefix.startswith("experiment.") and event == "string":
                file_seen.add(value)

    all_seen.update(file_seen)

    print(f"Distinct experiments in file: {len(file_seen):,}", flush=True)
    print(f"New experiments added: {len(file_seen - (all_seen - file_seen)):,}", flush=True)
    print(f"Elapsed time: {(time.time() - start) / 60:.2f} minutes", flush=True)

print("\n========== FULL DATASET AUDIT ==========")
print("Event files:", len(files))
print("Distinct experiments in event data:", len(all_seen))
print("Matching training labels:", len(all_seen & train_ids))
print("Matching test labels:", len(all_seen & test_ids))
print("Event experiments without labels:", len(all_seen - labelled_ids))
print("Training labels without events:", len(train_ids - all_seen))
print("Test labels without events:", len(test_ids - all_seen))
print("Total labeled experiments:", len(labelled_ids))
