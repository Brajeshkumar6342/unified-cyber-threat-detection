import json
from pathlib import Path

import numpy as np

DATA_DIR = Path("data/raw/malware/ember_dataset_2018_2.tar.bz2")
OUTPUT_DIR = Path("data/processed/malware")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "ember_sample.npz"
SAMPLES_PER_CLASS = 10000


def extract_features(record):
    features = {}

    for field in ("histogram", "byteentropy"):
        values = record.get(field, [])
        for index, value in enumerate(values):
            features[f"{field}_{index}"] = (
                float(value)
                if isinstance(value, (int, float)) and not isinstance(value, bool)
                else 0.0
            )

    for field in ("general", "strings"):
        values = record.get(field, {})
        for key in sorted(values):
            value = values[key]

            if isinstance(value, bool):
                features[f"{field}_{key}"] = float(value)
            elif isinstance(value, (int, float)):
                features[f"{field}_{key}"] = float(value)
            else:
                features[f"{field}_{key}"] = 0.0

    return features


samples = {0: [], 1: []}
feature_names = None

for path in sorted(DATA_DIR.glob("train_features_*.jsonl")):
    if path.name == "train_features_0.jsonl":
        continue

    print(f"Reading {path.name}...", flush=True)

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            record = json.loads(line)
            label = record.get("label", -1)

            if label not in (0, 1):
                continue

            if len(samples[label]) >= SAMPLES_PER_CLASS:
                continue

            features = extract_features(record)
            current_names = sorted(features)

            if feature_names is None:
                feature_names = current_names
            elif current_names != feature_names:
                raise ValueError(
                    f"Inconsistent feature structure in {path.name}"
                )

            samples[label].append(
                [features[name] for name in feature_names]
            )

    if all(len(samples[label]) >= SAMPLES_PER_CLASS for label in (0, 1)):
        break

if any(len(samples[label]) < SAMPLES_PER_CLASS for label in (0, 1)):
    raise RuntimeError("Not enough samples were collected for both classes.")

X = np.asarray(samples[0] + samples[1], dtype=np.float32)
y = np.asarray(
    [0] * len(samples[0]) + [1] * len(samples[1]),
    dtype=np.int8,
)

X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

np.savez_compressed(
    OUTPUT_FILE,
    X=X,
    y=y,
    feature_names=np.asarray(feature_names, dtype=str),
)

print("\nBalanced sample created.")
print(f"Samples: {len(y):,}")
print(f"Features: {X.shape[1]:,}")
print(f"Benign: {int(np.sum(y == 0)):,}")
print(f"Malicious: {int(np.sum(y == 1)):,}")
print(f"Feature names saved: {len(feature_names):,}")
print(f"Output: {OUTPUT_FILE}")
