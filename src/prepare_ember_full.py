import json
from pathlib import Path

import numpy as np

DATA_DIR = Path("data/raw/malware/ember_dataset_2018_2.tar.bz2")
OUTPUT_DIR = Path("data/processed/malware/full")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_NAMES_FILE = Path("data/processed/malware/ember_sample.npz")
BATCH_SIZE = 5000


def extract_features(record):
    features = {}

    for field in ("histogram", "byteentropy"):
        values = record.get(field, [])
        for index, value in enumerate(values):
            features[f"{field}_{index}"] = (
                float(value)
                if isinstance(value, (int, float))
                and not isinstance(value, bool)
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


def main():
    with np.load(FEATURE_NAMES_FILE, allow_pickle=False) as data:
        feature_names = data["feature_names"].tolist()

    X_path = OUTPUT_DIR / "X.npy"
    y_path = OUTPUT_DIR / "y.npy"

    # Preallocate disk-backed arrays for up to 600,000 labeled records.
    max_records = 600_000
    X = np.lib.format.open_memmap(
        X_path, mode="w+", dtype=np.float32,
        shape=(max_records, len(feature_names))
    )
    y = np.lib.format.open_memmap(
        y_path, mode="w+", dtype=np.int8,
        shape=(max_records,)
    )

    count = 0
    benign = 0
    malicious = 0
    batch_X = []
    batch_y = []

    for path in sorted(DATA_DIR.glob("train_features_*.jsonl")):
        print(f"Reading {path.name}...", flush=True)

        with path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue

                record = json.loads(line)
                label = record.get("label", -1)

                if label not in (0, 1):
                    continue

                features = extract_features(record)
                row = [features.get(name, 0.0) for name in feature_names]

                batch_X.append(row)
                batch_y.append(label)

                if label == 0:
                    benign += 1
                else:
                    malicious += 1

                if len(batch_y) >= BATCH_SIZE:
                    end = count + len(batch_y)
                    X[count:end] = np.asarray(batch_X, dtype=np.float32)
                    y[count:end] = np.asarray(batch_y, dtype=np.int8)
                    count = end
                    batch_X.clear()
                    batch_y.clear()
                    print(f"  Processed {count:,} labeled records", flush=True)

    if batch_y:
        end = count + len(batch_y)
        X[count:end] = np.asarray(batch_X, dtype=np.float32)
        y[count:end] = np.asarray(batch_y, dtype=np.int8)
        count = end

    X.flush()
    y.flush()

    # Trim disk-backed arrays to the actual number of labeled records.
    del X, y
    X_full = np.load(X_path, mmap_mode="r")
    y_full = np.load(y_path, mmap_mode="r")

    if count < max_records:
        X_trim = np.lib.format.open_memmap(
            OUTPUT_DIR / "X_trim.npy", mode="w+", dtype=np.float32,
            shape=(count, len(feature_names))
        )
        y_trim = np.lib.format.open_memmap(
            OUTPUT_DIR / "y_trim.npy", mode="w+", dtype=np.int8,
            shape=(count,)
        )

        for start in range(0, count, BATCH_SIZE):
            end = min(start + BATCH_SIZE, count)
            X_trim[start:end] = X_full[start:end]
            y_trim[start:end] = y_full[start:end]

        X_trim.flush()
        y_trim.flush()
        del X_trim, y_trim, X_full, y_full

        X_path.unlink()
        y_path.unlink()
        (OUTPUT_DIR / "X_trim.npy").replace(X_path)
        (OUTPUT_DIR / "y_trim.npy").replace(y_path)

    print("\nFull labeled dataset preparation complete.")
    print(f"Records: {count:,}")
    print(f"Benign: {benign:,}")
    print(f"Malicious: {malicious:,}")
    print(f"Features: {len(feature_names):,}")
    print(f"Feature matrix: {X_path}")
    print(f"Labels: {y_path}")


if __name__ == "__main__":
    main()
