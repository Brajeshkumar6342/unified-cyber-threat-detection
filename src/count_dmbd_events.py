import ijson
from pathlib import Path
import time

folder = Path(r"D:\dmdb")
files = sorted(folder.glob("trees_*.json"))

print(f"Found {len(files)} event files.", flush=True)
grand_total = 0

for path in files:
    start = time.time()
    count = 0
    print(f"\nCounting {path.name}...", flush=True)

    with path.open("rb") as f:
        for prefix, event, value in ijson.parse(f):
            if prefix == "experiment" and event == "map_key":
                count += 1

    grand_total += count
    elapsed = time.time() - start
    print(f"Events: {count:,}", flush=True)
    print(f"Time: {elapsed / 60:.1f} minutes", flush=True)

print(f"\nTOTAL EVENTS: {grand_total:,}", flush=True)
