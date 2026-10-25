import ijson

path = r"D:\dmdb\trees_0.json"
fields = ["experiment", "timestamp", "RuleName", "parent_name",
          "child_name", "parent_guid", "child_guid"]
samples = {field: [] for field in fields}

with open(path, "rb") as f:
    for prefix, event, value in ijson.parse(f):
        if "." not in prefix:
            continue

        field, index = prefix.rsplit(".", 1)

        if field in samples and event in ("string", "number", "null", "boolean"):
            if len(samples[field]) < 5:
                samples[field].append((index, value))

        if all(len(values) >= 5 for values in samples.values()):
            break

print("Sample values by field:")
for field, values in samples.items():
    print(f"\n{field}:")
    for index, value in values:
        print(f"  [{index}] = {value}")
