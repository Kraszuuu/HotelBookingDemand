import csv
import random
import sys
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: python sample_csv.py <input.csv> [output.csv]", file=sys.stderr)
    sys.exit(1)

input_path = Path(sys.argv[1])
output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else input_path.stem + "_sample.csv"

with open(input_path, newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)
    rows = list(reader)

sample = random.sample(rows, max(1, len(rows) // 100))

with open(output_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(sample)

print(f"Sampled {len(sample)} rows out of {len(rows)} -> {output_path}")
