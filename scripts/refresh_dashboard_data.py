"""Refresh descriptive chart values from the committed CSV; does not train a model."""
from pathlib import Path
import csv
import json
import collections

root = Path(__file__).resolve().parents[1]
with (root / 'train.csv').open(newline="") as stream:
    rows = list(csv.DictReader(stream))
target = root / "docs" / "data.json"
data = json.loads(target.read_text())
charts = data["charts"]
missing = sorted(
    ((key, sum(row[key].strip() in ("", "?", "NA", "nan") for row in rows))
     for key in rows[0]), key=lambda item: -item[1]
)[:10]
positive = sum(int(row["class"]) == 1 for row in rows)
charts[0]["series"][0]["values"] = [len(rows) - positive, positive]
charts[1]["labels"] = [key for key, value in missing]
charts[1]["series"][0]["values"] = [value for key, value in missing]
# Fold scores are transcribed from stored notebook output, not regenerated here.
target.write_text(json.dumps(data, indent=2) + "\n")
print(f"Refreshed descriptive charts from {len(rows):,} input rows; review captions if the source changed.")
