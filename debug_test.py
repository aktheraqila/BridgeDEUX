#!/usr/bin/env python3
from pathlib import Path
import csv
import statistics

ROOT = Path(__file__).resolve().parent

def read_csv(filename: str) -> list[dict[str, str]]:
    path = ROOT / filename
    print(f"Looking for: {path}")
    print(f"Exists: {path.exists()}")
    
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")
    
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

# Test the first file
filename = "analysis/covost2_100_desktop/covost2_100_whisper_desktop/covost2_100_whisper_desktop_results.csv"
try:
    rows = read_csv(filename)
    print(f"\nSuccessfully read {len(rows)} rows")
    print(f"Columns: {rows[0].keys()}")
    print(f"\nFirst row chrf_clean_f value: {rows[0].get('chrf_clean_f')}")
    print(f"Type: {type(rows[0].get('chrf_clean_f'))}")
    
    # Try to compute mean
    vals = [float(r['chrf_clean_f']) for r in rows if r.get('chrf_clean_f') not in (None, "")]
    print(f"Successfully extracted {len(vals)} values")
    mean_val = statistics.fmean(vals)
    print(f"Mean: {mean_val}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
