import json, glob, os
from collections import Counter

files = glob.glob("results/**/*.jsonl*", recursive=True)
cascade = [f for f in files if "cascaded" in f.lower()]

print("=== ALL CASCADE FILES ===")
for f in sorted(cascade):
    print(f"  {f}  ({os.path.getsize(f)/1e6:.1f} MB)")

print("\n=== ANY INT8 CASCADE? ===")
int8_cascade = [f for f in cascade if "int8" in f.lower()]
print(int8_cascade if int8_cascade else "  NONE FOUND -> INT8 cascade was never run")

for f in sorted(cascade):
    print(f"\n=== {f} ===")
    rows = []
    with open(f, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try: rows.append(json.loads(line))
                except json.JSONDecodeError: pass
    print(f"  records: {len(rows)}")
    if not rows: continue
    print(f"  keys: {sorted(rows[0].keys())}")
    print(f"  sample record: {json.dumps(rows[0], ensure_ascii=False)[:600]}")
    # which fields are actually populated?
    for k in sorted(rows[0].keys()):
        n = sum(1 for r in rows if r.get(k) not in (None, "", [], {}))
        print(f"    {k}: {n}/{len(rows)} populated")