from pathlib import Path
import re

base = Path('d:/BridgeDEUX/analysis/Mobile Benchmark Results')
print("Mobile Benchmark Result Files:")
for csv_file in sorted(base.rglob('*.csv')):
    rel_path = csv_file.relative_to('d:/BridgeDEUX').as_posix()
    print(f"  {csv_file.name}")
    print(f"    Path: {rel_path}")
