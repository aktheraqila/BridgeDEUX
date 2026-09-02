import csv
print("Columns in different files:")
files = [
    'analysis/covost2_100_desktop/covost2_100_whisper_desktop/covost2_100_whisper_desktop_results.csv',
    'analysis/covost2_100_desktop/covost2_100_w1_desktop/covost2_100_w1_desktop_results.csv',
    'analysis/mslt_100_desktop/whisper/mslt_100_desktop_results.csv',
]

for fpath in files:
    with open(fpath, encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames
        has_chrf_clean_f = 'chrf_clean_f' in cols if cols else False
        print(f"\n{fpath}")
        print(f"  Column count: {len(cols) if cols else 0}")
        print(f"  Has 'chrf_clean_f': {has_chrf_clean_f}")
        if cols:
            print(f"  Columns: {', '.join(cols)}")
