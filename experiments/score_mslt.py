import re, unicodedata
import pandas as pd
from jiwer import wer, cer

MSLT = r"datasets\cache\mslt\de_en\test\mslt_de_asr_test.parquet"
FILES = {
    "Parakeet": r"results\parakeet.cpp (tdt 0.6b v3 f16)_mslt_asr_test\parakeet.cpp (tdt 0.6b v3 f16)_mslt_asr_test_results.csv",
    "W0":       r"results\whisper.cpp (base)_mslt_test\whisper.cpp (base)_mslt_test_results.csv",
    "W1_hf":    r"experiments\results\w0_w1_w2_test\kd_eval_w1\kd_eval_w1_results.csv",
    "W2_hf":    r"experiments\results\w0_w1_w2_test\kd_eval_w2\kd_eval_w2_results.csv",
    # add the W1 -> GGML -> whisper.cpp run here
}

TAG = re.compile(r"<[^>]*>")
def norm(s):
    s = unicodedata.normalize("NFKC", str(s))
    s = s.replace("\\r\\n", " ").replace("\r\n", " ")
    return " ".join(re.sub(r"[^\w\s]", " ", TAG.sub(" ", s).lower()).split())

def pad(s):
    return s.astype(str).str.strip().str.zfill(4)

base = pd.read_parquet(MSLT, columns=["id", "t1_reference", "t2_reference"])
base["id"] = pad(base.id)

rows = []
for name, path in FILES.items():
    d = pd.read_csv(path, dtype=str)
    idcol = "sample_id" if "sample_id" in d.columns else "id"
    d = d[[idcol, "hypothesis"]].rename(columns={idcol: "id"})
    d["id"] = pad(d.id)
    m = base.merge(d, on="id", how="inner")
    assert len(m) == len(d) == len(base), f"{name}: join {len(d)} -> {len(m)}"

    h = [norm(x) for x in m.hypothesis.fillna("")]
    row = {"model": name, "n": len(m),
           "empty": sum(1 for x in h if not x)}
    for tag, col in [("T2", "t2_reference"), ("T1", "t1_reference")]:
        r = [norm(x) for x in m[col].fillna("")]
        k = [i for i in range(len(r)) if r[i].strip()]
        row[f"WER_{tag}"] = round(100 * wer([r[i] for i in k], [h[i] for i in k]), 2)
        row[f"CER_{tag}"] = round(100 * cer([r[i] for i in k], [h[i] for i in k]), 2)
    rows.append(row)

out = pd.DataFrame(rows)
print(out.to_string(index=False))
out.to_csv("results/mslt_final_comparison.csv", index=False)