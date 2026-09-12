import pandas as pd, re, unicodedata, glob, os
from jiwer import wer, cer

MSLT = r"datasets\cache\mslt\de_en\test\mslt_de_asr_test.parquet"
TAG = re.compile(r"<[^>]*>")
def norm(s):
    s = unicodedata.normalize("NFKC", str(s)).replace("\\r\\n"," ").replace("\r\n"," ")
    return " ".join(re.sub(r"[^\w\s]"," ",TAG.sub(" ",s).lower()).split())
def pad(x): return x.astype(str).str.strip().str.zfill(4)

base = pd.read_parquet(MSLT, columns=["id","t1_reference","t2_reference"])
base["id"] = pad(base.id)

# list every candidate result CSV — edit this glob to match your tree
FILES = glob.glob(r"results\**\*mslt*results.csv", recursive=True) + \
        glob.glob(r"experiments\results\**\*results.csv", recursive=True)

rows = []
for p in FILES:
    d = pd.read_csv(p, dtype=str)
    idcol = "sample_id" if "sample_id" in d.columns else "id"
    if idcol not in d.columns or "hypothesis" not in d.columns:
        continue
    d = d[[idcol,"hypothesis"]].rename(columns={idcol:"id"})
    d["id"] = pad(d.id)
    m = base.merge(d, on="id", how="inner")
    h = [norm(x) for x in m.hypothesis.fillna("")]
    out = {"file": os.path.basename(p)[:45], "n": len(m)}
    for tag,col in [("T2","t2_reference"),("T1","t1_reference")]:
        r = [norm(x) for x in m[col].fillna("")]
        k = [i for i in range(len(r)) if r[i].strip()]
        out[f"WER_{tag}"] = round(100*wer([r[i] for i in k],[h[i] for i in k]),2)
    rows.append(out)

print(pd.DataFrame(rows).to_string(index=False))