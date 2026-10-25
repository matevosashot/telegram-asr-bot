"""Download the first N clips of a FLEURS test split, e.g.:
    python scripts/fetch_fleurs.py hy_am tests/data/cache/fleurs 6
"""
import json
import os
import sys
import tarfile
import urllib.request

lang, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
base = f"https://huggingface.co/datasets/google/fleurs/resolve/main/data/{lang}"
tsv = urllib.request.urlopen(f"{base}/test.tsv").read().decode()
trans = {}
for row in tsv.splitlines():
    parts = row.split("\t")
    trans[parts[1]] = parts[2]
os.makedirs(out, exist_ok=True)
got = {}
with urllib.request.urlopen(f"{base}/audio/test.tar.gz") as r:
    with tarfile.open(fileobj=r, mode="r|gz") as tf:
        for m in tf:
            if not m.isfile():
                continue
            name = os.path.basename(m.name)
            if name not in trans:
                continue
            data = tf.extractfile(m).read()
            with open(os.path.join(out, f"{lang}_{len(got)}.wav"), "wb") as f:
                f.write(data)
            got[f"{lang}_{len(got)}.wav"] = trans[name]
            if len(got) >= n:
                break
json.dump(got, open(os.path.join(out, f"{lang}.json"), "w"), ensure_ascii=False, indent=1)
print(lang, len(got))
