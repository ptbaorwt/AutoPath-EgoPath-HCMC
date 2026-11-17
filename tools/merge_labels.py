"""Gộp nhãn của các thành viên (hcmc/labels_*.json) thành hcmc/labels.json, bỏ khung trùng và báo số lượng.

  python tools/merge_labels.py --hcmc hcmc
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ap = argparse.ArgumentParser()
ap.add_argument("--hcmc", default="hcmc")
a = ap.parse_args()
seen, rows, per_file, tags = set(), [], Counter(), Counter()
for f in sorted(glob.glob(os.path.join(a.hcmc, "labels_*.json"))):
    for line in open(f, encoding="utf-8"):
        if not line.strip():
            continue
        e = json.loads(line)
        if e["raw_file"] in seen:
            continue
        seen.add(e["raw_file"])
        rows.append(e)
        per_file[os.path.basename(f)] += 1
        tags.update(e.get("tags", []) or ["(không thẻ)"])
out = os.path.join(a.hcmc, "labels.json")
with open(out, "w", encoding="utf-8") as fo:
    fo.write("\n".join(json.dumps(e) for e in sorted(rows, key=lambda e: e["raw_file"])) + "\n")
by_vid = Counter(os.path.basename(e["raw_file"]).split("_")[0] for e in rows)
print(f"Đã gộp {len(rows)} khung -> {out}")
print("Theo người:", dict(per_file))
print("Theo video:", dict(by_vid))
print("Theo thẻ:", dict(tags))
