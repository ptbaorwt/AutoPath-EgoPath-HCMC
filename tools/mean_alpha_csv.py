"""Gộp nhiều tệp *_alpha_sweep.csv (mỗi video một tệp) thành một tệp trung bình theo alpha (Bảng Thí nghiệm 4)."""
import csv
import glob
import sys
from collections import defaultdict

out, pats = sys.argv[1], sys.argv[2:]
acc = defaultdict(lambda: defaultdict(list))
files = [f for p in pats for f in glob.glob(p)]
for f in files:
    for r in csv.DictReader(open(f)):
        for k in ("jitter_px", "jitter_raw_px", "lag_px"):
            acc[float(r["alpha"])][k].append(float(r[k]))
with open(out, "w", newline="") as fo:
    w = csv.writer(fo)
    w.writerow(["alpha", "jitter_px", "jitter_raw_px", "lag_px"])
    for al in sorted(acc):
        w.writerow([al] + [f"{sum(v) / len(v):.3f}" for v in (acc[al]["jitter_px"], acc[al]["jitter_raw_px"], acc[al]["lag_px"])])
print(f"Gộp {len(files)} tệp -> {out}")
