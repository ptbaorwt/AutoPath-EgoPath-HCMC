"""Vẽ lưới ảnh xem trước nhãn giả: khung được giữ (xanh) và khung bị loại (đỏ), để kiểm tra chất lượng trước khi huấn luyện."""
import argparse, os, random
import cv2, numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--pseudo", default="hcmc/pseudo"); ap.add_argument("--out", default="hcmc/pseudo/preview")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
z = np.load(os.path.join(a.pseudo, "pseudo.npz")); F, X, P, K = z["files"], z["xp"], z["prob"], z["keep"]
ROW = 80 + np.linspace(0, 1, 64) * 640
random.seed(1)
def tile(i, ok):
    im = cv2.imread(os.path.join(a.pseudo, "frames", str(F[i])))
    pts = np.array([(X[i][k] * 1280, ROW[k]) for k in range(64) if P[i][k] >= 0.5], np.int32)
    if len(pts) > 1: cv2.polylines(im, [pts], False, (0, 255, 0) if ok else (0, 0, 255), 8)
    cv2.putText(im, f"{F[i]} {'GIU' if ok else 'LOAI'}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 4)
    return cv2.resize(im, (640, 360))
for tag, ok, n in [("giu", True, 24), ("loai", False, 12)]:
    idx = [i for i in range(len(F)) if bool(K[i]) == ok]
    by = {}
    for i in idx: by.setdefault(str(F[i])[:5], []).append(i)
    pick = []
    for v, ii in sorted(by.items()): pick += random.sample(ii, min(len(ii), n // max(1, len(by))))
    for p in range(0, len(pick), 12):
        ts = [tile(i, ok) for i in pick[p:p + 12]]
        while len(ts) % 3: ts.append(np.zeros_like(ts[0]))
        g = np.vstack([np.hstack(ts[r:r + 3]) for r in range(0, len(ts), 3)])
        cv2.imwrite(os.path.join(a.out, f"{tag}_{p // 12 + 1}.jpg"), g, [cv2.IMWRITE_JPEG_QUALITY, 85])
by = {}
for f, k in zip(F, K): by.setdefault(str(f)[:5], [0, 0]); by[str(f)[:5]][0] += 1; by[str(f)[:5]][1] += int(k)
print("Thống kê (tổng, giữ):", by)
