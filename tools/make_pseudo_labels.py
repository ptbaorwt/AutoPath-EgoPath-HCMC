"""Sinh nhãn giả (pseudo-label) cho tinh chỉnh AutoPath trên video TP.HCM, dùng AutoSteer 2.0 làm "thầy".

Khung huấn luyện được trích cách đều từ video gốc, BỎ QUA:
  * ±GAP giây quanh mọi khung ứng viên của tập kiểm thử (hcmc/frames/frames_index.csv), để không rò rỉ dữ liệu test;
  * đoạn [DEMO_START, DEMO_END] dùng làm video demo.
Chỉ giữ khung mà thầy tự tin (xác suất trung bình cao, đường đi đủ dài, không giật) để giảm nhãn sai.

  python tools/make_pseudo_labels.py --videos hcmc/raw/vid01.mp4 hcmc/raw/vid03.mp4 hcmc/raw/vid04.mp4 \
      --out hcmc/pseudo --autosteer-repo auto_steer --autosteer-weights weights/autosteer.pth
"""
import argparse, csv, json, os, sys
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autopath.predictors import AutoSteerPredictor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--out", default="hcmc/pseudo")
    ap.add_argument("--test-index", default="hcmc/frames/frames_index.csv")
    ap.add_argument("--every-sec", type=float, default=2.0)
    ap.add_argument("--gap", type=float, default=8.0)
    ap.add_argument("--skip-start", type=float, default=60.0)
    ap.add_argument("--demo-start", type=float, default=60.0)
    ap.add_argument("--demo-end", type=float, default=240.0)
    ap.add_argument("--min-conf", type=float, default=0.8)
    ap.add_argument("--min-rows", type=int, default=12)
    ap.add_argument("--max-jump", type=float, default=0.03, help="bước nhảy hoành độ tối đa giữa 2 hàng liền kề")
    ap.add_argument("--max-frames", type=int, default=0, help="giới hạn số khung mỗi video (chạy thử)")
    ap.add_argument("--autosteer-repo", default="auto_steer")
    ap.add_argument("--autosteer-weights", default="weights/autosteer.pth")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "frames"), exist_ok=True)
    test_t = {}
    if os.path.exists(a.test_index):
        for r in csv.DictReader(open(a.test_index)):
            test_t.setdefault(r["video"], []).append(float(r["time_sec"]))
    teacher = AutoSteerPredictor(a.autosteer_repo, a.autosteer_weights)
    files, XP, PR, KEEP, META = [], [], [], [], []
    for v in a.videos:
        stem = os.path.splitext(os.path.basename(v))[0]
        tt = np.array(test_t.get(stem, []))
        cap = cv2.VideoCapture(v); fps = cap.get(cv2.CAP_PROP_FPS) or 30
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); step = max(1, int(round(fps * a.every_sec)))
        buf, names, k = [], [], 0
        def flush():
            if not buf: return
            xp, pr = teacher.predict(buf)
            for nm, x, p in zip(names, xp, pr):
                m = p >= 0.5
                ok = m.sum() >= a.min_rows and p[m].mean() >= a.min_conf
                if ok:
                    idx = np.where(m)[0]; ok = (idx.max() - idx.min() + 1 == len(idx)) and np.abs(np.diff(x[m])).max() <= a.max_jump
                files.append(nm); XP.append(x); PR.append(p); KEEP.append(bool(ok))
            buf.clear(); names.clear()
        for i in range(n):
            t = i / fps
            if i % step or t < a.skip_start or (a.demo_start <= t <= a.demo_end) or (len(tt) and np.abs(tt - t).min() < a.gap):
                if not cap.grab(): break
                continue
            ok, fr = cap.read()
            if not ok: break
            fr = cv2.resize(fr, (1280, 720), interpolation=cv2.INTER_AREA)
            nm = f"{stem}_p{k:05d}.jpg"; cv2.imwrite(os.path.join(a.out, "frames", nm), fr, [cv2.IMWRITE_JPEG_QUALITY, 90])
            META.append({"file": nm, "video": stem, "time_sec": round(t, 2)})
            buf.append(fr); names.append(nm); k += 1
            if len(buf) == 16: flush()
            if a.max_frames and k >= a.max_frames: break
        flush(); cap.release()
        print(f"{stem}: {k} khung, giữ {sum(KEEP[-k:]) if k else 0}", flush=True)
    np.savez_compressed(os.path.join(a.out, "pseudo.npz"), files=np.array(files), xp=np.array(XP, np.float32),
                        prob=np.array(PR, np.float32), keep=np.array(KEEP))
    json.dump(META, open(os.path.join(a.out, "meta.json"), "w"))
    print(f"Tổng: {len(files)} khung, giữ {int(np.sum(KEEP))} khung có nhãn giả tin cậy -> {a.out}")


if __name__ == "__main__":
    main()
