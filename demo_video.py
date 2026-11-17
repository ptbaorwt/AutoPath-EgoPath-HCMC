"""Demo: vẽ đường đi dự đoán lên video dashcam, có làm mượt EMA; đồng thời đo jitter/lag cho RQ4.

Ví dụ:
  python demo_video.py --video hcmc_01.mp4 --ckpt runs/res18_aug/best.pt --alpha 0.35 --out demo.mp4
  # Thí nghiệm 4: quét nhiều alpha, không ghi video
  python demo_video.py --video hcmc_01.mp4 --ckpt runs/res18_aug/best.pt --alphas 0.1 0.2 0.35 0.5 0.7 1.0 --no-write
  # Bật module cảnh báo vật cản trên đường đi (cần: pip install ultralytics)
  python demo_video.py --video hcmc_01.mp4 --ckpt runs/res18_aug/best.pt --warn --out demo_warn.mp4
"""
import argparse
import csv
import json
import os
import time

import cv2
import numpy as np

from autopath.data import crop_resize
from autopath.predictors import build_predictor
from autopath.smoothing import EMASmoother, draw_path, temporal_stats


def parse_time(t):
    """'90', '1:30' hoặc '0:01:30' -> số giây."""
    v = 0.0
    for part in str(t or 0).split(":"):
        v = v * 60 + float(part)
    return v


def open_at(video, start_sec=0.0):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"Không mở được video: {video}")
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
    if start_sec > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_sec * fps_in))
    return cap, fps_in


def run(pred, video, batch=8, max_frames=0, start_sec=0.0):
    """Suy luận video một lần (từ start_sec, tối đa max_frames khung), trả về chuỗi (xp, prob) thô và FPS mô hình."""
    cap, fps_in = open_at(video, start_sec)
    raw_x, raw_p, buf, t_model, n = [], [], [], 0.0, 0
    while True:
        ok, f = cap.read()
        if ok:
            buf.append(f)
        if buf and (len(buf) == batch or not ok):
            t0 = time.perf_counter()
            x, pr = pred.predict(buf)
            t_model += time.perf_counter() - t0
            raw_x.extend(x); raw_p.extend(pr); n += len(buf); buf = []
        if not ok or (max_frames and n >= max_frames):
            break
    cap.release()
    return np.array(raw_x), np.array(raw_p), fps_in, n / max(t_model, 1e-9)


def smooth_seq(raw_x, raw_p, alpha):
    ema = EMASmoother(alpha)
    return np.array([ema(x, p)[0] for x, p in zip(raw_x, raw_p)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--method", default="autopath", choices=["autopath", "autosteer", "classical"])
    ap.add_argument("--ckpt")
    ap.add_argument("--align-vp", type=float, nargs=2, metavar=("VX", "VY"),
                    help="căn góc camera: điểm tụ của video (px trên ảnh 1280x720), xem autopath/align.py")
    ap.add_argument("--align-ref", type=float, nargs=2, default=[260, 0.9], metavar=("VY_REF", "SCALE"))
    ap.add_argument("--autosteer-repo", default="auto_steer")
    ap.add_argument("--autosteer-weights")
    ap.add_argument("--alpha", type=float, default=0.35)
    ap.add_argument("--alphas", type=float, nargs="*", default=[])
    ap.add_argument("--out", default="results/demo.mp4")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--start", default="0", help="bắt đầu từ thời điểm này (giây hoặc mm:ss), bỏ phần mở đầu video")
    ap.add_argument("--warn", action="store_true", help="cảnh báo người/phương tiện nằm trên đường đi (YOLOv8n)")
    ap.add_argument("--yolo", default="yolov8n.pt")
    ap.add_argument("--stats-json", help="ghi số khung, FPS mô hình, FPS toàn hệ thống ra tệp JSON")
    a = ap.parse_args()

    pred = build_predictor(a.method, a.ckpt, a.autosteer_repo, a.autosteer_weights)
    if a.align_vp:
        from autopath.align import AlignedPredictor
        pred = AlignedPredictor(pred, tuple(a.align_vp), *a.align_ref)
    t_start = time.perf_counter()
    raw_x, raw_p, fps_in, fps_model = run(pred, a.video, max_frames=a.max_frames, start_sec=parse_time(a.start))
    t_infer = time.perf_counter() - t_start
    if len(raw_x) == 0:
        raise SystemExit(f"Video không có khung hình nào: {a.video}")
    valid = raw_p >= 0.5
    print(f"{len(raw_x)} khung hình, tốc độ mô hình (gồm tiền xử lý) ≈ {fps_model:.1f} FPS")
    stats = dict(video=a.video, method=pred.name, frames=int(len(raw_x)), fps_model=round(fps_model, 2),
                 fps_read_and_infer=round(len(raw_x) / t_infer, 2))

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    if a.alphas:
        csv_path = os.path.splitext(a.out)[0] + "_alpha_sweep.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["alpha", "jitter_px", "jitter_raw_px", "lag_px"])
            for al in a.alphas:
                s = temporal_stats(raw_x, smooth_seq(raw_x, raw_p, al), valid)
                w.writerow([al, f"{s['jitter_px']:.3f}", f"{s['jitter_raw_px']:.3f}", f"{s['lag_px']:.3f}"])
                print(f"alpha={al:<5} jitter={s['jitter_px']:.2f}px (thô {s['jitter_raw_px']:.2f}px)  lag={s['lag_px']:.2f}px")
        print(f"-> {csv_path}")
    if a.no_write:
        if a.stats_json:
            json.dump(stats, open(a.stats_json, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        return
    warner = None
    if a.warn:
        from autopath.hazard import HazardWarner
        warner = HazardWarner(a.yolo)
    t0 = time.perf_counter()
    sm = smooth_seq(raw_x, raw_p, a.alpha)
    cap, _ = open_at(a.video, parse_time(a.start))
    out_w, out_h = 1024, 512
    vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (out_w, out_h))
    n_warn, t_warn = 0, 0.0
    for i in range(len(sm)):
        ok, f = cap.read()
        if not ok:
            break
        base = crop_resize(f, out_w, out_h)
        img = draw_path(base, sm[i], raw_p[i])
        if warner is not None:
            tw = time.perf_counter()
            boxes, flags = warner.check(base, sm[i], raw_p[i])
            t_warn += time.perf_counter() - tw
            img = warner.draw(img, boxes, flags)
            n_warn += any(flags)
        cv2.putText(img, f"{pred.name} | EMA alpha={a.alpha} | {fps_model:.0f} FPS", (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        vw.write(img)
    vw.release(); cap.release()
    t_total = t_infer + time.perf_counter() - t0
    # FPS toàn hệ thống: đọc + tiền xử lý + mô hình + làm mượt + vẽ + ghi video (+ YOLO nếu --warn)
    stats.update(fps_system=round(len(sm) / t_total, 2), alpha=a.alpha, out=a.out)
    if warner is not None:
        stats.update(frames_with_warning=int(n_warn), yolo_ms_per_frame=round(1000 * t_warn / len(sm), 2))
    print(f"FPS toàn hệ thống ≈ {stats['fps_system']:.1f}")
    if warner is not None:
        print(f"Số khung có cảnh báo: {n_warn}/{len(sm)}")
    if a.stats_json:
        json.dump(stats, open(a.stats_json, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"Đã ghi video demo: {a.out}")


if __name__ == "__main__":
    main()
