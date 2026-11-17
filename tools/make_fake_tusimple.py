"""Sinh một bộ dữ liệu giả cùng cấu trúc TuSimple (chỉ để kiểm thử pipeline, không dùng cho báo cáo).

  python tools/make_fake_tusimple.py --out /tmp/fake_tusimple --n-train 60 --n-test 20 --video
"""
import argparse
import sys
import json
import os

import cv2
import numpy as np

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

H_SAMPLES = list(range(160, 720, 10))


def render(rng, curve, offset):
    img = np.full((720, 1280, 3), (135, 120, 105), np.uint8)
    img[:300] = (200, 170, 140)
    lanes = []
    for k, base in enumerate([-560, -180, 180, 560]):
        xs = []
        for y in H_SAMPLES:
            t = (y - 260) / 460.0
            x = 640 + offset + base * t + curve * (1 - t) ** 2 * 300 if y >= 260 else -2
            xs.append(int(x) if 0 <= x < 1280 and y >= 260 else -2)
        lanes.append(xs)
        pts = [(x, y) for x, y in zip(xs, H_SAMPLES) if x >= 0]
        if len(pts) > 1:
            cv2.polylines(img, [np.array(pts, np.int32)], False, (240, 240, 240), 6 if k in (1, 2) else 4)
    noise = rng.normal(0, 6, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8), lanes


def write_split(root, sub, json_name, n, rng, json_at_root=False):
    entries = []
    for i in range(n):
        clip = f"clips/0313-{i % 2 + 1}/{1000 + i // 2}"
        os.makedirs(os.path.join(root, sub, clip), exist_ok=True)
        img, lanes = render(rng, rng.uniform(-0.6, 0.6), rng.uniform(-60, 60))
        cv2.imwrite(os.path.join(root, sub, clip, "20.jpg"), img)
        entries.append({"raw_file": f"{clip}/20.jpg", "lanes": lanes, "h_samples": H_SAMPLES})
    jpath = os.path.join(root, json_name) if json_at_root else os.path.join(root, sub, json_name)
    with open(jpath, "w") as f:
        f.write("\n".join(json.dumps(e) for e in entries) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=60)
    ap.add_argument("--n-test", type=int, default=20)
    ap.add_argument("--video", action="store_true")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    write_split(a.out, "train_set", "label_data_0313.json", a.n_train, rng)
    write_split(a.out, "test_set", "test_label.json", a.n_test, rng, json_at_root=True)
    if a.video:
        vw = cv2.VideoWriter(os.path.join(a.out, "fake_drive.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (1280, 720))
        for t in range(90):
            img, _ = render(rng, 0.5 * np.sin(t / 15), 30 * np.sin(t / 10))
            vw.write(img)
        vw.release()
    print(f"Đã tạo dữ liệu giả tại {a.out}")


if __name__ == "__main__":
    main()
