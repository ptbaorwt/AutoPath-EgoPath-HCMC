"""Trích khung hình từ video dashcam tự quay, đưa về 1280x720 (cùng hình học với TuSimple).

  python tools/extract_frames.py --video raw/vid01.mp4 --out hcmc/frames --every-sec 1.0
  # Video dài (tour lái xe trên YouTube): bỏ phần mở đầu/kết thúc và lấy thưa hơn
  python tools/extract_frames.py --video hcmc/raw/vid01.mp4 --out hcmc/frames --every-sec 10 --start 1:30 --end 45:00
Tên file: <tên video>_<số thứ tự>.jpg. Ghi kèm frames_index.csv (video, thời điểm) để lập bảng thống kê dữ liệu.
"""
import argparse
import csv
import os

import cv2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True, nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--every-sec", type=float, default=1.0)
    p.add_argument("--start", default="0", help="bỏ qua đoạn đầu, dạng giây hoặc mm:ss / hh:mm:ss")
    p.add_argument("--end", default="", help="dừng tại thời điểm này (để trống = hết video)")
    a = p.parse_args()

    def sec(t):
        if not t:
            return None
        v = 0.0
        for part in str(t).split(":"):
            v = v * 60 + float(part)
        return v
    t_start, t_end = sec(a.start) or 0.0, sec(a.end)
    os.makedirs(a.out, exist_ok=True)
    idx_path = os.path.join(a.out, "frames_index.csv")
    new = not os.path.exists(idx_path)
    with open(idx_path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["file", "video", "time_sec"])
        for v in a.video:
            cap = cv2.VideoCapture(v)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            step = max(1, int(round(fps * a.every_sec)))
            stem = os.path.splitext(os.path.basename(v))[0]
            i = k = 0
            if t_start > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(t_start * fps))
                i = int(t_start * fps)
            while True:
                ok, fr = cap.read()
                if not ok or (t_end is not None and i / fps > t_end):
                    break
                if i % step == 0:
                    name = f"{stem}_{k:05d}.jpg"
                    cv2.imwrite(os.path.join(a.out, name), cv2.resize(fr, (1280, 720), interpolation=cv2.INTER_AREA))
                    w.writerow([name, stem, f"{i / fps:.2f}"])
                    k += 1
                i += 1
            cap.release()
            print(f"{v}: {k} khung hình")


if __name__ == "__main__":
    main()
