"""Công cụ gán nhãn nhanh vạch làn ego cho khung hình tự quay (xuất định dạng giống TuSimple).

  python tools/label_egopath.py --frames hcmc/frames --out hcmc/labels.json
  # Chia việc: mỗi người một danh sách khung, một tệp nhãn riêng (gộp lại bằng tools/merge_labels.py)
  python tools/label_egopath.py --frames hcmc/frames --list hcmc/phan_cong/bao.txt --out hcmc/labels_bao.json

Thao tác (cửa sổ OpenCV):
  Click trái     : thêm điểm cho vạch đang chọn (trái = xanh dương, phải = đỏ), click từ gần ra xa
  n              : chuyển giữa vạch TRÁI / PHẢI
  u              : xóa điểm cuối của vạch đang chọn
  1..5           : bật/tắt thẻ điều kiện: 1=night 2=no_marking 3=intersection 4=occlusion 5=rain
  s / Enter      : lưu khung hình và sang ảnh tiếp
  k              : bỏ qua khung hình (không có đường rõ ràng)
  q              : thoát (các ảnh đã lưu được giữ lại; chạy lại sẽ tiếp tục từ ảnh chưa gán)
Nếu vạch không có thật (đường không kẻ vạch), click theo mép đường đi được mà người lái sẽ bám theo.
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np

TAGS = {ord("1"): "night", ord("2"): "no_marking", ord("3"): "intersection", ord("4"): "occlusion", ord("5"): "rain"}
H_SAMPLES = list(range(160, 720, 10))  # giống TuSimple


def to_tusimple_lane(pts):
    """Nội suy các điểm click thành x tại H_SAMPLES; ngoài đoạn click -> -2."""
    if len(pts) < 2:
        return [-2] * len(H_SAMPLES)
    p = np.array(sorted(pts, key=lambda q: q[1]), dtype=np.float64)
    out = []
    for y in H_SAMPLES:
        out.append(int(round(np.interp(y, p[:, 1], p[:, 0]))) if p[0, 1] <= y <= p[-1, 1] else -2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--list", help="tệp văn bản, mỗi dòng một tên ảnh cần gán (chia việc cho từng người)")
    ap.add_argument("--auto-tags", default="vid03:night,vid04:rain",
                    help="thẻ bật sẵn theo tiền tố tên ảnh, ví dụ vid03:night (vẫn bật/tắt được bằng phím 1-5)")
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.frames, "*.jpg")) + glob.glob(os.path.join(a.frames, "*.png")))
    if a.list:
        want = {l.strip() for l in open(a.list, encoding="utf-8") if l.strip()}
        files = [f for f in files if os.path.basename(f) in want]
    auto = {}
    for item in filter(None, a.auto_tags.split(",")):
        pre, tag = item.split(":")
        auto.setdefault(pre, set()).add(tag)
    root = os.path.dirname(os.path.abspath(a.out))
    done = set()
    if os.path.exists(a.out):
        with open(a.out, encoding="utf-8") as f:
            done = {json.loads(l)["raw_file"] for l in f if l.strip()}
    skip_path = os.path.splitext(a.out)[0] + "_boqua.txt"   # khung đã bấm k, chạy lại không hỏi nữa
    if os.path.exists(skip_path):
        done |= {l.strip() for l in open(skip_path, encoding="utf-8") if l.strip()}
    n_total = len(files)
    state = {"lanes": [[], []], "cur": 0}

    def on_mouse(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN:
            state["lanes"][state["cur"]].append((x, y))

    cv2.namedWindow("label", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("label", on_mouse)
    for idx, fpath in enumerate(files, 1):
        rel = os.path.relpath(os.path.abspath(fpath), root).replace("\\", "/")
        if rel in done:
            continue
        img = cv2.imread(fpath)
        if img.shape[:2] != (720, 1280):
            img = cv2.resize(img, (1280, 720))
        state.update(lanes=[[], []], cur=0)
        tags = set().union(*[t for pre, t in auto.items() if os.path.basename(fpath).startswith(pre)])
        while True:
            vis = img.copy()
            for li, col in enumerate([(255, 120, 0), (0, 0, 255)]):
                for q in state["lanes"][li]:
                    cv2.circle(vis, q, 5, col, -1)
                if len(state["lanes"][li]) > 1:
                    cv2.polylines(vis, [np.array(state["lanes"][li], np.int32)], False, col, 2)
            side = "TRAI" if state["cur"] == 0 else "PHAI"
            cv2.putText(vis, f"[{idx}/{n_total}] {os.path.basename(fpath)} | vach: {side} | tags: {','.join(sorted(tags)) or '-'}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow("label", vis)
            k = cv2.waitKey(30) & 0xFF
            if k == ord("n"):
                state["cur"] ^= 1
            elif k == ord("u") and state["lanes"][state["cur"]]:
                state["lanes"][state["cur"]].pop()
            elif k in TAGS:
                tags ^= {TAGS[k]}
            elif k in (ord("s"), 13):
                if min(len(state["lanes"][0]), len(state["lanes"][1])) < 2:
                    print("Cần ít nhất 2 điểm cho mỗi vạch")
                    continue
                entry = {"raw_file": rel, "h_samples": H_SAMPLES,
                         "lanes": [to_tusimple_lane(state["lanes"][0]), to_tusimple_lane(state["lanes"][1])],
                         "tags": sorted(tags)}
                with open(a.out, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry) + "\n")
                break
            elif k == ord("k"):
                with open(skip_path, "a", encoding="utf-8") as f:
                    f.write(rel + "\n")
                break
            elif k == ord("q"):
                cv2.destroyAllWindows()
                return
    cv2.destroyAllWindows()
    print("Đã gán nhãn xong toàn bộ khung hình.")


if __name__ == "__main__":
    main()
