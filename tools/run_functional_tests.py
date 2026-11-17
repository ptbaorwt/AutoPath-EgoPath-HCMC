"""Chạy tự động các ca kiểm thử chức năng của chương trình demo và công cụ gán nhãn (Bảng 5.x trong báo cáo).

  python tools/run_functional_tests.py --ckpt runs/res18_aug/best.pt --autosteer-repo auto_steer --out results/functest

Nếu không truyền --ckpt, chương trình tạo một checkpoint khởi tạo ngẫu nhiên: đủ để kiểm thử chức năng
(đọc video, xử lý lô, ghi kết quả) nhưng không dùng để đánh giá độ chính xác.
"""
import argparse
import glob
import json
import os
import subprocess
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def make_video(path, w, h, n, fourcc, kind="road"):
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc), 30, (w, h))
    for t in range(n):
        if kind == "black":
            img = np.zeros((h, w, 3), np.uint8)
        else:
            img = np.full((h, w, 3), (120, 115, 110), np.uint8)
            img[: int(h * 0.42)] = (200, 170, 140)
            for base in (-0.18, 0.18):
                p0 = (int(w * (0.5 + base * 2.2 + 0.02 * np.sin(t / 8))), h - 1)
                cv2.line(img, p0, (int(w * (0.5 + base * 0.15)), int(h * 0.45)), (240, 240, 240), max(3, w // 200))
        vw.write(img)
    vw.release()


def count_frames(path):
    cap = cv2.VideoCapture(path)
    n = 0
    size = None
    while True:
        ok, f = cap.read()
        if not ok:
            break
        n += 1
        size = f.shape[1::-1]
    cap.release()
    return n, size


def demo(args, extra, timeout=900):
    cmd = [sys.executable, os.path.join(ROOT, "demo_video.py")] + extra
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=ROOT, env=env, timeout=timeout)
    return r.returncode, (r.stdout + r.stderr).strip()


def test_label_resume(tmp):
    """Mô phỏng người gán nhãn: gán khung 1, thoát (q), chạy lại; kiểm tra khung 1 không bị gán lại."""
    import importlib
    frames = os.path.join(tmp, "frames")
    os.makedirs(frames, exist_ok=True)
    for i in range(3):
        cv2.imwrite(os.path.join(frames, f"f{i}.jpg"), np.full((720, 1280, 3), 100, np.uint8))
    out = os.path.join(tmp, "labels.json")
    if os.path.exists(out):
        os.remove(out)
    import tools.label_egopath as L
    importlib.reload(L)
    shown = []

    def session(keys, clicks):
        cb = {}
        keys, clicks = list(keys), list(clicks)
        L.cv2.namedWindow = lambda *a, **k: None
        L.cv2.destroyAllWindows = lambda *a, **k: None
        L.cv2.setMouseCallback = lambda name, f: cb.setdefault("f", f)
        L.cv2.imshow = lambda name, img: shown.append(1)

        def waitKey(_):
            if clicks and clicks[0] is not None:
                x, y = clicks.pop(0)
                cb["f"](cv2.EVENT_LBUTTONDOWN, x, y)
                return 255
            if clicks:
                clicks.pop(0)
            return ord(keys.pop(0)) if keys else ord("q")
        L.cv2.waitKey = waitKey
        sys.argv = ["label_egopath.py", "--frames", frames, "--out", out]
        L.main()

    # Phiên 1: vạch trái 2 điểm, 'n' sang vạch phải, 2 điểm, 's' lưu, rồi 'q' thoát
    session(["n", "s", "q"], [(500, 700), (600, 400), None, (800, 700), (700, 400), None])
    n1 = sum(1 for _ in open(out, encoding="utf-8"))
    # Phiên 2: chỉ thoát ngay -> không được gán lại f0 nên số dòng giữ nguyên, và khung đầu tiên hiển thị phải là f1
    seen = []
    orig = L.cv2.imread
    L.cv2.imread = lambda p, *a: (seen.append(os.path.basename(p)), orig(p, *a))[1]
    session(["q"], [])
    L.cv2.imread = orig
    n2 = sum(1 for _ in open(out, encoding="utf-8"))
    ok = n1 == 1 and n2 == 1 and seen[:1] == ["f1.jpg"]
    return ok, f"phiên 1 lưu {n1} nhãn; phiên 2 bắt đầu từ {seen[:1]}, tổng {n2} nhãn"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--autosteer-repo", default="auto_steer")
    ap.add_argument("--autosteer-weights")
    ap.add_argument("--out", default="results/functest")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tmp = os.path.abspath(a.out)
    ckpt = a.ckpt
    if not ckpt:
        from autopath.model import AutoPathNet, save_checkpoint
        ckpt = os.path.join(tmp, "random_init.pt")
        save_checkpoint(ckpt, AutoPathNet("resnet18", pretrained=False))
    ckpt = os.path.abspath(ckpt)
    v1, v2, v3, v4 = (os.path.join(tmp, n) for n in ("t1_1080p.mp4", "t2_720p.avi", "t3_black.mp4", "t4_short.mp4"))
    make_video(v1, 1920, 1080, 45, "mp4v")
    make_video(v2, 1280, 720, 45, "MJPG")
    make_video(v3, 1280, 720, 30, "mp4v", kind="black")
    make_video(v4, 1280, 720, 5, "mp4v")
    res = []

    def add(i, name, ok, note):
        res.append(dict(id=i, case=name, passed=bool(ok), note=note))
        print(f"[{'ĐẠT' if ok else 'LỖI'}] {i}. {name}: {note}")

    for i, (v, name, n) in enumerate([(v1, "Video MP4 1920x1080 30 FPS", 45), (v2, "Video AVI 1280x720", 45)], 1):
        o = os.path.join(tmp, f"out{i}.mp4")
        rc, log = demo(a, ["--video", v, "--ckpt", ckpt, "--out", o, "--stats-json", o + ".json"])
        cnt, size = count_frames(o) if rc == 0 else (0, None)
        st = json.load(open(o + ".json", encoding="utf-8")) if rc == 0 else {}
        add(i, name, rc == 0 and cnt == n and size == (1024, 512),
            f"{cnt}/{n} khung, kích thước {size}, FPS toàn hệ thống {st.get('fps_system')}")

    o = os.path.join(tmp, "out3.mp4")
    rc, log = demo(a, ["--video", v3, "--ckpt", ckpt, "--out", o, "--alphas", "1.0", "--no-write"])
    add(3, "Khung hình không có đường (toàn đen)", rc == 0, "chạy hết video, không dừng" if rc == 0 else log[-300:])

    o = os.path.join(tmp, "out4.mp4")
    rc, log = demo(a, ["--video", v4, "--ckpt", ckpt, "--out", o])
    cnt, _ = count_frames(o) if rc == 0 else (0, None)
    add(4, "Video 5 khung (nhỏ hơn kích thước lô 8)", rc == 0 and cnt == 5, f"xuất {cnt}/5 khung")

    rc, log = demo(a, ["--video", os.path.join(tmp, "khong_ton_tai.mp4"), "--ckpt", ckpt])
    add(5, "Đường dẫn video không tồn tại", rc != 0 and "Không mở được video" in log, "báo lỗi: " + log.splitlines()[-1][:120])

    oks, notes = [], []
    for m in ("autosteer", "classical"):
        o = os.path.join(tmp, f"out6_{m}.mp4")
        extra = ["--video", v4, "--method", m, "--out", o, "--autosteer-repo", a.autosteer_repo]
        if a.autosteer_weights:
            extra += ["--autosteer-weights", a.autosteer_weights]
        rc, log = demo(a, extra)
        cnt, size = count_frames(o) if rc == 0 else (0, None)
        oks.append(rc == 0 and cnt == 5 and size == (1024, 512))
        notes.append(f"{m}: {cnt} khung {size}" if rc == 0 else f"{m}: {log[-200:]}")
    add(6, "Chạy với --method autosteer và --method classical", all(oks), "; ".join(notes))

    o = os.path.join(tmp, "out7.mp4")
    alphas = ["0.1", "0.2", "0.35", "0.5", "0.7", "1.0"]
    rc, log = demo(a, ["--video", v1, "--ckpt", ckpt, "--out", o, "--alphas", *alphas, "--no-write"])
    csvp = os.path.splitext(o)[0] + "_alpha_sweep.csv"
    rows = sum(1 for _ in open(csvp)) - 1 if os.path.exists(csvp) else 0
    add(7, "Quét alpha với --alphas và --no-write", rc == 0 and rows == len(alphas) and not os.path.exists(o),
        f"CSV có {rows}/{len(alphas)} dòng, không ghi video: {not os.path.exists(o)}")

    try:
        ok, note = test_label_resume(tmp)
    except Exception as e:  # noqa
        ok, note = False, repr(e)
    add(8, "Gán nhãn, thoát giữa chừng rồi chạy lại", ok, note)

    json.dump(res, open(os.path.join(tmp, "functional_tests.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"{sum(r['passed'] for r in res)}/{len(res)} ca đạt -> {os.path.join(tmp, 'functional_tests.json')}")


if __name__ == "__main__":
    main()
