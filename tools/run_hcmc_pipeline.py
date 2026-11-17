"""Chạy toàn bộ phần thực nghiệm trên tập TP.HCM (Kaggle hoặc máy có GPU), có điểm lưu để chạy lại không mất kết quả.

  python tools/run_hcmc_pipeline.py --hcmc /kaggle/input/autopath-hcmc/hcmc --runs runs --out /kaggle/working/hcmc_out \
      --autosteer-repo auto_steer --autosteer-weights weights/autosteer.pth

Các bước (mỗi bước ghi tệp đánh dấu trong <out>/_done/, lần sau tự bỏ qua):
  A. predict  Chạy 4 phương pháp trên các khung ứng viên (không cần nhãn), lưu .npz và lưới ảnh định tính.
  B. alpha    Thí nghiệm 4: quét alpha trên từng đoạn video, gộp trung bình.
  C. demo     Video demo có làm mượt (alpha = 0,35) và cảnh báo vật cản, kèm FPS toàn hệ thống.
  D. eval     Chỉ chạy khi có labels.json: đánh giá 5 phương pháp. Đánh dấu theo mã băm của labels.json,
              nên khi cập nhật nhãn (thêm khung) thì bước này tự chạy lại.
  E. figures  Bảng và hình cho báo cáo.
Một bước lỗi không làm dừng các bước sau; lỗi được ghi vào <out>/errors.log.
"""
import argparse
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

MODELS = [("res18_aug", "autopath"), ("res18_noaug", "autopath"), ("mbv3s_aug", "autopath"),
          ("autosteer", "autosteer"), ("classical", "classical")]


def sh(cmd, log):
    print("  $", " ".join(cmd))
    with open(log, "a", encoding="utf-8") as f:
        r = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                           env=dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1"))
    if r.returncode:
        raise RuntimeError(f"lệnh lỗi (mã {r.returncode}), xem {log}")


class Stages:
    def __init__(self, out):
        self.out = out
        self.dir = os.path.join(out, "_done")
        os.makedirs(self.dir, exist_ok=True)

    def done(self, key):
        return os.path.exists(os.path.join(self.dir, key))

    def mark(self, key, info=""):
        open(os.path.join(self.dir, key), "w", encoding="utf-8").write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {info}")

    def run(self, key, fn):
        if self.done(key):
            print(f"[bỏ qua] {key} (đã xong)")
            return
        print(f"[chạy] {key}")
        t0 = time.time()
        try:
            info = fn() or ""
            self.mark(key, info)
            print(f"[xong] {key} ({time.time() - t0:.0f}s)")
        except Exception:
            msg = traceback.format_exc()
            print(f"[LỖI] {key}\n{msg}")
            with open(os.path.join(self.out, "errors.log"), "a", encoding="utf-8") as f:
                f.write(f"--- {key} {time.strftime('%Y-%m-%d %H:%M:%S')}\n{msg}\n")


def ckpt_of(runs, name):
    p = os.path.join(runs, name, "best.pt")
    return p if os.path.exists(p) else None


def predictor_args(a, name, kind):
    if kind == "autopath":
        return ["--method", "autopath", "--ckpt", ckpt_of(a.runs, name)]
    if kind == "autosteer":
        return ["--method", "autosteer", "--autosteer-repo", a.autosteer_repo, "--autosteer-weights", a.autosteer_weights]
    return ["--method", "classical"]


def available(a):
    out = []
    for name, kind in MODELS:
        if kind == "autopath" and not ckpt_of(a.runs, name):
            continue
        if kind == "autosteer" and not (a.autosteer_weights and os.path.exists(a.autosteer_weights)):
            continue
        out.append((name, kind))
    return out


# ---------------------------------------------------------------- A. dự đoán trên khung ứng viên
def stage_predict(a, name, kind):
    import numpy as np
    import cv2
    from autopath.predictors import build_predictor
    frames = sorted(glob.glob(os.path.join(a.hcmc, "frames", "*.jpg")))
    if not frames:
        raise RuntimeError("không có khung hình trong hcmc/frames")
    pr = build_predictor(kind, ckpt_of(a.runs, name), a.autosteer_repo, a.autosteer_weights)
    xs, ps = [], []
    for i in range(0, len(frames), 16):
        x, p = pr.predict([cv2.imread(f) for f in frames[i:i + 16]])
        xs.append(x); ps.append(p)
    d = os.path.join(a.out, "predict"); os.makedirs(d, exist_ok=True)
    np.savez_compressed(os.path.join(d, f"{name}.npz"), files=np.array([os.path.basename(f) for f in frames]),
                        xp=np.concatenate(xs), prob=np.concatenate(ps))
    return f"{len(frames)} khung"


def stage_predict_figure(a):
    import numpy as np
    import cv2
    from autopath.data import crop_resize
    from autopath.smoothing import draw_path
    d = os.path.join(a.out, "predict")
    preds = {os.path.splitext(os.path.basename(p))[0]: np.load(p) for p in sorted(glob.glob(os.path.join(d, "*.npz")))}
    files = list(next(iter(preds.values()))["files"])
    vids = sorted({f.split("_")[0] for f in files})
    # Lưới 1: AutoPath-ResNet18 trên 2 khung mỗi video
    key = "res18_aug" if "res18_aug" in preds else next(iter(preds))
    pick = []
    for v in vids:
        fv = [i for i, f in enumerate(files) if f.startswith(v)]
        pick += [fv[len(fv) // 3], fv[2 * len(fv) // 3]]
    tiles = []
    for i in pick:
        im = crop_resize(cv2.imread(os.path.join(a.hcmc, "frames", files[i])), 640, 320)
        tiles.append(draw_path(im, preds[key]["xp"][i], preds[key]["prob"][i]))
    rows = [np.hstack(tiles[k:k + 2]) for k in range(0, len(tiles), 2)]
    cv2.imwrite(os.path.join(a.out, "hcmc_predictions_res18.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])
    # Lưới 2: so sánh các phương pháp trên 1 khung mỗi video (hàng = video, cột = phương pháp)
    order = [k for k in ("classical", "autosteer", "res18_aug", "mbv3s_aug") if k in preds]
    label = {"classical": "Canny+Hough", "autosteer": "AutoSteer 2.0", "res18_aug": "AutoPath-ResNet18", "mbv3s_aug": "AutoPath-MBV3"}
    grid = []
    for v in vids:
        i = [j for j, f in enumerate(files) if f.startswith(v)][len([f for f in files if f.startswith(v)]) // 2]
        base = crop_resize(cv2.imread(os.path.join(a.hcmc, "frames", files[i])), 480, 240)
        row = []
        for k in order:
            im = draw_path(base.copy(), preds[k]["xp"][i], preds[k]["prob"][i])
            cv2.putText(im, label[k], (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            row.append(im)
        grid.append(np.hstack(row))
    cv2.imwrite(os.path.join(a.out, "hcmc_compare_methods.jpg"), np.vstack(grid), [cv2.IMWRITE_JPEG_QUALITY, 90])
    # Thống kê: tỉ lệ hàng được dự đoán là có đường đi (không cần nhãn)
    stats = {k: {v: float((p["prob"][[f.startswith(v) for f in files]] >= 0.5).mean()) for v in vids} for k, p in preds.items()}
    json.dump(stats, open(os.path.join(a.out, "predict_valid_ratio.json"), "w"), indent=2)


# ---------------------------------------------------------------- B, C
def clips(a):
    return sorted(glob.glob(os.path.join(a.hcmc, "clips", "*.mp4")))


def stage_alpha(a, clip):
    stem = os.path.splitext(os.path.basename(clip))[0]
    sh([sys.executable, "demo_video.py", "--video", clip, "--ckpt", ckpt_of(a.runs, "res18_aug"),
        "--alphas", "0.1", "0.2", "0.35", "0.5", "0.7", "1.0", "--no-write",
        "--out", os.path.join(a.out, "alpha", f"{stem}.mp4")], os.path.join(a.out, "logs", "alpha.txt"))


def stage_alpha_mean(a):
    sh([sys.executable, "tools/mean_alpha_csv.py", os.path.join(a.out, "alpha_hcmc_mean.csv"),
        os.path.join(a.out, "alpha", "*_alpha_sweep.csv")], os.path.join(a.out, "logs", "alpha.txt"))


def stage_demo(a, clip, name, kind, warn):
    stem = os.path.splitext(os.path.basename(clip))[0]
    o = os.path.join(a.out, "demo", f"{stem}_{name}{'_warn' if warn else ''}.mp4")
    cmd = [sys.executable, "demo_video.py", "--video", clip, *predictor_args(a, name, kind), "--alpha", "0.35",
           "--out", o, "--stats-json", o.replace(".mp4", ".json")]
    if kind == "autosteer":
        cmd += []
    if warn:
        cmd += ["--warn", "--yolo", a.yolo]
    sh(cmd, os.path.join(a.out, "logs", "demo.txt"))


# ---------------------------------------------------------------- D, E
def labels_hash(a):
    p = os.path.join(a.hcmc, "labels.json")
    if not os.path.exists(p):
        return None, None
    h = hashlib.md5(open(p, "rb").read()).hexdigest()[:10]
    n = sum(1 for line in open(p, encoding="utf-8") if line.strip())
    return h, n


def stage_eval(a, name, kind):
    sh([sys.executable, "evaluate.py", "--custom-json", os.path.join(a.hcmc, "labels.json"), "--custom-root", a.hcmc,
        *predictor_args(a, name, kind), "--out", os.path.join(a.out, "results", f"{name}_hcmc")],
       os.path.join(a.out, "logs", "eval.txt"))


def stage_figures(a):
    cmd = [sys.executable, "tools/make_figures.py", "--out", os.path.join(a.out, "figures")]
    res = glob.glob(os.path.join(a.out, "results", "*_hcmc", "results.json"))
    if res:
        cmd += ["--results", os.path.join(a.out, "results", "*_hcmc", "results.json")]
    if os.path.exists(os.path.join(a.out, "alpha_hcmc_mean.csv")):
        cmd += ["--alpha-csv", os.path.join(a.out, "alpha_hcmc_mean.csv")]
    sh(cmd, os.path.join(a.out, "logs", "figures.txt"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hcmc", required=True, help="thư mục chứa clips/, frames/ và (nếu có) labels.json")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="hcmc_out")
    ap.add_argument("--prev", nargs="*", default=[], help="thư mục kết quả của lần chạy trước (Kaggle: output version cũ)")
    ap.add_argument("--autosteer-repo", default="auto_steer")
    ap.add_argument("--autosteer-weights", default="weights/autosteer.pth")
    ap.add_argument("--yolo", default="yolov8n.pt")
    ap.add_argument("--skip", nargs="*", default=[], help="bỏ các bước: predict alpha demo eval figures")
    a = ap.parse_args()
    a.hcmc, a.runs, a.out = (os.path.abspath(p) for p in (a.hcmc, a.runs, a.out))
    a.autosteer_repo = os.path.abspath(a.autosteer_repo)
    a.autosteer_weights = os.path.abspath(a.autosteer_weights) if a.autosteer_weights else None
    os.makedirs(os.path.join(a.out, "logs"), exist_ok=True)
    for p in a.prev:  # khôi phục kết quả lần trước để không phải chạy lại
        if os.path.isdir(p) and os.path.abspath(p) != a.out:
            shutil.copytree(p, a.out, dirs_exist_ok=True)
            print(f"Khôi phục kết quả từ {p}")
    S = Stages(a.out)
    models = available(a)
    print("Phương pháp dùng được:", [m for m, _ in models])

    if "predict" not in a.skip:
        for name, kind in models:
            S.run(f"predict_{name}", lambda n=name, k=kind: stage_predict(a, n, k))
        S.run("predict_figure", lambda: stage_predict_figure(a))
    if "alpha" not in a.skip and ckpt_of(a.runs, "res18_aug"):
        for c in clips(a):
            S.run(f"alpha_{os.path.basename(c)}", lambda c=c: stage_alpha(a, c))
        S.run(f"alpha_mean_{len(clips(a))}", lambda: stage_alpha_mean(a))
    if "demo" not in a.skip:
        for c in clips(a):
            if ckpt_of(a.runs, "res18_aug"):
                S.run(f"demo_{os.path.basename(c)}_res18_warn", lambda c=c: stage_demo(a, c, "res18_aug", "autopath", True))
            if any(k == "autosteer" for _, k in models):
                S.run(f"demo_{os.path.basename(c)}_autosteer", lambda c=c: stage_demo(a, c, "autosteer", "autosteer", False))
    h, n = labels_hash(a)
    if "eval" not in a.skip and h:
        print(f"Có labels.json: {n} khung, mã băm {h}")
        for name, kind in models:
            S.run(f"eval_{name}_{h}", lambda nm=name, k=kind: stage_eval(a, nm, k))
    elif not h:
        print("Chưa có labels.json: bỏ qua bước đánh giá định lượng.")
    if "figures" not in a.skip:
        S.run(f"figures_{h or 'nolabel'}_{len(clips(a))}", lambda: stage_figures(a))
    summary = {"labels": n, "labels_hash": h, "done": sorted(os.listdir(S.dir))}
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
