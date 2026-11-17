"""Đánh giá một phương pháp trên tập test và xuất số liệu cho Chương 4.

Ví dụ:
  python evaluate.py --data-root $TUS --method autopath --ckpt runs/res18_aug/best.pt
  python evaluate.py --data-root $TUS --method autosteer --autosteer-repo auto_steer --autosteer-weights weights/autosteer.pth
  python evaluate.py --data-root $TUS --method classical
  # Tập tự gán nhãn (TP.HCM), định dạng giống TuSimple:
  python evaluate.py --custom-json hcmc/labels.json --custom-root hcmc --method autopath --ckpt ...

Kết quả: <out>/results.json (tổng hợp + theo nhóm độ cong), per_image.csv, và ảnh minh họa
những trường hợp sai nhiều nhất trong <out>/worst/ để làm phần phân tích lỗi.
"""
import argparse
import csv
import json
import os
import time

import cv2
import numpy as np

from autopath.data import build_records, crop_resize, load_or_build
from autopath.metrics import per_sample, summarize
from autopath.predictors import build_predictor
from autopath.smoothing import draw_path

CURVE_BINS = [("thẳng (<20px)", 0, 20), ("cong vừa (20-60px)", 20, 60), ("cong gắt (>60px)", 60, 1e9)]


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True, choices=["autopath", "autosteer", "classical"])
    p.add_argument("--data-root")
    p.add_argument("--split", default="test", choices=["test", "val"])
    p.add_argument("--cache-dir", default="cache")
    p.add_argument("--custom-json", help="file nhãn kiểu TuSimple của tập tự thu thập")
    p.add_argument("--custom-root", help="thư mục gốc chứa ảnh của tập tự thu thập")
    p.add_argument("--ckpt")
    p.add_argument("--align-vp-json", help="JSON {tên video: [vx, vy]}: căn góc camera theo từng video (autopath/align.py)")
    p.add_argument("--align-ref", type=float, nargs=2, default=[260, 0.9])
    p.add_argument("--autosteer-repo", default="auto_steer")
    p.add_argument("--autosteer-weights")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--out", default=None)
    p.add_argument("--n-vis", type=int, default=12)
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def main():
    a = parse()
    if a.custom_json:
        recs, stats = build_records(a.custom_root or os.path.dirname(a.custom_json), [a.custom_json])
        set_name = "custom"
    else:
        d = load_or_build(a.data_root, a.split, a.cache_dir)
        recs, stats, set_name = d["records"], d["stats"], a.split
    if a.limit:
        recs = recs[: a.limit]
    pred = build_predictor(a.method, a.ckpt, a.autosteer_repo, a.autosteer_weights)
    vps = None
    if a.align_vp_json:
        from autopath.align import AlignedPredictor
        VP = json.load(open(a.align_vp_json))
        pred = AlignedPredictor(pred, None, *a.align_ref)
        vps = [next(v for k, v in VP.items() if k in r["image"]) for r in recs]
    out = a.out or os.path.join("results", f"{pred.name}_{set_name}")
    os.makedirs(os.path.join(out, "worst"), exist_ok=True)
    print(f"{pred.name} trên {set_name}: {len(recs)} ảnh")

    P, H = [], []
    t_model = 0.0
    for i in range(0, len(recs), a.batch):
        frames = [cv2.imread(r["image"]) for r in recs[i:i + a.batch]]
        t0 = time.perf_counter()
        xp, pr = pred.predict(frames, vps[i:i + a.batch]) if vps else pred.predict(frames)
        t_model += time.perf_counter() - t0
        P.append(xp); H.append(pr)
    P, H = np.concatenate(P), np.concatenate(H)
    G = np.stack([r["xp"] for r in recs]); V = np.stack([r["valid"] for r in recs])
    ps = per_sample(P, H, G, V)
    res = {"method": pred.name, "set": set_name, "label_stats": stats, "overall": summarize(ps),
           "throughput_fps_incl_preprocess": len(recs) / max(t_model, 1e-9), "by_curvature": {}}
    curv = np.array([r["curvature"] for r in recs])
    for lab, lo, hi in CURVE_BINS:
        m = (curv >= lo) & (curv < hi)
        if m.any():
            res["by_curvature"][lab] = summarize({k: v[m] for k, v in ps.items()})
    tags = sorted({t for r in recs for t in r.get("tags", [])})
    if tags:  # tập tự gán nhãn có thể gắn thẻ điều kiện: night, no_marking, intersection, ...
        res["by_tag"] = {}
        for t in tags:
            m = np.array([t in r.get("tags", []) for r in recs])
            res["by_tag"][t] = summarize({k: v[m] for k, v in ps.items()})
    with open(os.path.join(out, "results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out, "per_image.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["raw_file", "dist", "lat_err_px", "row_iou", "conf", "curvature_px", "tags", "dist_lab"])
        for r, d, l, iou, c, dl in zip(recs, ps["dist"], ps["lat_err_px"], ps["row_iou"], ps["conf"], ps["dist_lab"]):
            w.writerow([r["raw_file"], f"{d:.5f}", f"{l:.2f}", f"{iou:.3f}", f"{c:.3f}", f"{r['curvature']:.1f}",
                        ";".join(r.get("tags", [])), f"{dl:.5f}"])
    np.savez_compressed(os.path.join(out, "preds.npz"), P=P, H=H, G=G, V=V,
                        files=np.array([r["raw_file"] for r in recs]))
    # Ảnh minh họa các trường hợp sai nhiều nhất: xanh = nhãn, đỏ = dự đoán
    for rank, i in enumerate(np.argsort(-ps["dist"])[: a.n_vis]):
        img = crop_resize(cv2.imread(recs[i]["image"]), 1024, 512)
        img = draw_path(img, G[i], V[i], (0, 200, 0), fill=False, thickness=3)
        img = draw_path(img, P[i], H[i], (0, 0, 255), fill=False, thickness=3)
        cv2.putText(img, f"d={ps['dist'][i]:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.imwrite(os.path.join(out, "worst", f"{rank:02d}_{os.path.basename(os.path.dirname(recs[i]['image']))}.jpg"), img)
    o = res["overall"]
    print(f"mAP@50={o['mAP@50']:.4f} mAP={o['mAP']:.4f} acc@0.05={o['acc@0.05']:.4f} "
          f"lat={o['lat_err_px_mean']:.2f}px "
          f"mAP@50_lab={o.get('mAP@50_lab', float('nan')):.4f} -> {out}")


if __name__ == "__main__":
    main()
