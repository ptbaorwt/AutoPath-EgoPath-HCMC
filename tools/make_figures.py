"""Sinh hình và bảng cho Chương 4 từ kết quả chạy thật.

  python tools/make_figures.py --runs runs/* --results results/*/results.json \
      --benchmark results/benchmark.json --alpha-csv results/demo_alpha_sweep.csv --out figures
Đầu ra: figures/*.png (đường cong huấn luyện, đánh đổi độ chính xác - FPS, sai số theo độ cong,
jitter-lag theo alpha) và figures/tables.md (bảng số liệu để chép vào báo cáo).
"""
import argparse
import csv
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def read_hist(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {k: [float(r[k]) for r in rows] for k in rows[0]} if rows else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--results", nargs="*", default=[])
    ap.add_argument("--benchmark")
    ap.add_argument("--alpha-csv")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    md = []

    runs = [r for pat in a.runs for r in glob.glob(pat) if os.path.exists(os.path.join(r, "history.csv"))]
    if runs:
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
        for r in runs:
            h = read_hist(os.path.join(r, "history.csv"))
            n = os.path.basename(r)
            ax[0].plot(h["epoch"], h["train_loss"], label=n)
            ax[1].plot(h["epoch"], h["val_mAP50"], label=n)
        ax[0].set(xlabel="Epoch", ylabel="Loss huấn luyện"); ax[1].set(xlabel="Epoch", ylabel="mAP@50 (validation)")
        ax[1].legend(frameon=False, fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(a.out, "training_curves.png"), dpi=200); plt.close(fig)

    res = []
    for pat in a.results:
        for p in sorted(glob.glob(pat)):
            r = json.load(open(p, encoding="utf-8"))
            run = os.path.basename(os.path.dirname(p))  # tên thư mục kết quả, ví dụ res18_noaug
            r["method_name"] = r["method"]
            if run and run.lower() not in r["method"].lower():
                r["method"] = f"{r['method']} [{run}]"  # tránh hai lần chạy cùng backbone trùng tên
            res.append(r)
    if res:
        md += ["| Phương pháp | Tập | n | mAP@50 | mAP | Acc@0.05 | Sai số ngang TB (px) | Trung vị (px) | IoU hàng |",
               "|---|---|---|---|---|---|---|---|---|"]
        for r in res:
            o = r["overall"]
            md.append(f"| {r['method']} | {r['set']} | {o['n']} | {o['mAP@50']:.4f} | {o['mAP']:.4f} | {o['acc@0.05']:.4f} | "
                      f"{o['lat_err_px_mean']:.2f} | {o['lat_err_px_median']:.2f} | {o['row_iou']:.3f} |")
        md += ["", "| Phương pháp | Nhóm độ cong | n | mAP@50 | Sai số ngang TB (px) |", "|---|---|---|---|---|"]
        if any(r.get("by_tag") for r in res):
            md += ["", "| Phương pháp | Điều kiện (thẻ) | Số khung | mAP@50 | Sai số ngang TB (px) | Tỉ lệ khung sai (d ≥ 0,05) |",
                   "|---|---|---|---|---|---|"]
            for r in res:
                for t, s in r.get("by_tag", {}).items():
                    md.append(f"| {r['method']} | {t} | {s['n']} | {s['mAP@50']:.4f} | {s['lat_err_px_mean']:.2f} | {1 - s['acc@0.05']:.3f} |")
        bins = sorted({b for r in res for b in r.get("by_curvature", {})})
        for r in res:
            for b, s in r.get("by_curvature", {}).items():
                md.append(f"| {r['method']} ({r['set']}) | {b} | {s['n']} | {s['mAP@50']:.4f} | {s['lat_err_px_mean']:.2f} |")
        if bins:
            fig, ax = plt.subplots(figsize=(7, 3.6))
            width = 0.8 / max(1, len(res))
            for k, r in enumerate(res):
                vals = [r["by_curvature"].get(b, {}).get("lat_err_px_mean", 0) for b in bins]
                ax.bar([i + k * width for i in range(len(bins))], vals, width, label=f"{r['method']} ({r['set']})")
            ax.set_xticks([i + width * (len(res) - 1) / 2 for i in range(len(bins))], bins)
            ax.set_ylabel("Sai số ngang trung bình (px)"); ax.legend(frameon=False, fontsize=8)
            fig.tight_layout(); fig.savefig(os.path.join(a.out, "error_by_curvature.png"), dpi=200); plt.close(fig)

    if a.benchmark:
        b = json.load(open(a.benchmark, encoding="utf-8"))
        md += ["", f"Thiết bị đo: {b['device']}", "", "| Phương pháp | Đầu vào | Tham số (M) | Dung lượng (MB) | FPS | Độ trễ TB (ms) |",
               "|---|---|---|---|---|---|"]
        for r in b["results"]:
            md.append(f"| {r['method']} | {r['input'][0]}x{r['input'][1]} | {r['params_M']:.2f} | {r['size_MB_fp32']:.1f} | "
                      f"{r['fps']:.1f} | {r['latency_ms_mean']:.2f} |")
        if res:
            acc = {r["method_name"]: r["overall"]["mAP@50"] for r in res if r["set"] == "test" and "noaug" not in r["method"]}
            pts = [(r["fps"], acc[m], m) for r in b["results"] for m in acc if r["method"].startswith(m)]
            if pts:
                fig, ax = plt.subplots(figsize=(6, 4))
                for x, y, m in pts:
                    ax.scatter(x, y, s=50); ax.annotate(m, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
                ax.set(xlabel="FPS", ylabel="mAP@50 (test)")
                fig.tight_layout(); fig.savefig(os.path.join(a.out, "accuracy_vs_fps.png"), dpi=200); plt.close(fig)

    if a.alpha_csv:
        with open(a.alpha_csv) as f:
            rows = list(csv.DictReader(f))
        al = [float(r["alpha"]) for r in rows]
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.plot(al, [float(r["jitter_px"]) for r in rows], "o-", label="Độ rung (jitter)")
        ax.plot(al, [float(r["lag_px"]) for r in rows], "s--", label="Độ trễ (lag)")
        ax.set(xlabel="Hệ số alpha", ylabel="px"); ax.legend(frameon=False)
        fig.tight_layout(); fig.savefig(os.path.join(a.out, "ema_alpha_tradeoff.png"), dpi=200); plt.close(fig)
        md += ["", "| alpha | Jitter (px) | Jitter thô (px) | Lag (px) |", "|---|---|---|---|"]
        md += [f"| {r['alpha']} | {r['jitter_px']} | {r['jitter_raw_px']} | {r['lag_px']} |" for r in rows]

    with open(os.path.join(a.out, "tables.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(f"Đã ghi hình và bảng vào {a.out}/")


if __name__ == "__main__":
    main()
