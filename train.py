"""Huấn luyện AutoPath trên TuSimple.

Ví dụ:
  python train.py --data-root /kaggle/input/tusimple/TUSimple --backbone resnet18 --name res18_aug
  python train.py --data-root ... --backbone resnet18 --no-aug --name res18_noaug          # RQ2
  python train.py --data-root ... --backbone mobilenet_v3_small --name mbv3s_aug          # RQ1
"""
import argparse
import csv
import json
import math
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from autopath.data import EgoPathDataset, load_or_build
from autopath.metrics import collect_predictions, path_loss, per_sample, summarize
from autopath.model import AutoPathNet, count_params, save_checkpoint


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--cache-dir", default="cache")
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--name", default=None)
    p.add_argument("--backbone", default="resnet18",
                   choices=["resnet18", "resnet34", "mobilenet_v3_small", "mobilenet_v3_large"])
    p.add_argument("--head", default="spatial", choices=["spatial", "gap"])
    p.add_argument("--input-w", type=int, default=512)
    p.add_argument("--input-h", type=int, default=256)
    p.add_argument("--no-aug", action="store_true")
    p.add_argument("--loss", default="smoothl1", choices=["smoothl1", "l1", "mse"])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="chỉ dùng N ảnh (để chạy thử nhanh)")
    p.add_argument("--resume", action="store_true",
                   help="tiếp tục từ runs/<name>/last.pt nếu có (Kaggle bị ngắt giữa chừng); bỏ qua nếu lần chạy đã xong")
    p.add_argument("--wandb", default="", help="tên project W&B (để trống = không dùng)")
    return p.parse_args()


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def main():
    a = parse()
    seed_all(a.seed)
    name = a.name or f"{a.backbone}_{'noaug' if a.no_aug else 'aug'}_{a.loss}"
    out = os.path.join(a.out_dir, name)
    os.makedirs(out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tr = load_or_build(a.data_root, "train", a.cache_dir)
    va = load_or_build(a.data_root, "val", a.cache_dir)
    tr_rec, va_rec = tr["records"], va["records"]
    if a.limit:
        tr_rec, va_rec = tr_rec[: a.limit], va_rec[: max(8, a.limit // 5)]
    print(f"train={len(tr_rec)} val={len(va_rec)} | thống kê nhãn: {tr['stats']}")

    ds_tr = EgoPathDataset(tr_rec, a.input_w, a.input_h, augment=not a.no_aug)
    ds_va = EgoPathDataset(va_rec, a.input_w, a.input_h, augment=False)
    dl_tr = DataLoader(ds_tr, a.batch, shuffle=True, num_workers=a.workers, pin_memory=dev == "cuda", drop_last=len(ds_tr) > a.batch)
    dl_va = DataLoader(ds_va, a.batch, shuffle=False, num_workers=a.workers, pin_memory=dev == "cuda")

    model = AutoPathNet(a.backbone, not a.no_pretrained, a.input_w, a.input_h, a.head).to(dev)
    print(f"{name}: {count_params(model) / 1e6:.2f}M tham số, thiết bị={dev}")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    steps = a.epochs * max(1, len(dl_tr))
    warm = min(500, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    use_amp = dev == "cuda" and not a.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_ep, best, bad = 1, (-1.0, 1e9), 0
    last_p, done_p = os.path.join(out, "last.pt"), os.path.join(out, "DONE")
    if a.resume and os.path.exists(done_p):
        print(f"{name}: đã huấn luyện xong trước đó ({done_p}), bỏ qua.")
        return
    if a.resume and os.path.exists(last_p):
        ck = torch.load(last_p, map_location=dev, weights_only=False)
        st = ck.get("extra", {}).get("resume")
        if st:
            model.load_state_dict(ck["state_dict"])
            opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"]); scaler.load_state_dict(st["scaler"])
            start_ep, best, bad = ck["extra"]["epoch"] + 1, tuple(st["best"]), st["bad"]
            print(f"{name}: tiếp tục từ epoch {start_ep} (best mAP@50 = {best[0]:.4f})")

    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(vars(a) | {"n_train": len(tr_rec), "n_val": len(va_rec), "params": count_params(model)}, f, indent=2)
    wb = None
    if a.wandb:
        import wandb
        wb = wandb.init(project=a.wandb, name=name, config=vars(a))

    resumed = start_ep > 1
    hist_f = open(os.path.join(out, "history.csv"), "a" if resumed else "w", newline="")
    hist = csv.writer(hist_f)
    if not resumed:
      hist.writerow(["epoch", "lr", "train_loss", "train_loss_xp", "train_loss_h", "val_mAP50", "val_mAP",
                   "val_acc005", "val_lat_px", "val_row_iou", "sec"])
    for ep in range(start_ep, a.epochs + 1):
        t0 = time.time()
        model.train()
        tl = tx = th = 0.0
        n = 0
        for x, gx, gv, _ in dl_tr:
            x, gx, gv = x.to(dev, non_blocking=True), gx.to(dev), gv.to(dev)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                xp, hl = model(x)
            loss, lx, lh = path_loss(xp.float(), hl.float(), gx, gv, kind=a.loss)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(opt); scaler.update(); sched.step()
            b = x.shape[0]; n += b
            tl += loss.item() * b; tx += lx.item() * b; th += lh.item() * b
        P, H, G, V, _ = collect_predictions(model, dl_va, dev)
        s = summarize(per_sample(P, H, G, V))
        row = [ep, opt.param_groups[0]["lr"], tl / n, tx / n, th / n, s["mAP@50"], s["mAP"], s["acc@0.05"],
               s["lat_err_px_mean"], s["row_iou"], round(time.time() - t0, 1)]
        hist.writerow(row); hist_f.flush()
        print(f"ep {ep:3d} loss {tl / n:.4f} | val mAP@50 {s['mAP@50']:.4f} mAP {s['mAP']:.4f} "
              f"lat {s['lat_err_px_mean']:.2f}px ({row[-1]}s)")
        if wb:
            wb.log(dict(zip(["epoch", "lr", "train/loss", "train/loss_xp", "train/loss_h", "val/mAP50", "val/mAP",
                             "val/acc005", "val/lat_px", "val/row_iou"], row[:-1])))
        key = (s["mAP@50"], -s["lat_err_px_mean"] if np.isfinite(s["lat_err_px_mean"]) else -1e9)
        if key > best:
            best, bad = key, 0
            save_checkpoint(os.path.join(out, "best.pt"), model, {"epoch": ep, "val": s})
        else:
            bad += 1
        # last.pt lưu đủ trạng thái (optimizer, lịch lr, AMP, best) để --resume chạy tiếp đúng chỗ
        save_checkpoint(last_p, model, {"epoch": ep, "val": s, "resume": {
            "opt": opt.state_dict(), "sched": sched.state_dict(), "scaler": scaler.state_dict(),
            "best": list(best), "bad": bad}})
        if bad >= a.patience:
            print(f"Dừng sớm ở epoch {ep} (không cải thiện sau {a.patience} epoch)")
            break
    hist_f.close()
    open(done_p, "w").write("ok")
    if wb:
        wb.finish()
    print(f"Xong. Checkpoint tốt nhất: {os.path.join(out, 'best.pt')}")


if __name__ == "__main__":
    main()
