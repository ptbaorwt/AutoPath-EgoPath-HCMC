"""Tinh chỉnh AutoPath trên khung TP.HCM (nhãn giả từ AutoSteer và/hoặc nhãn tay của nhóm).

Thêm tăng cường hình học (dịch, co giãn quanh điểm tụ) để mô hình không phụ thuộc một vị trí camera.
Chọn checkpoint theo tập kiểm định = 15% cuối mỗi video của dữ liệu tinh chỉnh (KHÔNG dùng tập test 187 khung).

  python tools/finetune_hcmc.py --init runs/res18_aug/best.pt --pseudo hcmc/pseudo --name res18_ft_pseudo
  python tools/finetune_hcmc.py --init runs/res18_ft_pseudo/best.pt --manual hcmc/train_labels.json --manual-root hcmc --name res18_ft_manual
"""
import argparse, csv, json, math, os, random, sys, time
import cv2, numpy as np, torch
from torch.utils.data import DataLoader, Dataset
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autopath.data import build_records, crop_resize, to_tensor, build_photometric_aug
from autopath.metrics import path_loss, per_sample, summarize, collect_predictions
from autopath.model import load_checkpoint, save_checkpoint

W, H, N = 1280, 720, 64
ROW_PX = 80 + np.linspace(0, 1, N) * 640


def warp_labels(xp, valid, s, tx, ty):
    """Ảnh biến đổi x' = s*x + tx, y' = s*y + ty (px). Lấy mẫu lại nhãn trên 64 hàng của ảnh mới."""
    m = valid > 0.5
    if m.sum() < 2:
        return np.zeros(N, np.float32), np.zeros(N, np.float32)
    ysrc = ROW_PX[m]; xsrc = xp[m] * W
    ynew = s * ysrc + ty; xnew = s * xsrc + tx
    inside = (ROW_PX >= ynew.min()) & (ROW_PX <= ynew.max())
    x2 = np.interp(ROW_PX, ynew, xnew) / W
    ok = inside & (x2 > 0) & (x2 < 1)
    return np.where(ok, x2, 0).astype(np.float32), ok.astype(np.float32)


class FTDataset(Dataset):
    def __init__(self, recs, w=512, h=256, augment=False):
        self.r, self.w, self.h, self.aug = recs, w, h, augment
        self.photo = build_photometric_aug() if augment else None

    def __len__(self): return len(self.r)

    def __getitem__(self, i):
        r = self.r[i]
        img = cv2.imread(r["image"]); img = cv2.resize(img, (W, H)) if img.shape[:2] != (H, W) else img
        xp, v = r["xp"].astype(np.float32).copy(), r["valid"].astype(np.float32).copy()
        if self.aug:
            s = random.uniform(0.85, 1.15); cx, cy = W / 2, random.uniform(300, 480)
            tx = (1 - s) * cx + random.uniform(-60, 60); ty = (1 - s) * cy + random.uniform(-90, 90)
            img = cv2.warpAffine(img, np.float32([[s, 0, tx], [0, s, ty]]), (W, H), borderMode=cv2.BORDER_REPLICATE)
            xp, v = warp_labels(xp, v, s, tx, ty)
        img = cv2.cvtColor(crop_resize(img, self.w, self.h), cv2.COLOR_BGR2RGB)
        if self.aug:
            img = self.photo(image=img)["image"]
            if random.random() < 0.5:
                img = img[:, ::-1]; xp = np.where(v > 0.5, 1 - xp, 0).astype(np.float32)
        return to_tensor(img), torch.from_numpy(xp), torch.from_numpy(v), i


def load_pseudo(d):
    z = np.load(os.path.join(d, "pseudo.npz"))
    recs = []
    for f, x, p, k in zip(z["files"], z["xp"], z["prob"], z["keep"]):
        if k:
            recs.append({"image": os.path.join(d, "frames", str(f)), "xp": x, "valid": (p >= 0.5).astype(np.float32),
                         "video": str(f).split("_")[0], "src": "pseudo"})
    return recs


def split_val(recs, frac=0.15):
    tr, va = [], []
    by = {}
    for r in recs: by.setdefault(r["video"], []).append(r)
    for v, rs in by.items():
        k = int(len(rs) * (1 - frac)); tr += rs[:k]; va += rs[k:]
    return tr, va


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--pseudo", default="")
    ap.add_argument("--manual", default="", help="nhãn tay kiểu TuSimple (KHÔNG được chứa khung test)")
    ap.add_argument("--manual-root", default="hcmc")
    ap.add_argument("--manual-repeat", type=int, default=3, help="lặp nhãn tay để cân bằng với nhãn giả")
    ap.add_argument("--name", default="res18_ft")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(0); random.seed(0); np.random.seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    recs = load_pseudo(a.pseudo) if a.pseudo else []
    if a.manual:
        man, _ = build_records(a.manual_root, [a.manual])
        for r in man: r["video"] = os.path.basename(r["image"]).split("_")[0]; r["src"] = "manual"
        recs += man * a.manual_repeat if not a.pseudo else man * a.manual_repeat
    if a.limit: recs = recs[: a.limit]
    tr, va = split_val(recs)
    print(f"train={len(tr)} val={len(va)} thiết bị={dev}", flush=True)
    model = load_checkpoint(a.init, dev); model.train()
    w, h = model.cfg["input_w"], model.cfg["input_h"]
    dl = DataLoader(FTDataset(tr, w, h, True), a.batch, shuffle=True, num_workers=a.workers, drop_last=len(tr) > a.batch)
    dv = DataLoader(FTDataset(va, w, h, False), a.batch, shuffle=False, num_workers=a.workers)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * max(1, len(dl))
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / steps)))
    amp = dev == "cuda"; scaler = torch.amp.GradScaler("cuda", enabled=amp)
    out = os.path.join("runs", a.name); os.makedirs(out, exist_ok=True)
    hist = csv.writer(open(os.path.join(out, "history.csv"), "w", newline=""))
    hist.writerow(["epoch", "train_loss", "val_mAP50", "val_lat_px", "sec"])
    best = -1
    for ep in range(1, a.epochs + 1):
        t0 = time.time(); model.train(); tl = n = 0
        for x, gx, gv, _ in dl:
            x, gx, gv = x.to(dev), gx.to(dev), gv.to(dev)
            with torch.autocast(device_type="cuda", enabled=amp):
                xp, hl = model(x)
            loss, _, _ = path_loss(xp.float(), hl.float(), gx, gv)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sch.step()
            tl += loss.item() * x.shape[0]; n += x.shape[0]
        P, Hh, G, V, _ = collect_predictions(model, dv, dev); s = summarize(per_sample(P, Hh, G, V))
        hist.writerow([ep, tl / n, s["mAP@50"], s["lat_err_px_mean"], round(time.time() - t0)])
        print(f"ep {ep} loss {tl / n:.4f} | val mAP@50 {s['mAP@50']:.3f} lat {s['lat_err_px_mean']:.1f}px ({time.time() - t0:.0f}s)", flush=True)
        score = s["mAP@50"] - s["lat_err_px_mean"] / 1000
        if score > best:
            best = score; save_checkpoint(os.path.join(out, "best.pt"), model, {"epoch": ep, "val": s, "init": a.init})
    json.dump(vars(a) | {"n_train": len(tr), "n_val": len(va)}, open(os.path.join(out, "config.json"), "w"), indent=2)
    print("Xong:", os.path.join(out, "best.pt"))


if __name__ == "__main__":
    main()
