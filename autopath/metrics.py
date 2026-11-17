"""Hàm mất mát và chỉ số đánh giá.

mAP@50 được tính đúng theo mã nguồn AutoSteer 2.0 (training/auto_steer_trainer.py,
training/auto_steer_util.py::compute_vector_ap) để hai mô hình so sánh được:
  * Mỗi ảnh có khoảng cách d = trung bình trên 64 hàng của |xp_pred*m_pred - xp_gt*m_gt|
    (đơn vị: tỉ lệ bề rộng ảnh), m là mặt nạ hợp lệ (ngưỡng 0.5).
  * Dự đoán là đúng (TP) nếu d < t. Độ tin cậy = trung bình xác suất hợp lệ dự đoán.
  * AP tại ngưỡng t tính bằng nội suy 101 điểm trên đường Precision-Recall.
  * "mAP@50" của AutoSteer là AP tại t = 0.05 (5% bề rộng ảnh, ~64 px trên ảnh 1280),
    "mAP" là trung bình AP tại t in {0.02, 0.05, 0.10}.
Ngoài ra báo cáo thêm sai số ngang (px trên ảnh gốc 1280 px) để dễ diễn giải.
"""
from __future__ import annotations

import warnings

import numpy as np
import torch
import torch.nn.functional as F

from .geometry import ORIG_W

THRESHOLDS = (0.02, 0.05, 0.10)


def path_loss(xp, h_logit, gt_xp, gt_valid, kind="smoothl1", w_xp=10.0, beta=0.02):
    """Mất mát hồi quy hoành độ (chỉ trên hàng hợp lệ) + BCE cho cờ hợp lệ."""
    m = gt_valid
    denom = m.sum().clamp(min=1.0)
    if kind == "smoothl1":
        l = F.smooth_l1_loss(xp, gt_xp, reduction="none", beta=beta)
    elif kind == "l1":
        l = (xp - gt_xp).abs()
    elif kind == "mse":
        l = (xp - gt_xp) ** 2
    else:
        raise ValueError(kind)
    loss_xp = (l * m).sum() / denom
    loss_h = F.binary_cross_entropy_with_logits(h_logit, gt_valid)
    return w_xp * loss_xp + loss_h, loss_xp.detach(), loss_h.detach()


def per_sample(pred_xp, pred_prob, gt_xp, gt_valid, img_w=ORIG_W):
    """Các đại lượng cho từng ảnh (mảng numpy (N,64))."""
    pm = (pred_prob >= 0.5).astype(np.float32)
    gm = (gt_valid >= 0.5).astype(np.float32)
    d = np.abs(pred_xp * pm - gt_xp * gm).mean(axis=1)              # khoảng cách kiểu AutoSteer
    both = (pm * gm) > 0
    lat = np.where(both, np.abs(pred_xp - gt_xp) * img_w, np.nan)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lat_mean = np.nanmean(lat, axis=1)                           # sai số ngang trung bình (px)
    inter, union = both.sum(1), ((pm + gm) > 0).sum(1)
    row_iou = np.where(union > 0, inter / np.maximum(union, 1), 0.0)  # độ trùng khớp đoạn hợp lệ
    conf = pred_prob.mean(axis=1)
    # Khoảng cách chỉ trên các hàng có nhãn (bỏ qua hàng nhãn bị cắt do che khuất/không nhìn rõ).
    # Dùng cho tập tự gán nhãn, nơi người gán dừng vạch ở xe phía trước: hàng sau điểm dừng là
    # "không biết", không phải "không có đường". Hàng có nhãn mà mô hình báo không hợp lệ vẫn bị phạt |x|.
    gsum = gm.sum(axis=1)
    d_lab = np.where(gsum > 0, (np.abs(pred_xp * pm - gt_xp) * gm).sum(axis=1) / np.maximum(gsum, 1), np.nan)
    return dict(dist=d, lat_err_px=lat_mean, row_iou=row_iou, conf=conf, dist_lab=d_lab)


def average_precision(dist, conf, t):
    """AP tại một ngưỡng, nội suy 101 điểm (giống compute_vector_ap của AutoSteer)."""
    dist, conf = np.asarray(dist), np.asarray(conf)
    order = np.argsort(-conf)
    tp = (dist[order] < t).astype(np.float64)
    fp = 1.0 - tp
    tpc, fpc = tp.cumsum(), fp.cumsum()
    precision = tpc / (tpc + fpc + 1e-16)
    recall = tpc / (len(dist) + 1e-16)
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    trap = getattr(np, "trapezoid", None) or np.trapz
    return float(trap(np.interp(x, mrec, mpre), x))


def summarize(ps: dict) -> dict:
    d, c = ps["dist"], ps["conf"]
    ap = {t: average_precision(d, c, t) for t in THRESHOLDS}
    lat = ps["lat_err_px"]
    return {
        "n": int(len(d)),
        "mAP@50": ap[0.05],
        "mAP": float(np.mean(list(ap.values()))),
        "AP@0.02": ap[0.02], "AP@0.10": ap[0.10],
        "acc@0.05": float((d < 0.05).mean()) if len(d) else float("nan"),
        "mean_dist": float(np.mean(d)) if len(d) else float("nan"),
        "lat_err_px_mean": float(np.nanmean(lat)) if np.isfinite(lat).any() else float("nan"),
        "lat_err_px_median": float(np.nanmedian(lat)) if np.isfinite(lat).any() else float("nan"),
        "row_iou": float(np.mean(ps["row_iou"])) if len(d) else float("nan"),
        **({"mAP@50_lab": average_precision(ps["dist_lab"], c, 0.05),
            "acc@0.05_lab": float((ps["dist_lab"] < 0.05).mean()),
            "AP@0.02_lab": average_precision(ps["dist_lab"], c, 0.02),
            "AP@0.10_lab": average_precision(ps["dist_lab"], c, 0.10)} if "dist_lab" in ps and len(d) else {}),
    }


@torch.no_grad()
def collect_predictions(model, loader, device):
    """Chạy mô hình AutoPath trên DataLoader, trả về mảng numpy dự đoán và nhãn."""
    model.eval()
    P, H, G, V, I = [], [], [], [], []
    for x, gx, gv, idx in loader:
        xp, hl = model(x.to(device, non_blocking=True))
        P.append(xp.float().cpu()); H.append(torch.sigmoid(hl.float()).cpu())
        G.append(gx); V.append(gv); I.append(idx)
    cat = lambda a: torch.cat(a).numpy()
    return cat(P), cat(H), cat(G), cat(V), cat(I)
