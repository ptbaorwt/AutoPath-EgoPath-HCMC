"""Làm mượt đường đi theo thời gian (EMA) và vẽ overlay."""
from __future__ import annotations

import cv2
import numpy as np

from .geometry import ROW_Y


class EMASmoother:
    """x_s(t) = alpha * x_raw(t) + (1 - alpha) * x_s(t-1), áp dụng riêng từng hàng neo.

    Hàng vừa xuất hiện (không có giá trị trước đó) lấy luôn giá trị thô.
    alpha = 1 tương đương không làm mượt.
    """

    def __init__(self, alpha=0.35):
        self.alpha = alpha
        self.state = None
        self.state_valid = None

    def reset(self):
        self.state = self.state_valid = None

    def __call__(self, xp, prob):
        valid = prob >= 0.5
        if self.state is None:
            self.state, self.state_valid = xp.copy(), valid.copy()
            return xp.copy(), prob
        both = valid & self.state_valid
        out = xp.copy()
        a = self.alpha
        out[both] = a * xp[both] + (1 - a) * self.state[both]
        self.state, self.state_valid = out.copy(), valid.copy()
        return out, prob


def temporal_stats(raw_seq, smooth_seq, valid_seq, img_w=1280):
    """Đo độ rung (jitter) và độ trễ cho một chuỗi khung hình.

    jitter: trung bình |x(t) - x(t-1)| (px) trên các hàng hợp lệ ở cả hai khung.
    lag: trung bình |x_smooth(t) - x_raw(t)| (px), đại diện cho độ trễ do làm mượt.
    """
    raw, sm, v = map(np.asarray, (raw_seq, smooth_seq, valid_seq))
    vv = v[1:] & v[:-1]
    jitter = np.abs(np.diff(sm, axis=0))[vv].mean() * img_w if vv.any() else np.nan
    jitter_raw = np.abs(np.diff(raw, axis=0))[vv].mean() * img_w if vv.any() else np.nan
    lag = np.abs(sm - raw)[v].mean() * img_w if v.any() else np.nan
    return dict(jitter_px=float(jitter), jitter_raw_px=float(jitter_raw), lag_px=float(lag))


def draw_path(img_bgr, xp, prob, color=(0, 220, 0), y_offset=0, fill=True, thickness=4, half_width_px=None):
    """Vẽ đường đi lên ảnh đã cắt-resize (hoặc ảnh gốc với y_offset = số px đã cắt).

    fill=True vẽ thêm dải bán trong suốt, bề rộng tăng dần theo phối cảnh.
    """
    h_img, w = img_bgr.shape[:2]
    h = h_img - y_offset
    m = prob >= 0.5
    if m.sum() < 2:
        return img_bgr
    ys = ROW_Y[m] * (h - 1) + y_offset
    xs = xp[m] * w
    pts = np.stack([xs, ys], 1).astype(np.int32)
    out = img_bgr.copy()
    if fill:
        hw0 = half_width_px or w * 0.07
        t = (ys - ys.min()) / max(ys.max() - ys.min(), 1)       # 0 ở xa, 1 ở gần
        hw = hw0 * (0.15 + 0.85 * t)
        poly = np.concatenate([np.stack([xs - hw, ys], 1), np.stack([xs + hw, ys], 1)[::-1]]).astype(np.int32)
        layer = out.copy()
        cv2.fillPoly(layer, [poly], color)
        out = cv2.addWeighted(layer, 0.35, out, 0.65, 0)
    cv2.polylines(out, [pts], False, color, thickness, cv2.LINE_AA)
    return out
