"""Biểu diễn đường đi (ego-path) và chuyển nhãn TuSimple sang ego-path.

Quy ước dùng chung cho toàn dự án (trùng với AutoSteer 2.0 để so sánh được):
  * Ảnh gốc TuSimple 1280x720, cắt bỏ 80 px phía trên (trời) -> 1280x640.
  * Đường đi được biểu diễn bằng N_ROWS = 64 hàng neo (row anchors) cách đều
    theo trục dọc của ảnh đã cắt. Hàng i có tọa độ dọc chuẩn hóa ROW_Y[i] in [0, 1]
    (0 = mép trên vùng cắt, 1 = mép dưới ảnh).
  * Mỗi hàng có hoành độ chuẩn hóa xp[i] = x_px / 1280 in [0, 1] và cờ hợp lệ
    valid[i] in {0, 1} (hàng nằm trong đoạn đường nhìn thấy được).
"""
from __future__ import annotations

import numpy as np

ORIG_W, ORIG_H = 1280, 720
CROP_TOP = 80
CROP_H = ORIG_H - CROP_TOP          # 640
N_ROWS = 64
ROW_Y = np.linspace(0.0, 1.0, N_ROWS)  # tọa độ dọc chuẩn hóa của 64 hàng neo


def y_px_to_norm(y_px: np.ndarray, crop_top: float = CROP_TOP, crop_h: float = CROP_H) -> np.ndarray:
    """Tọa độ dọc (px, ảnh gốc) -> tọa độ dọc chuẩn hóa trong vùng đã cắt."""
    return (np.asarray(y_px, dtype=np.float64) - crop_top) / (crop_h - 1)


def row_y_px(h_out: int) -> np.ndarray:
    """Tọa độ dọc (px) của 64 hàng neo trên ảnh đã cắt có chiều cao h_out."""
    return ROW_Y * (h_out - 1)


def lanes_from_tusimple(entry: dict) -> list[np.ndarray]:
    """Tách các làn trong một dòng nhãn TuSimple thành mảng điểm (x, y) px.

    TuSimple lưu hoành độ x tại các tung độ cố định h_samples, x = -2 nghĩa là
    không có vạch tại hàng đó. Bỏ các làn có ít hơn 2 điểm.
    """
    hs = np.asarray(entry["h_samples"], dtype=np.float64)
    lanes = []
    for xs in entry["lanes"]:
        xs = np.asarray(xs, dtype=np.float64)
        m = xs >= 0
        if m.sum() >= 2:
            pts = np.stack([xs[m], hs[m]], axis=1)
            lanes.append(pts[np.argsort(pts[:, 1])])  # sắp theo y tăng dần
    return lanes


def bottom_anchor(lane: np.ndarray, y_bottom: float = ORIG_H - 1) -> float:
    """Hoành độ (px) nơi làn cắt mép dưới ảnh, ngoại suy tuyến tính từ 2 điểm thấp nhất."""
    (x1, y1), (x2, y2) = lane[-2], lane[-1]
    if abs(y2 - y1) < 1e-6:
        return float(x2)
    return float(x2 + (y_bottom - y2) * (x2 - x1) / (y2 - y1))


def pick_ego_lanes(lanes: list[np.ndarray], img_w: float = ORIG_W, img_h: float = ORIG_H):
    """Chọn hai vạch bao quanh xe: vạch trái và vạch phải gần tâm nhất.

    Trả về (left, right) hoặc None nếu thiếu một bên.
    """
    if len(lanes) < 2:
        return None
    anchors = [bottom_anchor(l, img_h - 1) for l in lanes]
    cx = img_w / 2
    left = [(a, l) for a, l in zip(anchors, lanes) if a < cx]
    right = [(a, l) for a, l in zip(anchors, lanes) if a >= cx]
    if not left or not right:
        return None
    ego_left = max(left, key=lambda t: t[0])[1]
    ego_right = min(right, key=lambda t: t[0])[1]
    return ego_left, ego_right


def center_line(left: np.ndarray, right: np.ndarray) -> np.ndarray | None:
    """Đường tâm = trung điểm hai vạch ego."""
    y_top = max(left[:, 1].min(), right[:, 1].min())
    y_bot = min(left[:, 1].max(), right[:, 1].max())
    if y_bot - y_top < 1:
        return None
    ys = np.unique(np.concatenate([left[:, 1], right[:, 1]]))
    ys = ys[(ys >= y_top) & (ys <= y_bot)]
    xl = np.interp(ys, left[:, 1], left[:, 0])
    xr = np.interp(ys, right[:, 1], right[:, 0])
    return np.stack([(xl + xr) / 2.0, ys], axis=1)


def sample_rows(points_px: np.ndarray, img_w: float = ORIG_W,
                crop_top: float = CROP_TOP, crop_h: float = CROP_H):
    """Lấy mẫu một đường (x, y px trên ảnh gốc) tại 64 hàng neo.

    Nội suy tuyến tính, không ngoại suy: hàng nằm ngoài đoạn có nhãn được đánh
    dấu không hợp lệ (valid = 0, xp = 0).
    """
    pts = points_px[np.argsort(points_px[:, 1])]
    yn = y_px_to_norm(pts[:, 1], crop_top, crop_h)
    xn = pts[:, 0] / img_w
    xp = np.zeros(N_ROWS, dtype=np.float32)
    valid = (ROW_Y >= yn.min()) & (ROW_Y <= yn.max())
    xp[valid] = np.interp(ROW_Y[valid], yn, xn)
    valid &= (xp >= 0) & (xp <= 1)
    xp[~valid] = 0.0
    return xp.astype(np.float32), valid.astype(np.float32)


def tusimple_to_egopath(entry: dict, img_w: float = ORIG_W, img_h: float = ORIG_H):
    """Một dòng nhãn TuSimple -> (xp, valid) của ego-path, hoặc None nếu không xác định được."""
    lanes = lanes_from_tusimple(entry)
    ego = pick_ego_lanes(lanes, img_w, img_h)
    if ego is None:
        return None
    c = center_line(*ego)
    if c is None:
        return None
    crop_top = CROP_TOP * img_h / ORIG_H
    xp, valid = sample_rows(c, img_w, crop_top, img_h - crop_top)
    if valid.sum() < 4:
        return None
    return xp, valid


def path_curvature_px(xp: np.ndarray, valid: np.ndarray, img_w: float = ORIG_W) -> float:
    """Độ cong gần đúng: độ lệch ngang lớn nhất (px) của đường so với dây cung nối hai đầu."""
    m = valid > 0.5
    if m.sum() < 3:
        return 0.0
    x = xp[m] * img_w
    t = np.linspace(0.0, 1.0, len(x))
    chord = x[0] + (x[-1] - x[0]) * t
    return float(np.abs(x - chord).max())


def to_pixels(xp: np.ndarray, valid: np.ndarray, out_w: int, out_h: int, y_offset: int = 0):
    """Đổi (xp, valid) sang danh sách điểm pixel trên ảnh đầu ra (đã cắt, cỡ out_w x out_h)."""
    ys = row_y_px(out_h) + y_offset
    m = valid > 0.5
    return np.stack([xp[m] * out_w, ys[m]], axis=1)
