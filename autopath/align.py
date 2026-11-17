"""Căn hình học camera (test-time camera alignment).

Video khác camera với TuSimple có điểm tụ (vanishing point) ở vị trí khác. Trước khi đưa vào mô hình,
ảnh được dịch và co giãn để điểm tụ (vx, vy) của camera rơi vào vị trí điểm tụ "chuẩn" (640, vy_ref)
mà mô hình đã quen khi huấn luyện; dự đoán được ánh xạ ngược về ảnh gốc và lấy mẫu lại theo 64 hàng neo.
"""
import cv2
import numpy as np

W, H, CROP_TOP, CROP_H, N = 1280, 720, 80, 640, 64
ROW_Y_PX = CROP_TOP + np.linspace(0, 1, N) * CROP_H


class AlignedPredictor:
    def __init__(self, base, vp, vy_ref=270.0, scale=1.0):
        self.base, self.vp, self.vy_ref, self.s = base, vp, vy_ref, scale
        self.name = base.name + "+align"

    def _M(self, vx, vy):
        s = self.s
        return np.float32([[s, 0, W / 2 - s * vx], [0, s, self.vy_ref - s * vy]])

    def predict(self, frames, vps=None):
        warped, Ms = [], []
        for i, f in enumerate(frames):
            f = cv2.resize(f, (W, H)) if f.shape[:2] != (H, W) else f
            vx, vy = (vps[i] if vps is not None else self.vp)
            M = self._M(vx, vy); Ms.append((vx, vy))
            warped.append(cv2.warpAffine(f, M, (W, H), borderMode=cv2.BORDER_REPLICATE))
        xp, pr = self.base.predict(warped)
        XP, PR = np.zeros_like(xp), np.zeros_like(pr)
        for i, (vx, vy) in enumerate(Ms):
            y_orig = (ROW_Y_PX - self.vy_ref) / self.s + vy          # tăng dần
            x_orig = (xp[i] * W - W / 2) / self.s + vx
            inside = (ROW_Y_PX >= y_orig.min()) & (ROW_Y_PX <= y_orig.max())
            XP[i] = np.interp(ROW_Y_PX, y_orig, x_orig) / W
            PR[i] = np.where(inside, np.interp(ROW_Y_PX, y_orig, pr[i]), 0.0)
        return XP, PR
