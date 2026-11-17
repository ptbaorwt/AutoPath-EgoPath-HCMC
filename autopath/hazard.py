"""Module phụ (theo gợi ý của GVHD): cảnh báo khi người hoặc phương tiện nằm trên đường đi dự đoán.

Đường đi do AutoPath dự đoán được mở rộng thành một "hành lang" có bề rộng tăng dần theo phối cảnh
(giống dải tô màu khi vẽ). Một đối tượng được coi là nguy hiểm khi điểm giữa cạnh dưới hộp bao
(điểm tiếp đất) nằm trong hành lang và ở nửa dưới của đoạn đường đi (gần xe).

Phát hiện đối tượng dùng YOLOv8n đã huấn luyện trên COCO (thư viện ultralytics). Module này
tách rời khỏi mô hình dự đoán đường đi, chỉ dùng ở chương trình demo (--warn).
"""
from __future__ import annotations

import cv2
import numpy as np

from .geometry import ROW_Y

# Lớp COCO liên quan giao thông: người, xe đạp, ô tô, xe máy, xe buýt, xe tải
COCO_TRAFFIC = {0: "nguoi", 1: "xe dap", 2: "o to", 3: "xe may", 5: "xe buyt", 7: "xe tai"}


def path_corridor(xp, prob, w, h, half_width_ratio=0.07):
    """Trả về đa giác hành lang (N x 2, int32) trên ảnh w x h, hoặc None nếu đường đi quá ngắn."""
    m = prob >= 0.5
    if m.sum() < 2:
        return None
    ys = ROW_Y[m] * (h - 1)
    xs = xp[m] * w
    t = (ys - ys.min()) / max(ys.max() - ys.min(), 1)
    hw = w * half_width_ratio * (0.15 + 0.85 * t)
    return np.concatenate([np.stack([xs - hw, ys], 1), np.stack([xs + hw, ys], 1)[::-1]]).astype(np.int32)


class HazardWarner:
    def __init__(self, weights="yolov8n.pt", conf=0.35, device=None, near_frac=0.5):
        from ultralytics import YOLO  # chỉ cần khi bật --warn
        self.model = YOLO(weights)
        self.conf, self.device, self.near_frac = conf, device, near_frac

    def detect(self, img_bgr):
        r = self.model.predict(img_bgr, conf=self.conf, classes=list(COCO_TRAFFIC), device=self.device, verbose=False)[0]
        return [(*map(float, b.xyxy[0].tolist()), int(b.cls), float(b.conf)) for b in r.boxes]

    def check(self, img_bgr, xp, prob):
        """Trả về (danh sách hộp, danh sách cờ nguy hiểm)."""
        h, w = img_bgr.shape[:2]
        boxes = self.detect(img_bgr)
        poly = path_corridor(xp, prob, w, h)
        flags = []
        for x1, y1, x2, y2, c, s in boxes:
            foot = ((x1 + x2) / 2, y2)
            inside = poly is not None and cv2.pointPolygonTest(poly.reshape(-1, 1, 2).astype(np.float32), foot, False) >= 0
            ys = poly[:, 1] if poly is not None else np.array([0, h])
            near = y2 >= ys.min() + (1 - self.near_frac) * (ys.max() - ys.min())
            flags.append(bool(inside and near))
        return boxes, flags

    @staticmethod
    def draw(img_bgr, boxes, flags):
        out = img_bgr
        for (x1, y1, x2, y2, c, s), f in zip(boxes, flags):
            col = (0, 0, 255) if f else (200, 200, 200)
            cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), col, 3 if f else 1)
            cv2.putText(out, COCO_TRAFFIC.get(c, str(c)), (int(x1), int(y1) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        if any(flags):
            cv2.putText(out, "CANH BAO: VAT CAN TREN DUONG DI", (12, out.shape[0] - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3, cv2.LINE_AA)
        return out
