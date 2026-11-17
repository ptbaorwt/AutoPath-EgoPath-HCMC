"""Giao diện dự đoán chung cho 3 phương pháp: AutoPath (của nhóm), AutoSteer 2.0 và baseline cổ điển.

Mọi predictor nhận ảnh BGR gốc (bất kỳ kích thước, tỉ lệ 16:9) và trả về
(xp, prob) dạng numpy (64,): hoành độ chuẩn hóa và xác suất hợp lệ của 64 hàng neo.
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np
import torch

from .data import crop_resize, to_tensor
from .geometry import N_ROWS, ROW_Y
from .model import load_checkpoint


class AutoPathPredictor:
    name = "AutoPath"

    def __init__(self, ckpt, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_checkpoint(ckpt, self.device)
        self.w, self.h = self.model.cfg["input_w"], self.model.cfg["input_h"]
        self.name = f"AutoPath-{self.model.cfg['backbone']}"

    def preprocess(self, frames_bgr):
        return torch.stack([to_tensor(cv2.cvtColor(crop_resize(f, self.w, self.h), cv2.COLOR_BGR2RGB))
                            for f in frames_bgr])

    @torch.no_grad()
    def predict_tensor(self, x):
        xp, hl = self.model(x.to(self.device))
        return xp.float().cpu().numpy(), torch.sigmoid(hl.float()).cpu().numpy()

    def predict(self, frames_bgr):
        return self.predict_tensor(self.preprocess(frames_bgr))


class AutoSteerPredictor:
    """Bọc mô hình AutoSteer 2.0 gốc (github.com/autowarefoundation/auto_steer).

    Tiền xử lý đúng như converter TuSimple của họ: cắt 80/720 phía trên, resize 1024x512,
    BGR->RGB, ToTensor (không chuẩn hóa ImageNet). Đầu ra 64 hàng trên lưới linspace(0, 511, 64)
    trùng với lưới hàng neo của AutoPath nên so sánh trực tiếp được.
    """
    name = "AutoSteer-2.0"

    def __init__(self, repo_dir, weights, device=None, version="n"):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        repo_dir = os.path.abspath(repo_dir)
        if repo_dir not in sys.path:
            sys.path.insert(0, repo_dir)
        from Models.model_components.auto_steer.auto_steer_network import AutoSteerNetwork  # noqa
        self.model = AutoSteerNetwork().build_model(version)
        if weights:
            ck = torch.load(weights, map_location="cpu", weights_only=False)
            sd = ck["model"].state_dict() if isinstance(ck, dict) and "model" in ck and hasattr(ck["model"], "state_dict") \
                else ck.get("state_dict", ck) if isinstance(ck, dict) else ck.state_dict()
            self.model.load_state_dict(sd)
        self.model = self.model.to(self.device).eval()

    def preprocess(self, frames_bgr):
        ims = []
        for f in frames_bgr:
            im = cv2.cvtColor(crop_resize(f, 1024, 512), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            ims.append(torch.from_numpy(im.transpose(2, 0, 1).copy()))
        return torch.stack(ims)

    @torch.no_grad()
    def predict_tensor(self, x):
        xp, hv = self.model(x.to(self.device))
        xp = xp.float().reshape(xp.shape[0], -1)[:, :N_ROWS].cpu().numpy()
        hv = hv.float().reshape(hv.shape[0], -1)[:, :N_ROWS].cpu().numpy()
        # AutoSteer huấn luyện h_vector bằng BCEWithLogits nhưng khi suy luận/đánh giá lại so
        # giá trị thô với 0.5 (xem tutorial.ipynb và auto_steer_trainer.py). Để giữ đúng quy tắc
        # đó trong giao diện chung (hợp lệ khi prob >= 0.5), ta dùng prob = sigmoid(raw - 0.5).
        return xp, 1.0 / (1.0 + np.exp(-(hv - 0.5)))

    def predict(self, frames_bgr):
        return self.predict_tensor(self.preprocess(frames_bgr))


class ClassicalPredictor:
    """Baseline cổ điển: Canny + Hough + khớp đường thẳng hai vạch ego, đường đi = trung điểm.

    Không học từ dữ liệu; dùng để đo mức cải thiện của phương pháp học sâu.
    """
    name = "Canny+Hough"

    def __init__(self, work_w=640, work_h=320, roi_top=0.45, canny=(50, 150)):
        self.w, self.h, self.roi_top, self.canny = work_w, work_h, roi_top, canny

    def _one(self, frame):
        img = crop_resize(frame, self.w, self.h)
        gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        edges = cv2.Canny(gray, *self.canny)
        mask = np.zeros_like(edges)
        y0 = int(self.h * self.roi_top)
        poly = np.array([[(0, self.h - 1), (int(self.w * 0.40), y0), (int(self.w * 0.60), y0),
                          (self.w - 1, self.h - 1)]], np.int32)
        cv2.fillPoly(mask, poly, 255)
        lines = cv2.HoughLinesP(edges & mask, 2, np.pi / 180, 40, minLineLength=20, maxLineGap=60)
        xp, prob = np.zeros(N_ROWS, np.float32), np.zeros(N_ROWS, np.float32)
        if lines is None:
            return xp, prob
        sides = {"L": [], "R": []}
        cx = self.w / 2
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            if y1 == y2:
                continue
            slope = (y2 - y1) / (x2 - x1 + 1e-6)
            if abs(slope) < 0.3:
                continue
            xm = (x1 + x2) / 2
            if slope < 0 and xm < cx:
                sides["L"].append((x1, y1, x2, y2))
            elif slope > 0 and xm > cx:
                sides["R"].append((x1, y1, x2, y2))
        fits = {}
        for k, segs in sides.items():
            if not segs:
                return xp, prob
            ys = np.array([[s[1], s[3]] for s in segs]).ravel()
            xs = np.array([[s[0], s[2]] for s in segs]).ravel()
            wts = np.repeat([np.hypot(s[2] - s[0], s[3] - s[1]) for s in segs], 2)
            fits[k] = np.polyfit(ys, xs, 1, w=wts)  # x = a*y + b
        ys = ROW_Y * (self.h - 1)
        center = (np.polyval(fits["L"], ys) + np.polyval(fits["R"], ys)) / 2
        ok = (ys >= y0) & (center >= 0) & (center < self.w)
        xp[ok] = center[ok] / self.w
        prob[ok] = 1.0
        return xp, prob

    def predict(self, frames_bgr):
        out = [self._one(f) for f in frames_bgr]
        return np.stack([o[0] for o in out]), np.stack([o[1] for o in out])


def build_predictor(kind, ckpt=None, autosteer_repo=None, autosteer_weights=None, device=None):
    if kind == "autopath":
        return AutoPathPredictor(ckpt, device)
    if kind == "autosteer":
        return AutoSteerPredictor(autosteer_repo, autosteer_weights, device)
    if kind == "classical":
        return ClassicalPredictor()
    raise ValueError(kind)
