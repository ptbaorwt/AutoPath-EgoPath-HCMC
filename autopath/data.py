"""Đọc bộ dữ liệu TuSimple (hoặc tập tự gán nhãn cùng định dạng), tạo nhãn ego-path và Dataset PyTorch."""
from __future__ import annotations

import glob
import json
import os
import random
from collections import Counter

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .geometry import CROP_TOP, ORIG_H, ORIG_W, path_curvature_px, tusimple_to_egopath

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------- tìm file nhãn
def find_label_files(root: str):
    """Tìm file nhãn train (label_data_*.json) và test (test_label.json) trong thư mục TuSimple."""
    # Chỉ tìm tối đa 3 cấp thư mục: duyệt đệ quy toàn bộ TuSimple (~130k file) rất chậm trên Kaggle.
    def find(name):
        return sorted({f for d in range(4) for f in glob.glob(os.path.join(root, *(["*"] * d), name))})
    train, test = find("label_data_*.json"), find("test_label*.json")
    return train, test


def _resolve_image(root: str, json_path: str, raw_file: str) -> str | None:
    """raw_file là đường dẫn tương đối; thử các vị trí hay gặp của bản TuSimple trên Kaggle."""
    jd = os.path.dirname(json_path)
    for base in (jd, os.path.join(jd, "test_set"), os.path.join(root, "test_set"),
                 os.path.join(root, "train_set"), root):
        p = os.path.join(base, raw_file)
        if os.path.isfile(p):
            return p
    return None


def clip_id(raw_file: str) -> str:
    """'clips/0313-1/6040/20.jpg' -> '0313-1/6040' (một clip = 20 khung hình liên tiếp)."""
    parts = raw_file.replace("\\", "/").split("/")
    return "/".join(parts[1:-1]) if len(parts) >= 3 else parts[0]


def build_records(root: str, json_files: list[str], img_size: tuple[int, int] | None = None):
    """Đọc các file nhãn -> danh sách bản ghi {image, clip, xp, valid, curvature}.

    img_size=(w, h) chỉ cần khi ảnh không phải 1280x720 (ví dụ tập tự gán nhãn).
    """
    records, stats, bases = [], Counter(), {}
    for jf in json_files:
        with open(jf, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                e = json.loads(line)
                stats["total"] += 1
                if jf not in bases:  # chỉ dò vị trí ảnh một lần cho mỗi file nhãn (stat trên Kaggle rất chậm)
                    p0 = _resolve_image(root, jf, e["raw_file"])
                    bases[jf] = p0[: -len(e["raw_file"])] if p0 else None
                img = os.path.join(bases[jf], e["raw_file"]) if bases[jf] else None
                if img is None:
                    stats["missing_image"] += 1
                    continue
                w, h = img_size or (e.get("width", ORIG_W), e.get("height", ORIG_H))
                lab = tusimple_to_egopath(e, w, h)
                if lab is None:
                    stats["no_ego_lanes"] += 1
                    continue
                xp, valid = lab
                records.append(dict(image=img, raw_file=e["raw_file"], clip=clip_id(e["raw_file"]),
                                    xp=xp, valid=valid,
                                    curvature=path_curvature_px(xp, valid, w),
                                    tags=e.get("tags", [])))
                stats["kept"] += 1
    return records, dict(stats)


def split_by_clip(records, val_frac=0.1, seed=42):
    """Chia train/val theo clip để khung hình của cùng một clip không nằm ở cả hai tập."""
    clips = sorted({r["clip"] for r in records})
    rng = random.Random(seed)
    rng.shuffle(clips)
    n_val = max(1, int(round(len(clips) * val_frac)))
    val_clips = set(clips[:n_val])
    return ([r for r in records if r["clip"] not in val_clips],
            [r for r in records if r["clip"] in val_clips])


def load_or_build(root, split, cache_dir, val_frac=0.1, seed=42):
    """split in {'train', 'val', 'test'}; lưu cache để lần sau không phải đọc lại JSON."""
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f"records_{split}.pt")
    if os.path.isfile(cache):
        return torch.load(cache, weights_only=False)
    train_files, test_files = find_label_files(root)
    if split == "test":
        recs, stats = build_records(root, test_files)
    else:
        allrec, stats = build_records(root, train_files)
        tr, va = split_by_clip(allrec, val_frac, seed)
        torch.save({"records": tr, "stats": stats}, os.path.join(cache_dir, "records_train.pt"))
        torch.save({"records": va, "stats": stats}, os.path.join(cache_dir, "records_val.pt"))
        recs = tr if split == "train" else va
    out = {"records": recs, "stats": stats}
    torch.save(out, cache)
    return out


# ---------------------------------------------------------------- tiền xử lý ảnh
def crop_resize(img_bgr: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """Cắt phần trời phía trên (tỉ lệ 80/720 chiều cao) rồi resize về out_w x out_h."""
    h = img_bgr.shape[0]
    top = int(round(h * CROP_TOP / ORIG_H))
    return cv2.resize(img_bgr[top:], (out_w, out_h), interpolation=cv2.INTER_AREA)


def to_tensor(img_rgb: np.ndarray) -> torch.Tensor:
    x = (img_rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


def build_photometric_aug():
    """Tăng cường quang học (không đổi hình học nên không cần sửa nhãn)."""
    import albumentations as A
    return A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=20, val_shift_limit=15, p=0.3),
        A.RandomShadow(p=0.2),
        A.OneOf([A.GaussianBlur(blur_limit=(3, 5)), A.MotionBlur(blur_limit=5)], p=0.2),
        A.GaussNoise(std_range=(0.01, 0.04), p=0.2),
    ])


class EgoPathDataset(Dataset):
    def __init__(self, records, input_w=512, input_h=256, augment=False, hflip_p=0.5):
        self.records = records
        self.w, self.h = input_w, input_h
        self.augment = augment
        self.hflip_p = hflip_p
        self.photo = build_photometric_aug() if augment else None

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        img = cv2.imread(r["image"])
        if img is None:
            raise FileNotFoundError(r["image"])
        img = cv2.cvtColor(crop_resize(img, self.w, self.h), cv2.COLOR_BGR2RGB)
        xp, valid = r["xp"].copy(), r["valid"].copy()
        if self.augment:
            img = self.photo(image=img)["image"]
            if random.random() < self.hflip_p:
                # Lật ngang: x' = W - x  <=>  xp' = 1 - xp trên các hàng hợp lệ
                img = img[:, ::-1]
                xp = np.where(valid > 0.5, 1.0 - xp, 0.0).astype(np.float32)
        return to_tensor(img), torch.from_numpy(xp), torch.from_numpy(valid), i
