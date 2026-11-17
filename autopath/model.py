"""Mô hình AutoPath: backbone CNN nhẹ + đầu hồi quy 64 hàng neo (row-anchor regression)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm

from .geometry import N_ROWS

BACKBONES = {
    # tên: (hàm dựng, enum trọng số ImageNet, số kênh đầu ra)
    "resnet18": (tvm.resnet18, tvm.ResNet18_Weights.IMAGENET1K_V1, 512),
    "resnet34": (tvm.resnet34, tvm.ResNet34_Weights.IMAGENET1K_V1, 512),
    "mobilenet_v3_small": (tvm.mobilenet_v3_small, tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1, 576),
    "mobilenet_v3_large": (tvm.mobilenet_v3_large, tvm.MobileNet_V3_Large_Weights.IMAGENET1K_V1, 960),
}


def _feature_extractor(name: str, pretrained: bool):
    fn, weights, ch = BACKBONES[name]
    net = fn(weights=weights if pretrained else None)
    if name.startswith("resnet"):
        body = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool,
                             net.layer1, net.layer2, net.layer3, net.layer4)
    else:
        body = net.features
    return body, ch


class AutoPathNet(nn.Module):
    """Ảnh (B,3,H,W) -> xp (B,64) in [0,1] và logit hợp lệ (B,64).

    head='spatial' (mặc định): nén kênh bằng conv 1x1 rồi làm phẳng, giữ thông tin
    vị trí ngang trên bản đồ đặc trưng (cần cho hồi quy hoành độ).
    head='gap': Global Average Pooling như đề cương ban đầu (dùng cho thí nghiệm so sánh).
    """

    def __init__(self, backbone="resnet18", pretrained=True, input_w=512, input_h=256,
                 head="spatial", hidden=256, dropout=0.2, n_rows=N_ROWS):
        super().__init__()
        self.cfg = dict(backbone=backbone, input_w=input_w, input_h=input_h,
                        head=head, hidden=hidden, dropout=dropout, n_rows=n_rows)
        self.n_rows = n_rows
        self.body, ch = _feature_extractor(backbone, pretrained)
        mobile = backbone.startswith("mobilenet")
        act = nn.Hardswish if mobile else nn.ReLU
        if head == "spatial":  # giữ lưới không gian 8x16
            self.reduce = nn.Sequential(
                nn.Conv2d(ch, 8, 1, bias=False),
                nn.BatchNorm2d(8), nn.ReLU(inplace=True))
            with torch.no_grad():
                dummy = torch.zeros(1, 3, input_h, input_w)
                fh, fw = self.body(dummy).shape[-2:]
            in_feat = 8 * fh * fw
        elif head == "gap":  # đối chứng theo đề cương ban đầu
            self.reduce = nn.AdaptiveAvgPool2d(1)
            in_feat = ch
        else:
            raise ValueError(head)
        self.fc = nn.Sequential(
            nn.Flatten(), nn.Linear(in_feat, hidden), act(),
            nn.Dropout(dropout), nn.Linear(hidden, 2 * n_rows))

    def forward(self, x):
        out = self.fc(self.reduce(self.body(x)))
        xp = torch.sigmoid(out[:, :self.n_rows])
        return xp, out[:, self.n_rows:]


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def save_checkpoint(path, model: AutoPathNet, extra=None):
    torch.save({"cfg": model.cfg, "state_dict": model.state_dict(), "extra": extra or {}}, path)


def load_checkpoint(path, device="cpu") -> AutoPathNet:
    ck = torch.load(path, map_location=device, weights_only=False)
    m = AutoPathNet(pretrained=False, **ck["cfg"])
    m.load_state_dict(ck["state_dict"])
    return m.to(device).eval()
