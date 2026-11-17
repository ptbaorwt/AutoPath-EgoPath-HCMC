"""Đo tốc độ suy luận (FPS), số tham số và dung lượng mô hình trên cùng một phần cứng.

Ví dụ:
  python benchmark_fps.py --ckpts runs/res18_aug/best.pt runs/mbv3s_aug/best.pt \
      --autosteer-repo auto_steer --autosteer-weights weights/autosteer.pth --classical
Nếu chưa có checkpoint vẫn đo được tốc độ kiến trúc: --backbones resnet18 mobilenet_v3_small
"""
import argparse
import json
import os
import platform
import time

import numpy as np
import torch

from autopath.model import AutoPathNet, count_params
from autopath.predictors import AutoPathPredictor, AutoSteerPredictor, ClassicalPredictor


def bench_tensor(fn, x, warmup=20, iters=200, cuda=False):
    for _ in range(warmup):
        fn(x)
    if cuda:
        torch.cuda.synchronize()
    t = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn(x)
        if cuda:
            torch.cuda.synchronize()
        t.append(time.perf_counter() - t0)
    t = np.array(t)
    return dict(fps=float(1.0 / t.mean()), latency_ms_mean=float(t.mean() * 1e3), latency_ms_p95=float(np.percentile(t, 95) * 1e3))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="*", default=[])
    p.add_argument("--backbones", nargs="*", default=[])
    p.add_argument("--autosteer-repo")
    p.add_argument("--autosteer-weights")
    p.add_argument("--classical", action="store_true")
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--out", default="results/benchmark.json")
    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cuda = dev == "cuda"
    torch.backends.cudnn.benchmark = True
    hw = torch.cuda.get_device_name(0) if cuda else platform.processor() or "CPU"
    rows = []

    def add(name, model, x):
        model = model.to(dev).eval()
        with torch.no_grad():
            r = bench_tensor(lambda t: model(t), x.to(dev), iters=a.iters, cuda=cuda)
        n = count_params(model)
        rows.append(dict(method=name, input=list(x.shape[-2:][::-1]), params_M=n / 1e6, size_MB_fp32=n * 4 / 2**20, **r))

    for b in a.backbones:
        add(f"AutoPath-{b} (kiến trúc)", AutoPathNet(b, pretrained=False), torch.randn(1, 3, 256, 512))
    for c in a.ckpts:
        pr = AutoPathPredictor(c, dev)
        add(pr.name, pr.model, torch.randn(1, 3, pr.h, pr.w))
    if a.autosteer_repo:
        pr = AutoSteerPredictor(a.autosteer_repo, a.autosteer_weights, dev)
        add(pr.name, pr.model, torch.rand(1, 3, 512, 1024))
    if a.classical:
        cp = ClassicalPredictor()
        frame = (np.random.rand(720, 1280, 3) * 255).astype(np.uint8)
        r = bench_tensor(lambda f: cp.predict([f]), frame, warmup=5, iters=min(a.iters, 100))
        rows.append(dict(method=cp.name + " (CPU)", input=[cp.w, cp.h], params_M=0.0, size_MB_fp32=0.0, **r))

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"device": hw, "torch": torch.__version__, "results": rows}, f, indent=2, ensure_ascii=False)
    print(f"Thiết bị: {hw}")
    for r in rows:
        print(f"{r['method']:<40} {r['fps']:8.1f} FPS  {r['latency_ms_mean']:7.2f} ms  {r['params_M']:6.2f}M  {r['size_MB_fp32']:6.1f} MB")


if __name__ == "__main__":
    main()
