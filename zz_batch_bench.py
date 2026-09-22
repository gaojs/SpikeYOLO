# -*- coding: utf-8 -*-
"""SpikeYOLO batch>1 推理验证与收益测量

对比 batch=1 逐帧 2 次 与 batch=2 单批 1 次的延迟、加速比与结果一致性。

用法:
    /home/hill/lynxi/venv/bin/python zz_batch_bench.py --imgsz 320 --runs 20
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import zz_conf

zz_conf.is_use_sdk = False

import zz_test_apu as T


def preprocess_batch(paths, imgsz):
    from ultralytics.data.augment import LetterBox

    lb = LetterBox(imgsz, auto=False)
    arrs = []
    for p in paths:
        im = lb(image=cv2.imread(p))
        im = np.ascontiguousarray(im[..., ::-1].transpose(2, 0, 1))
        arrs.append(im.transpose(1, 2, 0))
    return np.ascontiguousarray(np.stack(arrs)).astype("uint8")


def apu_run(model, x):
    """按批拼接输出。run_batch=N 时 output_list() 含 N 个 (1,N,84)，需逐帧取出后拼接。"""
    t = model.input_tensor().from_numpy(x).apu()
    model(t)
    outs = [g[0].cpu().numpy() for g in model.output_list()]
    return np.float32(np.concatenate(outs, axis=0))


def bench(fn, runs, warmup=3):
    for _ in range(warmup):
        fn()
    lat = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        lat.append((time.perf_counter() - t0) * 1000.0)
    return np.array(lat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-b1", default="model_spikeyolov8_320_c16/Net_0")
    ap.add_argument("--model-b2", default="model_spikeyolov8_320_b2_c16/Net_0")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--images", nargs="+", default=["assets/bus.jpg", "assets/zidane.jpg"])
    a = ap.parse_args()

    x2 = preprocess_batch(a.images, a.imgsz)
    print("输入图片:", a.images)
    print("批输入形状:", x2.shape, x2.dtype, "\n")

    print("=== batch=1 模型（逐帧） ===")
    m1 = T.build_apu_model(a.model_b1)
    out1 = apu_run(m1, x2[0:1])
    print("  单帧输出形状:", out1.shape)
    l1 = bench(lambda: apu_run(m1, x2[0:1]), a.runs)
    print(f"  单帧延迟: 均值 {l1.mean():.2f} ms  中位 {np.median(l1):.2f} ms  FPS {1000 / l1.mean():.2f}")
    two = l1.mean() * 2
    print(f"  处理 2 帧（两次调用）: {two:.2f} ms")

    print("\n=== batch=2 模型 ===")
    m2 = T.build_apu_model(a.model_b2)
    out2 = apu_run(m2, x2)
    print("  批输出形状:", out2.shape)
    l2 = bench(lambda: apu_run(m2, x2), a.runs)
    print(f"  单批延迟: 均值 {l2.mean():.2f} ms  中位 {np.median(l2):.2f} ms  按帧计 FPS {2 * 1000 / l2.mean():.2f}")

    print("\n=== 收益与一致性 ===")
    print(f"  2 帧耗时: batch=1 逐帧 {two:.2f} ms  vs  batch=2 单批 {l2.mean():.2f} ms")
    print(f"  批处理加速比: {two / l2.mean():.2f}x（单批延迟为单帧的 {l2.mean() / l1.mean():.2f} 倍）")

    if out2.shape[0] == 2:
        refs = [out1[0], apu_run(m1, x2[1:2])[0]]
        for i, r in enumerate(refs):
            p = out2[i]
            if p.shape != r.shape:
                print(f"  第 {i} 帧形状不一致: {p.shape} vs {r.shape}")
                continue
            rel = np.sqrt(((p - r) ** 2).sum()) / (np.sqrt((r ** 2).sum()) + 1e-12)
            cos = float(np.dot(p.ravel(), r.ravel()) /
                        (np.linalg.norm(p.ravel()) * np.linalg.norm(r.ravel()) + 1e-12))
            print(f"  第 {i} 帧 相对RMSE={rel:.4f}  余弦={cos:.6f}  最大绝对误差={np.abs(p - r).max():.4f}")
    else:
        print(f"  警告: batch=2 输出第 0 维为 {out2.shape[0]}，与预期 2 不一致")


if __name__ == "__main__":
    main()
