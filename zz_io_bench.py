# -*- coding: utf-8 -*-
"""320 模型推理耗时分解：定位主机侧 I/O 与 APU 执行各占多少

用于解释同一 320 产物在不同脚本下的延迟差异。

用法：
    /home/hill/lynxi/venv/bin/python zz_io_bench.py \
        --model-dir model_spikeyolov8_320_c16/Net_0 --imgsz 320 --runs 30
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import zz_conf

zz_conf.is_use_sdk = False

import zz_test_apu as T


def bench(fn, n=30, warm=5):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(ts)), float(np.min(ts)), float(np.max(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="model_spikeyolov8_320_c16/Net_0")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--runs", type=int, default=30)
    args = ap.parse_args()

    lyn = T.build_apu_model(args.model_dir)
    _, _, inp = T.preprocess("assets/bus.jpg", args.imgsz)

    print(f"模型: {args.model_dir}  输入: {args.imgsz}  轮次: {args.runs}\n")

    m1, lo1, hi1 = bench(lambda: T.apu_infer(lyn, inp), args.runs)
    print(f"  A 完整 apu_infer（建输入 + 推理 + 取输出）: {m1:8.2f} ms  [{lo1:.2f}, {hi1:.2f}]")

    dev_in = lyn.input_tensor().from_numpy(inp).apu()

    m2, lo2, _ = bench(lambda: (lyn(dev_in), lyn.output_list()[0][0].cpu().numpy()), args.runs)
    print(f"  B 复用设备输入（推理 + 取输出）:            {m2:8.2f} ms  [{lo2:.2f}, ...]")

    m3, lo3, _ = bench(lambda: lyn(dev_in), args.runs)
    print(f"  C 仅 lyn(dev_in)（不读输出）:               {m3:8.2f} ms  [{lo3:.2f}, ...]")

    m4, lo4, _ = bench(lambda: lyn.input_tensor().from_numpy(inp).apu(), args.runs)
    print(f"  D 仅 H2D（每次重建输入张量）:               {m4:8.2f} ms  [{lo4:.2f}, ...]")

    m5, lo5, _ = bench(lambda: lyn.output_list()[0][0].cpu().numpy(), args.runs)
    print(f"  E 仅 D2H（读取输出）:                       {m5:8.2f} ms  [{lo5:.2f}, ...]")

    n = 20
    t0 = time.perf_counter()
    for _ in range(n):
        lyn(dev_in)
    t_submit = (time.perf_counter() - t0) / n * 1000.0
    t0 = time.perf_counter()
    lyn.output_list()[0][0].cpu().numpy()
    t_sync = (time.perf_counter() - t0) * 1000.0
    print(f"\n  连续提交 {n} 次平均（后统一同步）:           {t_submit:8.2f} ms")
    print(f"  最终单次同步耗时:                           {t_sync:8.2f} ms")

    print("\n结论口径：A 为 host 端到端（含每次重建输入张量与取输出）；")
    print("B 去掉每次 H2D 后仍为同量级，说明延迟主体是 APU 执行本身；")
    print("C/D/E 用于给出各环节相对量级。")


if __name__ == "__main__":
    main()
