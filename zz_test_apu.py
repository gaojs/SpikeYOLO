# -*- coding: utf-8 -*-
"""
SpikeYOLO HP300/KA200 APU 功能与性能验证脚本
- 功能：APU 输出与 PyTorch golden 输出逐元素相对误差 + NMS 检测框对比
- 性能：APU 端到端推理延迟 / FPS（含 H2D 搬运与 D2H 拷贝）
- 能耗：通过 lynxi-smi 板卡级 Power Usage 低频采样，给出空闲/推理功耗与每帧能耗粗估
用法：
    /home/hill/lynxi/venv/bin/python zz_test_apu.py \
        --model-dir model_spikeyolov8_320_c16/Net_0 --imgsz 320 --runs 50
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 先关闭 SDK 分支，避免 predictor 导入时按硬编码路径加载模型
import zz_conf
zz_conf.is_use_sdk = False


def build_apu_model(model_dir):
    """用仓库自带的 lynpy.py 封装加载 ABC 编译产物。"""
    import lynpy
    return lynpy.Model(path=model_dir)


def preprocess(image_path, imgsz=320):
    """复刻 ultralytics/engine/predictor.py 的 letterbox + NHWC uint8 输入准备。"""
    import cv2
    from ultralytics.data.augment import LetterBox

    im0 = cv2.imread(image_path)  # BGR
    lb = LetterBox(imgsz, auto=False)
    im = lb(image=im0)
    im = im[..., ::-1].transpose(2, 0, 1)  # BGR->RGB, HWC->CHW
    im = np.ascontiguousarray(im)
    im_t = torch.from_numpy(im).unsqueeze(0)  # (1,3,imgsz,imgsz) uint8
    input_data = im_t.detach().clone().numpy().transpose(0, 2, 3, 1).astype("uint8")
    return im0, im_t, input_data


_torch_model = None


def torch_golden(im_t):
    """PyTorch golden：与 predictor 相同的 uint8->float/255 预处理，返回 raw 输出。"""
    global _torch_model
    from ultralytics import YOLO
    if _torch_model is None:
        _torch_model = YOLO("best.pt")
        _torch_model.model.eval()
    im = im_t.float() / 255.0
    with torch.no_grad():
        res = _torch_model.model(im)
    if isinstance(res, (list, tuple)):
        res = res[0]
    return np.float32(res.cpu().numpy())


def apu_infer(lyn_model, input_data):
    """APU 单次推理，返回 (1,84,N) float32。"""
    lyn_input = lyn_model.input_tensor().from_numpy(input_data).apu()
    lyn_model(lyn_input)
    apu_out = lyn_model.output_list()[0][0].cpu().numpy()
    return np.float32(apu_out.transpose(0, 2, 1))


def functional_check(lyn_model, images, imgsz=320):
    """功能验证：APU vs PyTorch 相对 RMSE，并做 NMS 后检测框数量对比。"""
    from ultralytics.utils.ops import non_max_suppression

    print("\n================ 功能验证 ================")
    for img in images:
        _, im_t, input_data = preprocess(img, imgsz)

        torch_out = torch_golden(im_t)          # (1,84,N)
        apu_out = apu_infer(lyn_model, input_data)  # (1,84,N)

        a, t = apu_out, torch_out
        if a.shape != t.shape:
            print(f"[{os.path.basename(img)}] 形状不一致: APU {a.shape} vs torch {t.shape}")
            continue
        rel_rmse = np.sqrt(np.sum((a - t) ** 2)) / np.sqrt(np.sum(a ** 2))
        cos = np.dot(a.ravel(), t.ravel()) / (
            np.linalg.norm(a.ravel()) * np.linalg.norm(t.ravel()) + 1e-12
        )
        max_abs = np.max(np.abs(a - t))

        apu_det = non_max_suppression(torch.from_numpy(a).float(), conf_thres=0.25, iou_thres=0.45, max_det=300)
        tor_det = non_max_suppression(torch.from_numpy(t).float(), conf_thres=0.25, iou_thres=0.45, max_det=300)
        print(f"\n[{os.path.basename(img)}]  输入 {imgsz}x{imgsz}")
        print(f"  torch raw shape : {t.shape}   APU raw shape : {a.shape}")
        print(f"  相对 RMSE       : {rel_rmse:.4f}   (历史 640 参考 ~8%)")
        print(f"  余弦相似度      : {cos:.6f}")
        print(f"  最大绝对误差    : {max_abs:.4f}")
        n_apu, n_tor = len(apu_det[0]), len(tor_det[0])
        print(f"  APU 检测框数    : {n_apu}   PyTorch 检测框数 : {n_tor}")
        if n_apu:
            cls, cnt = np.unique(apu_det[0][:, 5].int().numpy(), return_counts=True)
            names = _torch_model.names if _torch_model is not None else {}
            print(f"  APU 类别        : { {names.get(int(c), int(c)): int(n) for c, n in zip(cls, cnt)} }")
    print("==========================================")


def perf_check(lyn_model, image, runs=50, imgsz=320):
    """性能验证：端到端 APU 推理延迟与 FPS。"""
    print("\n================ 性能验证 ================")
    _, _, input_data = preprocess(image, imgsz)

    for _ in range(5):  # warmup
        apu_infer(lyn_model, input_data)

    lat = []
    for _ in range(runs):
        t0 = time.time()
        apu_infer(lyn_model, input_data)
        lat.append((time.time() - t0) * 1000.0)

    lat = np.array(lat)
    print(f"  轮次            : {runs} 次（另有 warmup 5 次）")
    print(f"  平均延迟        : {lat.mean():.2f} ms")
    print(f"  中位延迟        : {np.median(lat):.2f} ms")
    print(f"  最小 / 最大     : {lat.min():.2f} / {lat.max():.2f} ms")
    print(f"  吞吐率 FPS      : {1000.0 / lat.mean():.2f}")
    print("  （延迟含：H2D 输入搬运 + APU 推理 + D2H 输出拷贝，为 host 端到端耗时）")
    print("==========================================")


SMI_PATH = "/usr/bin/lynxi-smi"


def sample_power_w():
    """从 lynxi-smi 读取板卡级 Power Usage（W）；读取失败返回 None。"""
    import re
    import subprocess

    try:
        out = subprocess.run(
            [SMI_PATH], capture_output=True, text=True, timeout=10
        ).stdout
    except Exception:
        return None
    m = re.search(r"Power\(Usage/Cap\):\s*([0-9.]+)W", out)
    return float(m.group(1)) if m else None


def energy_check(
    lyn_model,
    image,
    imgsz=320,
    idle_seconds=5.0,
    load_seconds=15.0,
    power_interval=1.0,
):
    """能耗验证：lynxi-smi 低频采样空闲/推理期功耗，估算每帧能耗。

    口径说明：lynxi-smi 为板卡级低频点采样，非积分测量；每帧能耗按
    「推理窗口内平均功耗 × 窗口时长 / 实际完成帧数」计算。采样开销会降低
    窗口内帧数，但功耗与帧数同窗口统计，故每帧能耗仍成立。
    """
    print("\n================ 能耗验证 ================")
    if sample_power_w() is None:
        print(f"  {SMI_PATH} 未返回功耗字段，无法实测")
        return

    idle = []
    t_end = time.time() + idle_seconds
    while time.time() < t_end:
        v = sample_power_w()
        if v is not None:
            idle.append(v)
        time.sleep(power_interval)
    idle_avg = float(np.mean(idle)) if idle else float("nan")

    _, _, input_data = preprocess(image, imgsz)
    for _ in range(5):  # warmup
        apu_infer(lyn_model, input_data)

    load = []
    frames = 0
    t0 = time.time()
    t_end = t0 + load_seconds
    while time.time() < t_end:
        apu_infer(lyn_model, input_data)
        frames += 1
        v = sample_power_w()
        if v is not None:
            load.append(v)
    dur = time.time() - t0
    load_avg = float(np.mean(load)) if load else float("nan")

    print(f"  采集方式        : {SMI_PATH} 板卡级 Power Usage，间隔 {power_interval}s 点采样")
    print(f"  空闲功耗        : {idle_avg:.2f} W（{len(idle)} 个采样点 / {idle_seconds:.0f}s）")
    print(f"  推理功耗        : {load_avg:.2f} W（{len(load)} 个采样点 / {dur:.2f}s）")
    print(f"  功耗增量        : {load_avg - idle_avg:+.2f} W")
    print(f"  窗口内帧数      : {frames} 帧（{frames / dur:.2f} FPS）")
    print(f"  每帧能耗(整板)  : {load_avg * dur / frames * 1000:.2f} mJ/frame")
    print(f"  每帧能耗(净增)  : {(load_avg - idle_avg) * dur / frames * 1000:.2f} mJ/frame")
    print("  说明：低频点采样、非积分测量，仅用于量级参考；净增值受采样噪声影响。")
    print("==========================================")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="model_spikeyolov8_320_c16/Net_0")
    ap.add_argument("--images", nargs="+", default=["assets/bus.jpg", "assets/zidane.jpg"])
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--skip-functional", action="store_true")
    ap.add_argument("--skip-energy", action="store_true")
    ap.add_argument("--energy-only", action="store_true")
    ap.add_argument("--load-seconds", type=float, default=15.0)
    args = ap.parse_args()

    print(f"加载 APU 模型: {args.model_dir}")
    lyn_model = build_apu_model(args.model_dir)
    print("模型加载成功")

    if not args.skip_functional and not args.energy_only:
        functional_check(lyn_model, args.images, imgsz=args.imgsz)
    if not args.energy_only:
        perf_check(lyn_model, args.images[0], runs=args.runs, imgsz=args.imgsz)
    if not args.skip_energy:
        energy_check(
            lyn_model, args.images[0], imgsz=args.imgsz, load_seconds=args.load_seconds
        )


if __name__ == "__main__":
    main()
