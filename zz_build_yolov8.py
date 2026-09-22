# -*- coding: utf-8 -*-
"""SpikeYOLO 离线模型编译脚本（支持 batch / 分辨率 / 核数参数化）

用法示例：
    # 320、单批、16 宏核（与既有产物一致）
    /home/hill/lynxi/venv/bin/python zz_build_yolov8.py

    # 320、batch=2
    /home/hill/lynxi/venv/bin/python zz_build_yolov8.py \
        --batch 2 --save model_spikeyolov8_320_b2_c16

    # 224、batch=2（内存受限时的降分辨率尝试）
    /home/hill/lynxi/venv/bin/python zz_build_yolov8.py \
        --imgsz 224 --batch 2 --save model_spikeyolov8_224_b2_c16

内存采样：编译期间后台按 --mem-interval 秒采样 cgroup 内存用量，写入 --mem-log，
用于判断是否触及容器内存上限。
"""
import argparse
import os
import threading
import time

import lyngor as lyn

import zz_conf

zz_conf.is_use_sdk = False  # 编译时不推理

CGROUP_CURRENT_CANDIDATES = [
    "/sys/fs/cgroup/memory/memory.usage_in_bytes",  # cgroup v1
    "/sys/fs/cgroup/memory.current",                 # cgroup v2
]
CGROUP_PEAK_CANDIDATES = [
    "/sys/fs/cgroup/memory/memory.max_usage_in_bytes",  # cgroup v1
    "/sys/fs/cgroup/memory.peak",                       # cgroup v2
]


def _read_int(path):
    try:
        with open(path) as f:
            return int(f.read().strip())
    except Exception:
        return None


def _read_first(paths):
    for p in paths:
        v = _read_int(p)
        if v is not None:
            return v, p
    return None, None


def _proc_tree_rss_kb(root_pid):
    """统计 root_pid 及其所有子进程的 RSS 合计（KB）。

    cgroup usage 含可回收的 page cache，不能反映真实内存压力；编译是否 OOM
    取决于匿名内存，因此以进程树 RSS 作为主要判据。
    """

    def rss_kb(pid):
        try:
            with open(f"/proc/{pid}/status") as f:
                for line in f:
                    if line.startswith("VmRSS"):
                        return int(line.split()[1])
        except Exception:
            pass
        return 0

    seen, stack = set(), [root_pid]
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        try:
            for tid in os.listdir(f"/proc/{pid}/task"):
                try:
                    with open(f"/proc/{pid}/task/{tid}/children") as f:
                        stack += [int(x) for x in f.read().split()]
                except Exception:
                    pass
        except Exception:
            pass
    return sum(rss_kb(p) for p in seen), len(seen)


class MemWatcher(threading.Thread):
    """后台采样内存：同时记录 cgroup 用量与编译进程树 RSS。"""

    def __init__(self, log_path, interval=10.0):
        super().__init__(daemon=True)
        self.log_path = log_path
        self.interval = interval
        self.peak_cgroup = 0
        self.peak_rss = 0
        self.source = None
        self._stop_event = threading.Event()

    def run(self):
        with open(self.log_path, "w") as f:
            while not self._stop_event.is_set():
                cur, src = _read_first(CGROUP_CURRENT_CANDIDATES)
                rss, nproc = _proc_tree_rss_kb(os.getpid())
                if cur is not None:
                    self.source = src
                    self.peak_cgroup = max(self.peak_cgroup, cur)
                self.peak_rss = max(self.peak_rss, rss)
                f.write(
                    f"{time.strftime('%H:%M:%S')} "
                    f"cgroup={(cur // 1024 // 1024) if cur else -1}MB "
                    f"rss_sum={rss // 1024}MB procs={nproc}\n"
                )
                f.flush()
                self._stop_event.wait(self.interval)

    def stop(self):
        self._stop_event.set()
        self.join(timeout=10)


def lyn_build(model_file, save_path, model_shape, run_batch=1, num_cores=16,
              opt_level=3, core_mem_mode=0, post_mode=500, lut_opt=1):
    mod = lyn.DLModel()
    mod.load(model_file, model_type="Pytorch", inputs_dict={"images": model_shape},
             in_type="uint8",                  # 前处理：指定输入数据类型
             out_type="float16",               # 后处理：指定输出数据类型
             variance=(255,),                  # 前处理：添加归一化参数
             out_transpose_axis=[(0, 2, 1)],   # 后处理：输出转置(1,84,N)->(1,N,84)样式
             transpose_axis=[(0, 3, 1, 2)])    # 前处理：输入转置NCHW->NHWC
    offline_builder = lyn.Builder(target="apu")
    offline_builder.build(mod.graph, mod.param, out_path=save_path, run_batch=run_batch,
                          serialize=False, core_mem_mode=core_mem_mode,
                          post_mode=post_mode, lut_opt=lut_opt, opt_level=opt_level,
                          num_cores=num_cores)
    print(" ###[lyn_build] model build end! save path is", save_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-file", default="best.pt")
    ap.add_argument("--save", default="model_spikeyolov8_320_c16")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--batch", type=int, default=1, help="run_batch，运行时 batch 数")
    ap.add_argument("--num-cores", type=int, default=16)
    ap.add_argument("--opt-level", type=int, default=3)
    ap.add_argument("--core-mem-mode", type=int, default=0)
    ap.add_argument("--post-mode", type=int, default=500)
    ap.add_argument("--lut-opt", type=int, default=1)
    ap.add_argument("--mem-log", default=None)
    ap.add_argument("--mem-interval", type=float, default=10.0)
    args = ap.parse_args()

    model_shape = (1, 3, args.imgsz, args.imgsz)
    mem_log = args.mem_log or f"mem_stat_{os.path.basename(args.save)}.log"

    print(f"编译配置: imgsz={args.imgsz} batch={args.batch} num_cores={args.num_cores} "
          f"opt_level={args.opt_level} trace_shape={model_shape} -> {args.save}")

    watcher = MemWatcher(mem_log, args.mem_interval)
    watcher.start()
    t0 = time.time()
    try:
        lyn_build(args.model_file, args.save, model_shape, run_batch=args.batch,
                  num_cores=args.num_cores, opt_level=args.opt_level,
                  core_mem_mode=args.core_mem_mode, post_mode=args.post_mode,
                  lut_opt=args.lut_opt)
    finally:
        elapsed = time.time() - t0
        watcher.stop()

    peak_cg, peak_src = _read_first(CGROUP_PEAK_CANDIDATES)
    print(f" ###[lyn_build] 耗时 {elapsed:.1f}s")
    print(f" ###[lyn_build] 进程树 RSS 峰值 {watcher.peak_rss / 1024 / 1024:.0f} MB（日志 {mem_log}）")
    print(f" ###[lyn_build] cgroup 用量峰值 {watcher.peak_cgroup / 1024 / 1024:.0f} MB"
          f"（含 page cache，来源 {watcher.source}）")
    if peak_cg:
        print(f" ###[lyn_build] cgroup 历史峰值 {peak_cg / 1024 / 1024:.0f} MB（{peak_src}）")


if __name__ == "__main__":
    main()
