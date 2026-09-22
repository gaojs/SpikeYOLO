# -*- coding: utf-8 -*-
"""探针：查看 run_batch=2 产物的输出结构"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import zz_conf

zz_conf.is_use_sdk = False

import zz_test_apu as T
from zz_batch_bench import preprocess_batch


def main():
    x2 = preprocess_batch(["assets/bus.jpg", "assets/zidane.jpg"], 320)
    print("输入形状:", x2.shape)
    m2 = T.build_apu_model("model_spikeyolov8_320_b2_c16/Net_0")
    t = m2.input_tensor().from_numpy(x2).apu()
    m2(t)
    ol = m2.output_list()
    print("output_list 外层长度:", len(ol))
    for i, o in enumerate(ol):
        print(f"  [{i}] 类型 {type(o).__name__}")
        try:
            n = len(o)
        except Exception:
            n = -1
        print(f"       元素数 {n}")
        try:
            for j, tt in enumerate(o):
                arr = tt.cpu().numpy()
                print(f"        ({i},{j}) shape {arr.shape} dtype {arr.dtype} "
                      f"min {arr.min():.3f} max {arr.max():.3f}")
        except Exception as e:
            print("        遍历失败:", e)


if __name__ == "__main__":
    main()
