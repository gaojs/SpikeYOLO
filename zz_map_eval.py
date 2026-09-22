# -*- coding: utf-8 -*-
"""SpikeYOLO 在 HP300/KA200 上的检测精度评估（mAP / Precision / Recall）

在 coco128（128 张带标注图片）上评估 APU 离线模型的检测精度，并与
PyTorch（best.pt）参考结果对比，用于量化 INT8 编译带来的精度损失。

用法：
    /home/hill/lynxi/venv/bin/python zz_map_eval.py \
        --model-dir model_spikeyolov8_320_c16/Net_0 --imgsz 320
    /home/hill/lynxi/venv/bin/python zz_map_eval.py \
        --model-dir model_spikeyolov8_640_1220/Net_0 --imgsz 640 --torch

注意：coco128 取自 COCO train2017 前 128 张，与 SpikeYOLO 训练集重叠，
绝对 mAP 偏乐观，不能作为泛化精度；本脚本主要用于 APU(INT8) 与
PyTorch(FP32) 的同数据对比。
"""
import argparse
import glob
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import zz_conf

zz_conf.is_use_sdk = False

DATA_ROOT = "/home/hill/lynxi/data/coco128"
NC = 80


def build_apu_model(model_dir):
    import lynpy

    return lynpy.Model(path=model_dir)


def preprocess(image_path, imgsz):
    from ultralytics.data.augment import LetterBox

    im0 = cv2.imread(image_path)
    lb = LetterBox(imgsz, auto=False)
    im = lb(image=im0)
    im = im[..., ::-1].transpose(2, 0, 1)
    im = np.ascontiguousarray(im)
    im_t = torch.from_numpy(im).unsqueeze(0)
    input_data = im_t.detach().clone().numpy().transpose(0, 2, 3, 1).astype("uint8")
    return im0, im_t, input_data


_torch_model = None


def apu_raw(lyn_model, input_data):
    """APU 原始输出为 (1,N,84)，转置为 (1,84,N) 以匹配 NMS 输入约定。"""
    lyn_input = lyn_model.input_tensor().from_numpy(input_data).apu()
    lyn_model(lyn_input)
    out = lyn_model.output_list()[0][0].cpu().numpy()
    return np.float32(out.transpose(0, 2, 1))


def torch_raw(im_t):
    """PyTorch best.pt 输出本身即为 (1,84,N)，不做转置。"""
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


def load_gt(label_path, w, h):
    if not os.path.exists(label_path):
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.int64)
    rows = []
    with open(label_path) as f:
        for line in f:
            p = line.split()
            if len(p) < 5:
                continue
            rows.append([float(x) for x in p[:5]])
    if not rows:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.int64)
    a = np.array(rows, np.float32)
    cls = a[:, 0].astype(np.int64)
    cx, cy, bw, bh = a[:, 1] * w, a[:, 2] * h, a[:, 3] * w, a[:, 4] * h
    boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)
    return boxes, cls


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None]
    area_b = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[None, :]
    union = area_a + area_b - inter + 1e-9
    return inter / union


def compute_ap(recall, precision):
    """101 点插值 AP（COCO 口径）。"""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    r = np.linspace(0, 1, 101)
    return float(np.interp(r, mrec, mpre).mean())


def eval_detections(preds, gts, iou_levels, pr_conf=0.25):
    """preds/gts: {img_id: (boxes, scores, classes)} → mAP 与操作点 P/R。"""
    img_ids = sorted(preds.keys())
    stats = {t: {"tp": 0, "fp": 0, "fn": 0} for t in iou_levels}
    ap50_per_class, ap_per_class = {}, {}

    # 按类别聚合预测，供 AP 计算
    by_cls = {}
    for iid in img_ids:
        pb, ps, pc = preds[iid]
        for j in range(len(pb)):
            by_cls.setdefault(int(pc[j]), []).append((iid, pb[j], float(ps[j])))

    gt_by_cls = {}
    for iid in img_ids:
        gb, gc = gts[iid]
        for j in range(len(gb)):
            gt_by_cls.setdefault(int(gc[j]), []).append((iid, gb[j]))

    for c in sorted(gt_by_cls.keys()):
        gts_c = gt_by_cls[c]
        n_gt = len(gts_c)
        dets = sorted(by_cls.get(c, []), key=lambda x: -x[2])

        # 每个 IoU 阈值下的 AP
        aps = []
        for t in iou_levels:
            # 本类别在每个图中的 GT 索引
            cls_gt_idx = {}
            for iid in {i for i, _ in gts_c}:
                gb, gc = gts[iid]
                cls_gt_idx[iid] = [k for k in range(len(gb)) if int(gc[k]) == c]
            matched = {iid: np.zeros(len(v), bool) for iid, v in cls_gt_idx.items()}

            tp = np.zeros(len(dets), np.float32)
            fp = np.zeros(len(dets), np.float32)
            for di, (iid, box, _) in enumerate(dets):
                idxs = cls_gt_idx.get(iid, [])
                if not idxs:
                    fp[di] = 1
                    continue
                gb = gts[iid][0][idxs]
                ious = iou_matrix(box[None, :], gb)[0]
                best = int(np.argmax(ious)) if len(ious) else -1
                if best >= 0 and ious[best] >= t and not matched[iid][best]:
                    matched[iid][best] = True
                    tp[di] = 1
                else:
                    fp[di] = 1
            tp_c, fp_c = np.cumsum(tp), np.cumsum(fp)
            rec = tp_c / (n_gt + 1e-9)
            prec = tp_c / (tp_c + fp_c + 1e-9)
            aps.append(compute_ap(rec, prec))

        ap50_per_class[c] = aps[0]
        ap_per_class[c] = float(np.mean(aps))

    # 操作点 P/R（IoU=0.5，仅统计分数 >= pr_conf 的预测）
    t = 0.5
    tp = fp = fn = 0
    for iid in img_ids:
        gb, gc = gts[iid]
        pb, ps, pc = preds[iid]
        keep = ps >= pr_conf
        pb, ps, pc = pb[keep], ps[keep], pc[keep]
        used = np.zeros(len(gb), bool)
        order = np.argsort(-ps)
        for j in order:
            cand = [k for k in range(len(gb)) if int(gc[k]) == int(pc[j]) and not used[k]]
            if not cand:
                fp += 1
                continue
            ious = iou_matrix(pb[j][None, :], gb[cand])[0]
            b = int(np.argmax(ious))
            if ious[b] >= t:
                used[cand[b]] = True
                tp += 1
            else:
                fp += 1
        fn += int((~used).sum())

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    return {
        "mAP50": float(np.mean(list(ap50_per_class.values()))) if ap50_per_class else 0.0,
        "mAP50_95": float(np.mean(list(ap_per_class.values()))) if ap_per_class else 0.0,
        "pr_conf": pr_conf,
        "precision": float(precision),
        "recall": float(recall),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "classes_with_gt": len(gt_by_cls),
    }


def run_backend(name, raw_fn, images, imgsz, conf, iou, max_det, max_images=None,
                pr_conf=0.25):
    from ultralytics.utils.ops import non_max_suppression, scale_boxes

    preds, gts = {}, {}
    t0 = time.time()
    files = images[:max_images] if max_images else images
    for i, img_path in enumerate(files):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        label = os.path.join(DATA_ROOT, "labels/train2017", stem + ".txt")
        im0, im_t, input_data = preprocess(img_path, imgsz)
        h, w = im0.shape[:2]
        gts[i] = load_gt(label, w, h)

        raw = raw_fn(input_data, im_t)
        det = non_max_suppression(
            torch.from_numpy(raw).float(),
            conf_thres=conf, iou_thres=iou, max_det=max_det,
        )[0]
        if det is None or len(det) == 0:
            preds[i] = (np.zeros((0, 4), np.float32), np.zeros((0,), np.float32),
                        np.zeros((0,), np.int64))
        else:
            # NMS 输出位于 letterbox 后的输入坐标系，需还原到原图尺寸
            det = det.clone()
            det[:, :4] = scale_boxes(im_t.shape[2:], det[:, :4], im0.shape).round()
            d = det.cpu().numpy()
            preds[i] = (d[:, :4].astype(np.float32), d[:, 4].astype(np.float32),
                        d[:, 5].astype(np.int64))
        if (i + 1) % 20 == 0:
            print(f"  [{name}] {i + 1}/{len(files)} 完成，用时 {time.time() - t0:.1f}s",
                  flush=True)

    levels = [round(0.5 + 0.05 * k, 2) for k in range(10)]
    metrics = eval_detections(preds, gts, levels, pr_conf=pr_conf)
    metrics["elapsed_s"] = round(time.time() - t0, 1)
    metrics["images"] = len(files)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="model_spikeyolov8_320_c16/Net_0")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--pr-conf", type=float, default=0.25,
                    help="操作点 P/R 使用的置信度阈值")
    ap.add_argument("--torch", action="store_true", help="同时评估 PyTorch best.pt")
    ap.add_argument("--out", default="zz_map_results.json")
    args = ap.parse_args()

    images = sorted(glob.glob(os.path.join(DATA_ROOT, "images/train2017", "*.jpg")))
    if not images:
        print(f"未找到图片: {DATA_ROOT}/images/train2017")
        return
    print(f"数据集: {DATA_ROOT}  图片数: {len(images)}")
    print(f"APU 模型: {args.model_dir}  输入: {args.imgsz}  conf={args.conf} iou={args.iou}")

    payload = {"dataset": DATA_ROOT, "images": len(images), "imgsz": args.imgsz,
               "conf": args.conf, "iou_nms": args.iou, "results": {}}

    lyn_model = build_apu_model(args.model_dir)

    def apu_fn(input_data, im_t):
        return apu_raw(lyn_model, input_data)

    print("\n=== APU 评估 ===")
    m = run_backend("APU", apu_fn, images, args.imgsz, args.conf, args.iou,
                    args.max_det, args.max_images, args.pr_conf)
    payload["results"]["apu"] = m
    print(f"  mAP@0.5={m['mAP50']:.4f}  mAP@0.5:0.95={m['mAP50_95']:.4f}  |  "
          f"conf>={m['pr_conf']}: P={m['precision']:.4f} R={m['recall']:.4f} "
          f"(TP={m['tp']} FP={m['fp']} FN={m['fn']})  用时 {m['elapsed_s']}s")

    if args.torch:
        print("\n=== PyTorch 评估 ===")

        def torch_fn(input_data, im_t):
            return torch_raw(im_t)

        mt = run_backend("Torch", torch_fn, images, args.imgsz, args.conf, args.iou,
                         args.max_det, args.max_images, args.pr_conf)
        payload["results"]["torch"] = mt
        print(f"  mAP@0.5={mt['mAP50']:.4f}  mAP@0.5:0.95={mt['mAP50_95']:.4f}  |  "
              f"conf>={mt['pr_conf']}: P={mt['precision']:.4f} R={mt['recall']:.4f} "
              f"(TP={mt['tp']} FP={mt['fp']} FN={mt['fn']})  用时 {mt['elapsed_s']}s")
        print(f"\nAPU 相对 PyTorch 差值: mAP@0.5 {m['mAP50'] - mt['mAP50']:+.4f}  "
              f"mAP@0.5:0.95 {m['mAP50_95'] - mt['mAP50_95']:+.4f}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
