#!/usr/bin/env python3
"""
DA150X Unified Peak Evaluation Script (The "Old Way")
Evaluates BOTH Baseline and ViT architectures using 7x7 Local Maxima.
Sweeps thresholds properly scaled to the 0.03 target maximum.
Includes the / 1000.0 normalization fix to counter the training multiplier.
"""

import argparse
import glob
import os
import sys
import json
import gc
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.ndimage import maximum_filter

from SNN_final_model import SNN
from SNN_ViT_model_all_versions import SNNViT

CLASS_NAMES = ["person", "car", "bus", "truck"]
DATA_DIRS = [
    "week_32-box_3", "week_33-box_2", "week_34-box_1",
    "week_35-box_2", "week_36-box_3",
]

# ═══════════════════════════════════════════════════════════════════════
# PEAK DETECTION (THE OLD WAY)
# ═══════════════════════════════════════════════════════════════════════
def find_peaks(heatmap, threshold, min_distance=3):
    filtered = maximum_filter(heatmap, size=min_distance * 2 + 1)
    peaks_mask = (heatmap == filtered) & (heatmap > threshold)
    ys, xs = np.where(peaks_mask)
    return [(y, x, heatmap[y, x]) for y, x in zip(ys, xs)]

def match_detections(pred_peaks, target_peaks, distance_threshold=5.0):
    if not target_peaks and not pred_peaks: return 0, 0, 0
    if not target_peaks: return 0, len(pred_peaks), 0
    if not pred_peaks: return 0, 0, len(target_peaks)

    pred_coords = np.array([(p[0], p[1]) for p in pred_peaks])
    target_coords = np.array([(t[0], t[1]) for t in target_peaks])

    matched_targets = set()
    tp, fp = 0, 0

    pred_sorted = sorted(range(len(pred_peaks)), key=lambda i: pred_peaks[i][2], reverse=True)

    for pi in pred_sorted:
        py, px = pred_coords[pi]
        best_dist = float("inf")
        best_ti = -1

        for ti in range(len(target_coords)):
            if ti in matched_targets: continue
            ty, tx = target_coords[ti]
            dist = np.sqrt((py - ty) ** 2 + (px - tx) ** 2)
            if dist < best_dist:
                best_dist = dist
                best_ti = ti

        if best_dist <= distance_threshold and best_ti >= 0:
            tp += 1
            matched_targets.add(best_ti)
        else:
            fp += 1

    fn = len(target_peaks) - len(matched_targets)
    return tp, fp, fn

def find_best_checkpoint(model_dir):
    pattern = os.path.join(model_dir, "multiclass-adamw-*.pth")
    checkpoints = glob.glob(pattern)
    if not checkpoints: return None
    best_loss, best_path = float("inf"), None
    for path in checkpoints:
        try:
            val_loss = float(os.path.basename(path).replace(".pth", "").split("-")[-1])
            if val_loss < best_loss: best_loss, best_path = val_loss, path
        except (ValueError, IndexError):
            continue
    return best_path

# ═══════════════════════════════════════════════════════════════════════
# MAIN EVALUATION
# ═══════════════════════════════════════════════════════════════════════
def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Output Directory Setup
    eval_output_dir = os.path.join(args.model_dir, "evaluation_peaks")
    os.makedirs(eval_output_dir, exist_ok=True)

    # 1. Initialize Architecture
    checkpoint_path = args.checkpoint or find_best_checkpoint(args.model_dir)
    if not checkpoint_path: sys.exit(f"ERROR: No checkpoint in {args.model_dir}")
    print(f"Checkpoint: {os.path.basename(checkpoint_path)}")

    if args.baseline:
        print("Loading SNN Baseline...")
        model = SNN()
    else:
        print(f"Loading SNNViT ({args.version})...")
        model = SNNViT(version=args.version)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model = model.to(device).eval()

    # 2. Collect file paths
    data_files = []
    for dname in DATA_DIRS:
        path = os.path.join(args.data_dir, dname)
        if os.path.isdir(path):
            data_files.extend([os.path.join(path, f) for f in os.listdir(path) if f.endswith('.pt')])

    # PROPER THRESHOLDS based on 0.03 Max Target Value
    thresholds = [0.005, 0.010, 0.015, 0.020, 0.025, 0.030]
    metrics = {t: {c: {"tp": 0, "fp": 0, "fn": 0} for c in CLASS_NAMES} for t in thresholds}
    
    total_frames = 0
    total_loss = 0.0
    loss_function = nn.MSELoss()

    for data_idx, pt_path in enumerate(data_files):
        print(f"[{data_idx+1}/{len(data_files)}] Processing: {os.path.basename(pt_path)}")
        tensors = torch.load(pt_path, map_location="cpu", weights_only=False)
        dense_tensors = [t.to_dense() if t.is_sparse else t for t in tensors]
        
        events = dense_tensors[0]
        targets = dense_tensors[1:5]
        n_frames = events.shape[0]

        ev_off, tg_off = 28, 7
        events_cropped = events[:, ev_off:ev_off+200, ev_off:ev_off+200]
        targets_cropped = []
        for tgt in targets:
            cropped = tgt[:, tg_off:tg_off+50, tg_off:tg_off+50].unsqueeze(1).float()
            targets_cropped.append(F.interpolate(cropped, size=(64, 64), mode="bilinear", align_corners=False).squeeze(1))

        mem_states = (None,) * (11 if args.baseline else 3)
        seq_start, sequence_length, step_size = 0, 60, 35

        while seq_start + sequence_length <= n_frames:
            for t in range(seq_start, seq_start + sequence_length):
                frame = events_cropped[t].unsqueeze(0).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    with torch.amp.autocast(device_type="cuda" if "cuda" in str(device) else "cpu"):
                        res = model(frame, mem_states)
                        
                        # =========================================================================
                        # THE CRITICAL FIX: Divide predictions by 1000.0
                        # The training script multiplied targets by 1000.0 before MSE loss.
                        # This forced the model to output values up to ~30.0 instead of ~0.03.
                        # We divide by 1000.0 here to normalize predictions back to the 0.03 scale.
                        # =========================================================================
                        preds_t = [(p.view(64, 64).cpu().float() / 1000.0).numpy() for p in res[:4]]
                        
                        mem_states = res[4]

                tgts_t = [targets_cropped[i][t].numpy() for i in range(4)]

                for thresh in thresholds:
                    for ci, cname in enumerate(CLASS_NAMES):
                        pred_peaks = find_peaks(preds_t[ci], threshold=thresh)
                        target_peaks = find_peaks(tgts_t[ci], threshold=thresh * 0.5)
                        
                        tp, fp, fn = match_detections(pred_peaks, target_peaks)
                        metrics[thresh][cname]["tp"] += tp
                        metrics[thresh][cname]["fp"] += fp
                        metrics[thresh][cname]["fn"] += fn

                total_frames += 1
            seq_start += step_size

        del tensors, dense_tensors, events, targets, events_cropped, targets_cropped
        gc.collect()

    # ═══════════════════════════════════════════════════════════════════
    # RESULTS REPORTING
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "="*60 + "\nPEAK EVALUATION RESULTS\n" + "="*60)
    results_table = {}
    for thresh in thresholds:
        print(f"\n Threshold: {thresh}")
        print(f" {'Class':>10s} | {'TP':>6s} | {'FP':>6s} | {'FN':>6s} | {'Prec':>7s} | {'Recall':>7s} | {'F1':>7s}")
        print("-" * 70)
        thresh_results = {}
        for cname in CLASS_NAMES:
            m = metrics[thresh][cname]
            tp, fp, fn = m["tp"], m["fp"], m["fn"]
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-8)
            print(f" {cname:>10s} | {tp:6d} | {fp:6d} | {fn:6d} | {prec:7.3f} | {rec:7.3f} | {f1:7.3f}")
            thresh_results[cname] = {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1}
        results_table[str(thresh)] = thresh_results

    results = {
        "checkpoint": os.path.basename(checkpoint_path),
        "total_frames": total_frames,
        "detection_metrics": results_table,
    }
    results_path = os.path.join(eval_output_dir, "eval_results_peaks.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults securely saved to: {results_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    parser.add_argument("model_dir")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--version", type=str, default="v2", choices=["v1", "v2", "v3"])
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu)
    evaluate(args)
