#!/usr/bin/env python3
"""
DA150X SNN Evaluation Script — Streaming version (one clip at a time, no OOM)

Usage:
  CUDA_VISIBLE_DEVICES=3 python3 evaluate_model_SNN.py \
      ../data/raw_scaled/ \
      ../data/processed/experiments/snn_baseline_clean \
      --gpu 0 --save_images
"""

import argparse
import glob
import gc
import os
import sys
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from norse.torch import LILinearCell
from norse.torch.module.lif import LIFCell, LIFParameters

from SNN_ViT_model_v2 import SNNViT

try:
    from scipy.ndimage import maximum_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("WARNING: scipy not found. Using simple peak detection.")

# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

data_dirs = [
    "week_32-box_3",
    "week_33-box_2",
    "week_34-box_1",
    "week_35-box_2",
    "week_36-box_3",
]

CLASS_NAMES = ["person", "car", "bus", "truck"]

# ═══════════════════════════════════════════════════════════════════════
# PEAK DETECTION & MATCHING
# ═══════════════════════════════════════════════════════════════════════

def find_peaks(heatmap, threshold, min_distance=3):
    if HAS_SCIPY:
        filtered = maximum_filter(heatmap, size=min_distance * 2 + 1)
        peaks_mask = (heatmap == filtered) & (heatmap > threshold)
        ys, xs = np.where(peaks_mask)
        return [(y, x, heatmap[y, x]) for y, x in zip(ys, xs)]
    else:
        ys, xs = np.where(heatmap > threshold)
        return [(y, x, heatmap[y, x]) for y, x in zip(ys, xs)]


def match_detections(pred_peaks, target_peaks, distance_threshold=5.0):
    tp, fp, fn = 0, 0, 0
    matched_targets = set()
    for py, px, _ in pred_peaks:
        best_dist = float("inf")
        best_idx = -1
        for i, (ty, tx, _) in enumerate(target_peaks):
            if i in matched_targets:
                continue
            dist = ((py - ty) ** 2 + (px - tx) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        if best_idx >= 0 and best_dist <= distance_threshold:
            tp += 1
            matched_targets.add(best_idx)
        else:
            fp += 1
    fn = len(target_peaks) - len(matched_targets)
    return tp, fp, fn


# ═══════════════════════════════════════════════════════════════════════
# CHECKPOINT FINDER
# ═══════════════════════════════════════════════════════════════════════

def find_best_checkpoint(model_dir):
    pattern = os.path.join(model_dir, "multiclass-adamw-*.pth")
    checkpoints = glob.glob(pattern)
    if not checkpoints:
        return None
    best_loss = float("inf")
    best_path = None
    for path in checkpoints:
        basename = os.path.basename(path)
        parts = basename.replace(".pth", "").split("-")
        try:
            val_loss = float(parts[-1])
            if val_loss < best_loss:
                best_loss = val_loss
                best_path = path
        except (ValueError, IndexError):
            continue
    return best_path


# ═══════════════════════════════════════════════════════════════════════
# VISUALISATION
# ═══════════════════════════════════════════════════════════════════════

def save_comparison_image(event_frame, targets, predictions, frame_idx,
                          output_dir, vmax_target=0.03):
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    for i, name in enumerate(CLASS_NAMES):
        axes[0, i].imshow(targets[i], cmap="magma", vmin=0, vmax=vmax_target)
        axes[0, i].set_title(f"Target: {name}", fontsize=10)
        axes[0, i].axis("off")
        pred_data = predictions[i]
        axes[1, i].imshow(pred_data, cmap="magma", vmin=0,
                          vmax=max(pred_data.max(), 1e-6))
        axes[1, i].set_title(f"Pred: {name} (max={pred_data.max():.4f})", fontsize=10)
        axes[1, i].axis("off")
    for i in range(4):
        axes[2, i].axis("off")
    axes[2, 0].imshow(event_frame, cmap="gray")
    axes[2, 0].set_title(f"Event Frame (frame {frame_idx})", fontsize=10)
    for i in range(1, 4):
        axes[2, i].set_visible(False)
    fig.suptitle(f"Frame {frame_idx}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"eval_frame_{frame_idx:05d}.png"), dpi=100)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# CLIP ITERATOR — loads one clip at a time, frees after use
# ═══════════════════════════════════════════════════════════════════════

def iter_clips(data_root):
    """Yield (name, dense_tensors) one clip at a time. Frees memory after each."""
    for dname in data_dirs:
        dir_path = os.path.join(data_root, dname)
        if not os.path.isdir(dir_path):
            continue
        pt_files = sorted([f for f in os.listdir(dir_path) if f.endswith('.pt')])
        for pt_file in pt_files:
            pt_path = os.path.join(dir_path, pt_file)
            print(f"  Loading {dname}/{pt_file}...", flush=True)
            tensors = torch.load(pt_path, map_location="cpu")
            dense = []
            for t in tensors:
                if t.is_sparse:
                    t = t.to_dense()
                dense.append(t.float())
            yield (f"{dname}_{pt_file}", dense)
            del tensors, dense
            gc.collect()


def count_clips(data_root):
    total = 0
    for dname in data_dirs:
        dir_path = os.path.join(data_root, dname)
        if not os.path.isdir(dir_path):
            continue
        total += len([f for f in os.listdir(dir_path) if f.endswith('.pt')])
    return total


# ═══════════════════════════════════════════════════════════════════════
# MAIN EVALUATION
# ═══════════════════════════════════════════════════════════════════════

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Find checkpoint ──
    checkpoint_path = args.checkpoint or find_best_checkpoint(args.model_dir)
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        print(f"ERROR: No checkpoint found in {args.model_dir}")
        sys.exit(1)
    print(f"Checkpoint: {checkpoint_path}")

    # ── Load model ──
    model = SNNViT(model_version=args.version)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device,
                                     weights_only=True))
    model = model.to(device)
    model.eval()
    print("Model loaded.")

    # ── Output dirs ──
    eval_output_dir = os.path.join(args.model_dir, "evaluation")
    os.makedirs(eval_output_dir, exist_ok=True)
    img_dir = None
    if args.save_images:
        img_dir = os.path.join(eval_output_dir, "images")
        os.makedirs(img_dir, exist_ok=True)

    # ── Setup metrics ──
    sequence_length = 60
    overlap = 25
    step_size = sequence_length - overlap  # 35
    thresholds = [0.001, 0.005, 0.01, 0.02, 0.03, 0.05]

    metrics = {t: {c: {"tp": 0, "fp": 0, "fn": 0} for c in CLASS_NAMES}
               for t in thresholds}
    pred_stats = {c: {"max_vals": [], "mean_vals": [], "nonzero_frac": []}
                  for c in CLASS_NAMES}

    total_frames = 0
    loss_function = nn.MSELoss()
    total_loss = 0.0
    per_class_loss = {c: 0.0 for c in CLASS_NAMES}

    n_clips = count_clips(args.data_root if hasattr(args, 'data_root') else args.data_dir)
    print(f"\nFound {n_clips} clips to evaluate.")
    print(f"Thresholds: {thresholds}\n")

    # ── Stream through clips one at a time ──
    for clip_idx, (dname, tensors) in enumerate(iter_clips(args.data_dir)):
        events = tensors[0]      # [5400, 256, 256]
        targets_list = tensors[1:5]  # person, car, bus, truck each [5400, 64, 64]

        n_frames = events.shape[0]
        print(f"  [{clip_idx+1}/{n_clips}] {dname}: {n_frames} frames", flush=True)

        # Center crop events 256→200
        ev_off = (256 - 200) // 2  # 28
        tg_off = (64 - 50) // 2    # 7
        events_cropped = events[:, ev_off:ev_off+200, ev_off:ev_off+200]

        targets_cropped = []
        for tgt in targets_list:
            cropped = tgt[:, tg_off:tg_off+50, tg_off:tg_off+50]
            cropped_4d = cropped.unsqueeze(1)
            resized = F.interpolate(cropped_4d, size=(64, 64), mode="bilinear",
                                    align_corners=False)
            targets_cropped.append(resized.squeeze(1))

        # Process sequences
        mem_states = (None, None, None)
        seq_start = 0

        while seq_start + sequence_length <= n_frames:
            for t in range(seq_start, seq_start + sequence_length):
                frame = events_cropped[t].unsqueeze(0).unsqueeze(0).to(device)

                with torch.no_grad():
                    with torch.amp.autocast(
                            device_type="cuda" if "cuda" in str(device) else "cpu"):
                        p1, p2, p3, p4, mem_states = model(frame, mem_states)

                preds = [
                    p1.view(64, 64).cpu().float().numpy(),
                    p2.view(64, 64).cpu().float().numpy(),
                    p3.view(64, 64).cpu().float().numpy(),
                    p4.view(64, 64).cpu().float().numpy(),
                ]
                tgts = [targets_cropped[i][t].numpy() for i in range(4)]

                # Loss
                for ci, cname in enumerate(CLASS_NAMES):
                    closs = loss_function(torch.tensor(preds[ci]),
                                         torch.tensor(tgts[ci])).item()
                    per_class_loss[cname] += closs
                    total_loss += closs

                # Prediction stats
                for ci, cname in enumerate(CLASS_NAMES):
                    pred_stats[cname]["max_vals"].append(float(preds[ci].max()))
                    pred_stats[cname]["mean_vals"].append(float(preds[ci].mean()))
                    nz = float((np.abs(preds[ci]) > 1e-6).sum()) / (64 * 64)
                    pred_stats[cname]["nonzero_frac"].append(nz)

                # Detection metrics
                for thresh in thresholds:
                    for ci, cname in enumerate(CLASS_NAMES):
                        pred_peaks = find_peaks(preds[ci], threshold=thresh)
                        tgt_peaks = find_peaks(tgts[ci], threshold=thresh * 0.5)
                        tp, fp, fn = match_detections(pred_peaks, tgt_peaks,
                                                      distance_threshold=5.0)
                        metrics[thresh][cname]["tp"] += tp
                        metrics[thresh][cname]["fp"] += fp
                        metrics[thresh][cname]["fn"] += fn

                # Save images
                if args.save_images and img_dir and (total_frames % args.image_every == 0):
                    save_comparison_image(
                        events_cropped[t].numpy(), tgts, preds, total_frames,
                        img_dir, vmax_target=0.03
                    )

                total_frames += 1

            seq_start += step_size

        # Free clip memory immediately
        del tensors, events, targets_list, events_cropped, targets_cropped
        gc.collect()

    # ═══════════════════════════════════════════════════════════════════
    # RESULTS
    # ═══════════════════════════════════════════════════════════════════

    print(f"\n{'='*70}")
    print("EVALUATION RESULTS")
    print(f"{'='*70}")
    print(f"Checkpoint: {os.path.basename(checkpoint_path)}")
    print(f"Total frames: {total_frames}")

    avg_loss = total_loss / max(total_frames, 1)
    print(f"\nAverage MSE Loss: {avg_loss:.4f}")
    for cname in CLASS_NAMES:
        cl = per_class_loss[cname] / max(total_frames, 1)
        print(f"  {cname:8s}: {cl:.4f}")

    print(f"\n--- Prediction Statistics ---")
    print(f"{'Class':>10s} | {'Max(mean)':>10s} | {'Max(max)':>10s} | {'Nonzero%':>9s}")
    print(f"{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*9}")
    for cname in CLASS_NAMES:
        maxes = pred_stats[cname]["max_vals"]
        nzs = pred_stats[cname]["nonzero_frac"]
        print(f"{cname:>10s} | {np.mean(maxes):10.6f} | {np.max(maxes):10.6f} | "
              f"{np.mean(nzs)*100:8.2f}%")

    print(f"\n--- Detection Metrics (F1 / Precision / Recall) ---")
    results_table = {}
    for thresh in thresholds:
        print(f"\n  Threshold: {thresh}")
        print(f"  {'Class':>10s} | {'TP':>6s} | {'FP':>6s} | {'FN':>6s} | "
              f"{'Prec':>7s} | {'Recall':>7s} | {'F1':>7s}")
        print(f"  {'-'*10}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-"
              f"{'-'*7}-+-{'-'*7}-+-{'-'*7}")
        thresh_results = {}
        for cname in CLASS_NAMES:
            m = metrics[thresh][cname]
            tp, fp, fn = m["tp"], m["fp"], m["fn"]
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-8)
            print(f"  {cname:>10s} | {tp:6d} | {fp:6d} | {fn:6d} | "
                  f"{prec:7.3f} | {rec:7.3f} | {f1:7.3f}")
            thresh_results[cname] = {"tp": tp, "fp": fp, "fn": fn,
                                     "precision": prec, "recall": rec, "f1": f1}
        results_table[str(thresh)] = thresh_results

    # Save JSON
    results = {
        "checkpoint": os.path.basename(checkpoint_path),
        "total_frames": total_frames,
        "avg_loss": avg_loss,
        "per_class_loss": {c: per_class_loss[c] / max(total_frames, 1)
                           for c in CLASS_NAMES},
        "prediction_stats": {
            c: {
                "max_mean": float(np.mean(pred_stats[c]["max_vals"])),
                "max_max": float(np.max(pred_stats[c]["max_vals"])),
                "mean_mean": float(np.mean(pred_stats[c]["mean_vals"])),
                "nonzero_pct": float(np.mean(pred_stats[c]["nonzero_frac"]) * 100),
            }
            for c in CLASS_NAMES
        },
        "detection_metrics": results_table,
    }
    results_path = os.path.join(eval_output_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    if args.save_images:
        print(f"Images saved to: {img_dir}")
    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="DA150X SNN Evaluation (streaming)")
    parser.add_argument("data_dir", help="Root of raw_scaled/ directory")
    parser.add_argument("model_dir", help="Experiment directory with checkpoints")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--save_images", action="store_true")
    parser.add_argument("--image_every", type=int, default=200)
    parser.add_argument("--version", type=str, default="v1",
                    choices=["v1", "v2.1", "v2.2"])
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu)
    evaluate(args)


if __name__ == "__main__":
    main()
