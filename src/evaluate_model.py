#!/usr/bin/env python3
"""
DA150X SNN Evaluation Script
Loads a trained checkpoint, runs inference on validation data, computes
detection metrics (F1, precision, recall), and saves visual comparisons
of predicted vs target heatmaps.

Usage:
  # Evaluate best checkpoint from Run 2:
  CUDA_VISIBLE_DEVICES=2 python3 evaluate_model.py \
      data/training_output_scaled/ \
      data/model_output/scaled/3-11-15-30/ \
      --gpu 0

  # Evaluate Run 3 instead:
  CUDA_VISIBLE_DEVICES=2 python3 evaluate_model.py \
      data/training_output_scaled/ \
      data/model_output/scaled/3-11-16-5/ \
      --gpu 0

  # Specify a particular checkpoint:
  CUDA_VISIBLE_DEVICES=2 python3 evaluate_model.py \
      data/training_output_scaled/ \
      data/model_output/scaled/3-11-15-30/ \
      --gpu 0 --checkpoint data/model_output/scaled/3-11-15-30/multiclass-adamw-30-97.3751.pth

  # Also save a video of predictions:
  CUDA_VISIBLE_DEVICES=2 python3 evaluate_model.py \
      data/training_output_scaled/ \
      data/model_output/scaled/3-11-15-30/ \
      --gpu 0 --save_video
"""

import argparse
import glob
import os
import sys
import time
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

# Try importing scipy for peak detection; fallback to simple version if missing
try:
    from scipy.ndimage import maximum_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("WARNING: scipy not found. Using simple peak detection (less accurate).")
    print("         Install with: pip install scipy --break-system-packages")


# ═══════════════════════════════════════════════════════════════════════
# MODEL DEFINITION — exact copy from SNN_final_model.py
# Must match the architecture used during training.
#
# Architecture: 3 conv layers (8 channels each) -> flatten (3200) ->
#               4 parallel FC heads (person, car, bus, truck)
#               each: 3200 -> 500 -> 500 -> 4096 (= 64x64 heatmap)
# ═══════════════════════════════════════════════════════════════════════

tau_mem = 180
layer_nr = 500


class SNN(nn.Module):
    def __init__(self):
        super(SNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=7, stride=2, padding=0)
        self.bn1 = nn.BatchNorm2d(8)
        self.lif1 = LIFCell(p=LIFParameters(tau_mem_inv=tau_mem))

        self.conv2 = nn.Conv2d(8, 8, kernel_size=5, stride=2, padding=0)
        self.bn2 = nn.BatchNorm2d(8)
        self.lif2 = LIFCell(p=LIFParameters(tau_mem_inv=tau_mem))

        self.conv3 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(8)
        self.lif3 = LIFCell(p=LIFParameters(tau_mem_inv=tau_mem))

        self.lif4 = LIFCell(p=LIFParameters(tau_mem_inv=tau_mem))
        self.lif5 = LIFCell(p=LIFParameters(tau_mem_inv=tau_mem))
        self.lif6 = LIFCell(p=LIFParameters(tau_mem_inv=tau_mem))
        self.lif7 = LIFCell(p=LIFParameters(tau_mem_inv=tau_mem))

        self.fcperson1 = nn.Linear(3200, layer_nr)
        self.fcperson2 = nn.Linear(layer_nr, layer_nr)
        self.lifperson = LILinearCell(layer_nr, 4096)

        self.fccar1 = nn.Linear(3200, layer_nr)
        self.fccar2 = nn.Linear(layer_nr, layer_nr)
        self.lifcar = LILinearCell(layer_nr, 4096)

        self.fcbus1 = nn.Linear(3200, layer_nr)
        self.fcbus2 = nn.Linear(layer_nr, layer_nr)
        self.lifbus = LILinearCell(layer_nr, 4096)

        self.fctruck1 = nn.Linear(3200, layer_nr)
        self.fctruck2 = nn.Linear(layer_nr, layer_nr)
        self.liftruck = LILinearCell(layer_nr, 4096)

        self.maxpool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.3)

    def forward(self, x, mem_states):
        batch_size, C, W, H = x.shape
        x = (x != 0).float()

        (mem1, mem2, mem3,
         mem5_1, mem5_2, mem6_1, mem6_2,
         mem7_1, mem7_2, mem8_1, mem8_2) = mem_states

        v1 = self.bn1(self.conv1(x))
        spk1, mem1 = self.lif1(v1, mem1)

        v2 = self.dropout1(self.bn2(self.conv2(self.maxpool(spk1))))
        spk2, mem2 = self.lif2(v2, mem2)

        v3 = self.dropout1(self.bn3(self.conv3(spk2)))
        spk3, mem3 = self.lif3(v3, mem3)

        spk3_flat = spk3.view(batch_size, -1)

        v5 = self.dropout2(self.fcperson1(spk3_flat))
        spk5_1, mem5_1 = self.lif4(v5, mem5_1)
        v5 = self.dropout2(self.fcperson2(spk5_1))
        spk5_2, mem5_2 = self.lifperson(v5, mem5_2)

        v6 = self.dropout2(self.fccar1(spk3_flat))
        spk6_1, mem6_1 = self.lif5(v6, mem6_1)
        v6 = self.dropout2(self.fccar2(spk6_1))
        spk6_2, mem6_2 = self.lifcar(v6, mem6_2)

        v7 = self.dropout2(self.fcbus1(spk3_flat))
        spk7_1, mem7_1 = self.lif6(v7, mem7_1)
        v7 = self.dropout2(self.fcbus2(spk7_1))
        spk7_2, mem7_2 = self.lifbus(v7, mem7_2)

        v8 = self.dropout2(self.fctruck1(spk3_flat))
        spk8_1, mem8_1 = self.lif7(v8, mem8_1)
        v8 = self.dropout2(self.fctruck2(spk8_1))
        spk8_2, mem8_2 = self.liftruck(v8, mem8_2)

        return (
            spk5_2, spk6_2, spk7_2, spk8_2,
            (mem1, mem2, mem3,
             mem5_1, mem5_2, mem6_1, mem6_2,
             mem7_1, mem7_2, mem8_1, mem8_2),
        )


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING — mirrors the logic from SNN_final_model.py / data_loading.py
# ═══════════════════════════════════════════════════════════════════════

data_dirs = [
    "week_32-box_3",
    "week_33-box_2",
    "week_34-box_1",
    "week_35-box_2",
    "week_36-box_3",
]

CLASS_NAMES = ["person", "car", "bus", "truck"]


def load_data(data_root):
    """
    Load all .pt training files.
    Each .pt file contains a list of 5 sparse tensors:
      [0] event frames  [5400, 256, 256]
      [1] person heatmaps [5400, 64, 64]
      [2] car heatmaps    [5400, 64, 64]
      [3] bus heatmaps    [5400, 64, 64]
      [4] truck heatmaps  [5400, 64, 64]
    """
    all_data = []
    for dname in data_dirs:
        pt_path = os.path.join(data_root, dname, "clip_000.pt")
        if not os.path.isfile(pt_path):
            print(f"  WARNING: {pt_path} not found, skipping.")
            continue
        print(f"  Loading {dname}...")
        tensors = torch.load(pt_path, map_location="cpu")
        dense = []
        for t in tensors:
            if t.is_sparse:
                t = t.to_dense()
            dense.append(t)
        all_data.append((dname, dense))
    return all_data


# ═══════════════════════════════════════════════════════════════════════
# PEAK DETECTION & METRICS
# ═══════════════════════════════════════════════════════════════════════

def find_peaks(heatmap, threshold, min_distance=3):
    """
    Find local maxima in a 2D heatmap above a threshold.
    Returns list of (y, x, value) tuples.
    """
    if HAS_SCIPY:
        filtered = maximum_filter(heatmap, size=min_distance * 2 + 1)
        peaks_mask = (heatmap == filtered) & (heatmap > threshold)
        ys, xs = np.where(peaks_mask)
        return [(y, x, heatmap[y, x]) for y, x in zip(ys, xs)]
    else:
        # Simple fallback: just threshold
        ys, xs = np.where(heatmap > threshold)
        return [(y, x, heatmap[y, x]) for y, x in zip(ys, xs)]


def match_detections(pred_peaks, target_peaks, distance_threshold=5.0):
    """
    Match predicted peaks to target peaks greedily by distance.
    Returns (TP, FP, FN).
    """
    if len(target_peaks) == 0 and len(pred_peaks) == 0:
        return 0, 0, 0
    if len(target_peaks) == 0:
        return 0, len(pred_peaks), 0
    if len(pred_peaks) == 0:
        return 0, 0, len(target_peaks)

    pred_coords = np.array([(p[0], p[1]) for p in pred_peaks])
    target_coords = np.array([(t[0], t[1]) for t in target_peaks])

    matched_targets = set()
    tp, fp = 0, 0

    # Sort predictions by confidence (highest first)
    pred_sorted = sorted(range(len(pred_peaks)),
                         key=lambda i: pred_peaks[i][2], reverse=True)

    for pi in pred_sorted:
        py, px = pred_coords[pi]
        best_dist = float("inf")
        best_ti = -1

        for ti in range(len(target_coords)):
            if ti in matched_targets:
                continue
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


# ═══════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════

def save_comparison_image(event_frame, targets, predictions, frame_idx,
                          output_dir, vmax_target=0.03):
    """
    Save a figure with 3 rows x 4 columns:
      Row 1: Target heatmaps (person, car, bus, truck)
      Row 2: Predicted heatmaps
      Row 3: Event frame (spanning all columns)
    """
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))

    for i, name in enumerate(CLASS_NAMES):
        # Target
        axes[0, i].imshow(targets[i], cmap="magma", vmin=0, vmax=vmax_target)
        axes[0, i].set_title(f"Target: {name}", fontsize=10)
        axes[0, i].axis("off")

        # Prediction
        pred_data = predictions[i]
        im = axes[1, i].imshow(pred_data, cmap="magma", vmin=0,
                                vmax=max(pred_data.max(), 1e-6))
        axes[1, i].set_title(f"Pred: {name} (max={pred_data.max():.4f})", fontsize=10)
        axes[1, i].axis("off")

    # Event frame across bottom row
    for i in range(4):
        axes[2, i].axis("off")
    # Merge bottom row into one image
    axes[2, 0].imshow(event_frame, cmap="gray")
    axes[2, 0].set_title(f"Event Frame (frame {frame_idx})", fontsize=10)
    for i in range(1, 4):
        axes[2, i].set_visible(False)

    fig.suptitle(f"Frame {frame_idx}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"eval_frame_{frame_idx:05d}.png"), dpi=100)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# FIND BEST CHECKPOINT
# ═══════════════════════════════════════════════════════════════════════

def find_best_checkpoint(model_dir):
    """
    Find the checkpoint with the lowest validation loss.
    Filenames are: multiclass-adamw-<epoch>-<val_loss>.pth
    """
    pattern = os.path.join(model_dir, "multiclass-adamw-*.pth")
    checkpoints = glob.glob(pattern)
    if not checkpoints:
        return None

    best_loss = float("inf")
    best_path = None
    for path in checkpoints:
        # Extract val_loss from filename
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
# MAIN EVALUATION LOOP
# ═══════════════════════════════════════════════════════════════════════

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Find checkpoint ──
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = find_best_checkpoint(args.model_dir)
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        print(f"ERROR: No checkpoint found in {args.model_dir}")
        sys.exit(1)
    print(f"Checkpoint: {checkpoint_path}")

    # ── Load model ──
    model = SNN()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    print("Model loaded and set to eval mode.")

    # ── Load data ──
    print(f"\nLoading data from {args.data_dir}...")
    all_data = load_data(args.data_dir)
    if not all_data:
        print("ERROR: No data loaded.")
        sys.exit(1)

    # ── Output directory ──
    eval_output_dir = os.path.join(args.model_dir, "evaluation")
    os.makedirs(eval_output_dir, exist_ok=True)
    if args.save_images:
        img_dir = os.path.join(eval_output_dir, "images")
        os.makedirs(img_dir, exist_ok=True)

    # ── Run inference ──
    sequence_length = 60
    overlap = 25
    step_size = sequence_length - overlap  # 35

    # Detection thresholds to evaluate
    # The targets peak at 0.03, so we try a range of thresholds on predictions
    thresholds = [0.001, 0.005, 0.01, 0.02, 0.03, 0.05]

    # Per-threshold, per-class counters
    metrics = {t: {c: {"tp": 0, "fp": 0, "fn": 0} for c in CLASS_NAMES}
               for t in thresholds}

    # Also track raw prediction statistics
    pred_stats = {c: {"max_vals": [], "mean_vals": [], "nonzero_frac": []}
                  for c in CLASS_NAMES}

    total_frames = 0
    loss_function = nn.MSELoss()
    total_loss = 0.0
    per_class_loss = {c: 0.0 for c in CLASS_NAMES}

    print(f"\nRunning evaluation...")
    print(f"  Sequence length: {sequence_length}, overlap: {overlap}")
    print(f"  Thresholds: {thresholds}")
    print()

    for data_idx, (dname, tensors) in enumerate(all_data):
        events = tensors[0]      # [5400, 256, 256]
        targets = tensors[1:5]   # [person, car, bus, truck], each [5400, 64, 64]

        n_frames = events.shape[0]
        print(f"  [{data_idx+1}/{len(all_data)}] {dname}: {n_frames} frames")

        # ── Center crop to match training ──
        # Training does random 200x200 crops of events and 50x50 crops of
        # targets (then resized back to 64x64). For eval we center crop.
        # Events: 256 -> 200 (offset 28)
        # Targets: 64 -> 50 (offset 7), then resize to 64
        ev_off = (256 - 200) // 2  # = 28
        tg_off = (64 - 50) // 2    # = 7
        events_cropped = events[:, ev_off:ev_off+200, ev_off:ev_off+200]

        targets_cropped = []
        for tgt in targets:
            cropped = tgt[:, tg_off:tg_off+50, tg_off:tg_off+50]  # [5400, 50, 50]
            # Resize back to 64x64 using bilinear interpolation
            cropped_4d = cropped.unsqueeze(1).float()  # [5400, 1, 50, 50]
            resized = F.interpolate(cropped_4d, size=(64, 64), mode="bilinear",
                                    align_corners=False)
            targets_cropped.append(resized.squeeze(1))  # [5400, 64, 64]

        # Process in sequences (same way training does)
        mem_states = tuple([None] * 11)

        seq_start = 0
        frame_count = 0

        while seq_start + sequence_length <= n_frames:
            # Get sequence of frames
            for t in range(seq_start, seq_start + sequence_length):
                frame = events_cropped[t].unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, 200, 200]

                with torch.no_grad():
                    with torch.amp.autocast(device_type="cuda"):
                        pred_person, pred_car, pred_bus, pred_truck, mem_states = model(frame, mem_states)

                # Reshape predictions to [64, 64]
                preds = [
                    pred_person.view(64, 64).cpu().float().numpy(),
                    pred_car.view(64, 64).cpu().float().numpy(),
                    pred_bus.view(64, 64).cpu().float().numpy(),
                    pred_truck.view(64, 64).cpu().float().numpy(),
                ]

                tgts = [targets_cropped[i][t].numpy() for i in range(4)]

                # Compute loss
                for ci, cname in enumerate(CLASS_NAMES):
                    pred_tensor = torch.tensor(preds[ci])
                    tgt_tensor = torch.tensor(tgts[ci])
                    closs = loss_function(pred_tensor, tgt_tensor).item()
                    per_class_loss[cname] += closs
                    total_loss += closs

                # Track prediction statistics
                for ci, cname in enumerate(CLASS_NAMES):
                    pred_stats[cname]["max_vals"].append(float(preds[ci].max()))
                    pred_stats[cname]["mean_vals"].append(float(preds[ci].mean()))
                    nz = float((np.abs(preds[ci]) > 1e-6).sum()) / (64 * 64)
                    pred_stats[cname]["nonzero_frac"].append(nz)

                # Peak detection & matching at each threshold
                for thresh in thresholds:
                    for ci, cname in enumerate(CLASS_NAMES):
                        pred_peaks = find_peaks(preds[ci], threshold=thresh)
                        target_peaks = find_peaks(tgts[ci], threshold=thresh * 0.5)
                        tp, fp, fn = match_detections(pred_peaks, target_peaks,
                                                       distance_threshold=5.0)
                        metrics[thresh][cname]["tp"] += tp
                        metrics[thresh][cname]["fp"] += fp
                        metrics[thresh][cname]["fn"] += fn

                total_frames += 1

                # Save sample images (every Nth frame, and first/last of each sequence)
                if args.save_images and (total_frames % args.image_every == 0):
                    save_comparison_image(
                        events_cropped[t].numpy(), tgts, preds, t,
                        img_dir, vmax_target=0.03
                    )

            seq_start += step_size
            frame_count += sequence_length

        print(f"    Processed {frame_count} frames in sequences")

    # ═══════════════════════════════════════════════════════════════════
    # RESULTS
    # ═══════════════════════════════════════════════════════════════════

    print(f"\n{'='*70}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*70}")
    print(f"Checkpoint: {os.path.basename(checkpoint_path)}")
    print(f"Total frames evaluated: {total_frames}")

    # ── Loss ──
    avg_loss = total_loss / max(total_frames, 1)
    print(f"\nAverage MSE Loss: {avg_loss:.4f}")
    for cname in CLASS_NAMES:
        cl = per_class_loss[cname] / max(total_frames, 1)
        print(f"  {cname:8s}: {cl:.4f}")

    # ── Prediction statistics (most important diagnostic!) ──
    print(f"\n--- Prediction Output Statistics ---")
    print(f"{'Class':>10s} | {'Max (mean)':>12s} | {'Max (max)':>12s} | {'Mean (mean)':>12s} | {'Nonzero %':>10s}")
    print(f"{'-'*10}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}")
    for cname in CLASS_NAMES:
        maxes = pred_stats[cname]["max_vals"]
        means = pred_stats[cname]["mean_vals"]
        nzs = pred_stats[cname]["nonzero_frac"]
        print(f"{cname:>10s} | {np.mean(maxes):12.6f} | {np.max(maxes):12.6f} | "
              f"{np.mean(means):12.6f} | {np.mean(nzs)*100:9.2f}%")

    print(f"\n  (Target heatmaps peak at ~0.03. If prediction max is much smaller")
    print(f"   or much larger, the model is not producing useful heatmaps.)")

    # ── Detection metrics per threshold ──
    print(f"\n--- Detection Metrics (F1 / Precision / Recall) ---")
    results_table = {}

    for thresh in thresholds:
        print(f"\n  Threshold: {thresh}")
        print(f"  {'Class':>10s} | {'TP':>6s} | {'FP':>6s} | {'FN':>6s} | {'Prec':>7s} | {'Recall':>7s} | {'F1':>7s}")
        print(f"  {'-'*10}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}")

        thresh_results = {}
        for cname in CLASS_NAMES:
            m = metrics[thresh][cname]
            tp, fp, fn = m["tp"], m["fp"], m["fn"]
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-8)
            print(f"  {cname:>10s} | {tp:6d} | {fp:6d} | {fn:6d} | {prec:7.3f} | {rec:7.3f} | {f1:7.3f}")
            thresh_results[cname] = {"tp": tp, "fp": fp, "fn": fn,
                                      "precision": prec, "recall": rec, "f1": f1}
        results_table[str(thresh)] = thresh_results

    # ── Save results to JSON ──
    results = {
        "checkpoint": os.path.basename(checkpoint_path),
        "total_frames": total_frames,
        "avg_loss": avg_loss,
        "per_class_loss": {c: per_class_loss[c] / max(total_frames, 1) for c in CLASS_NAMES},
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
        print(f"Sample images saved to: {img_dir}")

    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description="DA150X SNN Evaluation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_dir",
                        help="Path to training data (e.g. data/training_output_scaled/)")
    parser.add_argument("model_dir",
                        help="Path to model output dir (e.g. data/model_output/scaled/3-11-15-30/)")
    parser.add_argument("--gpu", type=int, default=0,
                        help="CUDA device ID (after CUDA_VISIBLE_DEVICES mapping)")
    parser.add_argument("--checkpoint", default=None,
                        help="Specific .pth file to evaluate (default: auto-find best)")
    parser.add_argument("--save_images", action="store_true",
                        help="Save visual comparison images (pred vs target)")
    parser.add_argument("--image_every", type=int, default=200,
                        help="Save an image every N frames (default: 200)")
    parser.add_argument("--save_video", action="store_true",
                        help="Save a video of predictions (requires ffmpeg)")

    args = parser.parse_args()
    torch.cuda.set_device(args.gpu)

    evaluate(args)


if __name__ == "__main__":
    main()
