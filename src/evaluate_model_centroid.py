#!/usr/bin/env python3
"""
DA150X SNN-ViT Centroid-Based Evaluation Script

This script evaluates the Hybrid SNN-ViT model using a Connected Components 
(Center of Mass) approach rather than strict local maxima. This provides a 
much fairer evaluation for models with global receptive fields (like ViTs) 
that output broad, multi-pixel attention blobs for large objects like trucks.

Usage:
  CUDA_VISIBLE_DEVICES=0 python3 evaluate_model_centroid.py \
      data/training_output_scaled/ \
      data/processed/experiments/<run_dir>/ \
      --version v3
"""

import argparse
import glob
import os
import sys
import json
import numpy as np
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.ndimage import label, center_of_mass, gaussian_filter

# Import your model (ensure SNN_ViT_model_all_versions.py is in the same dir)
from SNN_ViT_model_all_versions import SNNViT
from SNN_final_model import SNN

CLASS_NAMES = ["person", "car", "bus", "truck"]
DATA_DIRS = [
    "week_32-box_3",
    "week_33-box_2",
    "week_34-box_1",
    "week_35-box_2",
    "week_36-box_3",
]

# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

def load_data(data_root):
    all_data = []
    for dname in DATA_DIRS:
        dir_path = os.path.join(data_root, dname)
        if not os.path.isdir(dir_path):
            continue
            
        pt_files = [f for f in os.listdir(dir_path) if f.endswith('.pt')]
        for pt_file in pt_files:
            pt_path = os.path.join(dir_path, pt_file)
            print(f"  Loading {dname}/{pt_file}...")
            tensors = torch.load(pt_path, map_location="cpu", weights_only=False)
            
            # Convert sparse to dense
            dense = [t.to_dense() if t.is_sparse else t for t in tensors]
            all_data.append((f"{dname}_{pt_file}", dense))
    return all_data

# ═══════════════════════════════════════════════════════════════════════
# CENTROID DETECTION (THE FIX)
# ═══════════════════════════════════════════════════════════════════════

def find_centroids(heatmap, threshold):
    """
    Finds the center of mass of connected blobs above a threshold.
    Returns a list of (y, x, max_val_in_blob) tuples.
    """
    # 1. Create binary mask of activations
    mask = heatmap >= threshold
    
    # 2. Label connected components (blobs)
    labeled_array, num_features = label(mask)
    
    centroids = []
    for i in range(1, num_features + 1):
        # Find exact center of mass for the blob
        y, x = center_of_mass(heatmap, labeled_array, i)
        
        # Extract the highest confidence value inside this specific blob
        blob_mask = (labeled_array == i)
        max_val = np.max(heatmap[blob_mask])
        
        centroids.append((y, x, max_val))
        
    return centroids

def match_detections(pred_centroids, target_centroids, distance_threshold=5.0):
    """Greedy matching of predicted centroids to target centroids."""
    if not target_centroids and not pred_centroids: return 0, 0, 0
    if not target_centroids: return 0, len(pred_centroids), 0
    if not pred_centroids: return 0, 0, len(target_centroids)

    pred_coords = np.array([(p[0], p[1]) for p in pred_centroids])
    target_coords = np.array([(t[0], t[1]) for t in target_centroids])

    matched_targets = set()
    tp, fp = 0, 0

    # Sort predictions by confidence (highest first)
    pred_sorted = sorted(range(len(pred_centroids)),
                         key=lambda i: pred_centroids[i][2], reverse=True)

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

    fn = len(target_centroids) - len(matched_targets)
    return tp, fp, fn

def find_best_checkpoint(model_dir):
    pattern = os.path.join(model_dir, "multiclass-adamw-*.pth")
    checkpoints = glob.glob(pattern)
    if not checkpoints: return None

    best_loss, best_path = float("inf"), None
    for path in checkpoints:
        try:
            val_loss = float(os.path.basename(path).replace(".pth", "").split("-")[-1])
            if val_loss < best_loss:
                best_loss, best_path = val_loss, path
        except (ValueError, IndexError):
            continue
    return best_path

# ═══════════════════════════════════════════════════════════════════════
# MAIN EVALUATION
# ═══════════════════════════════════════════════════════════════════════

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    eval_output_dir = os.path.join(args.model_dir, "evaluation_centroid")
    os.makedirs(eval_output_dir, exist_ok=True)

    # 1. Initialize Architecture
    checkpoint_path = args.checkpoint or find_best_checkpoint(args.model_dir)
    if not checkpoint_path:
        sys.exit(f"ERROR: No checkpoint in {args.model_dir}")
        
    if args.baseline:
        from SNN_final_model import SNN
        model = SNN()
    else:
        model = SNNViT(version=args.version)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model = model.to(device).eval()

    # 2. Collect file paths instead of loading data
    data_files = []
    for dname in DATA_DIRS:
        path = os.path.join(args.data_dir, dname)
        if os.path.isdir(path):
            data_files.extend([os.path.join(path, f) for f in os.listdir(path) if f.endswith('.pt')])
    
    if not data_files:
        sys.exit("ERROR: No .pt files found in data directory.")

    print(f"Found {len(data_files)} clips. Starting stream evaluation...")

    # 3. Setup Metrics
    thresholds = [0.001, 0.005, 0.01, 0.02, 0.03, 0.05]
    metrics = {t: {c: {"tp": 0, "fp": 0, "fn": 0} for c in CLASS_NAMES} for t in thresholds}
    total_frames = 0
    total_loss = 0.0
    loss_function = nn.MSELoss()

    # 4. Stream-Process Loop
    for data_idx, pt_path in enumerate(data_files):
        print(f"[{data_idx+1}/{len(data_files)}] Processing: {os.path.basename(pt_path)}")
        
        # Load one file at a time
        tensors = torch.load(pt_path, map_location="cpu", weights_only=False)
        # Only convert to dense for current processing
        dense_tensors = [t.to_dense() if t.is_sparse else t for t in tensors]
        
        events = dense_tensors[0] # [5400, 256, 256]
        targets = dense_tensors[1:5] # List of 4 [5400, 64, 64]
        n_frames = events.shape[0]

        # Crop and Resize (256->200, 64->50->64)
        ev_off, tg_off = 28, 7
        events_cropped = events[:, ev_off:ev_off+200, ev_off:ev_off+200]
        
        targets_cropped = []
        for tgt in targets:
            cropped = tgt[:, tg_off:tg_off+50, tg_off:tg_off+50].unsqueeze(1).float()
            resized = F.interpolate(cropped, size=(64, 64), mode="bilinear", align_corners=False)
            targets_cropped.append(resized.squeeze(1))

        # Carry membrane states through the sequence
        # Baseline SNN has 11 states, ViT has 3
        mem_states = (None,) * (11 if args.baseline else 3)
        
        sequence_length, overlap = 60, 25
        step_size = sequence_length - overlap
        seq_start = 0

        while seq_start + sequence_length <= n_frames:
            for t in range(seq_start, seq_start + sequence_length):
                frame = events_cropped[t].unsqueeze(0).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    with torch.amp.autocast(device_type="cuda" if "cuda" in str(device) else "cpu"):
                        # Unpack results based on model type
                        res = model(frame, mem_states)
                        p_person, p_car, p_bus, p_truck, mem_states = res

                preds = [p_person.view(64, 64).cpu().float().numpy(),
                         p_car.view(64, 64).cpu().float().numpy(),
                         p_bus.view(64, 64).cpu().float().numpy(),
                         p_truck.view(64, 64).cpu().float().numpy()]
                tgts_frame = [targets_cropped[i][t].numpy() for i in range(4)]

                # Accumulate Global Metrics
                for thresh in thresholds:
                    for ci, cname in enumerate(CLASS_NAMES):
                        smoothed = gaussian_filter(preds[ci], sigma=1.0)
                        pred_cents = find_centroids(smoothed, threshold=thresh)
                        target_cents = find_centroids(tgts_frame[ci], threshold=thresh * 0.5)
                        
                        tp, fp, fn = match_detections(pred_cents, target_cents)
                        metrics[thresh][cname]["tp"] += tp
                        metrics[thresh][cname]["fp"] += fp
                        metrics[thresh][cname]["fn"] += fn

                total_frames += 1
            seq_start += step_size

        # CRITICAL: Clear memory after each file
        del tensors, dense_tensors, events, targets, events_cropped, targets_cropped
        gc.collect()
        torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════════════════════
    # RESULTS REPORTING
    # ═══════════════════════════════════════════════════════════════════
    
    print("\n" + "="*60 + "\nCENTROID EVALUATION RESULTS\n" + "="*60)
    
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

    # Save to JSON
    results = {
        "checkpoint": os.path.basename(checkpoint_path),
        "total_frames": total_frames,
        "avg_loss": total_loss / max(total_frames, 1),
        "detection_metrics": results_table,
    }

    results_path = os.path.join(eval_output_dir, "eval_results_centroid.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DA150X SNN-ViT Centroid Evaluation")
    parser.add_argument("data_dir", help="Path to training data")
    parser.add_argument("model_dir", help="Path to model output dir")
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device ID")
    parser.add_argument("--checkpoint", default=None, help="Specific .pth file to evaluate")
    parser.add_argument("--version", type=str, default="v2", choices=["v1", "v2", "v3"], help="Architecture version")
    parser.add_argument("--baseline", action="store_true", help="Set this flag if evaluating the SNN Baseline (FC heads)")
    args = parser.parse_args()
    
    torch.cuda.set_device(args.gpu)
    evaluate(args)
