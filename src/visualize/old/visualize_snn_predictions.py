"""
SNN Prediction Video Generator
DA150X - KTH Royal Institute of Technology
Authors: Axel Prander & Viggo Jahr

Generates visualization videos from trained SNN baseline model predictions.
Same 3 video types as visualize_vit_predictions.py but for the SNN (FC heads)
architecture. Compare these side-by-side with ViT predictions for the thesis.

Usage:
  # From src/:
  CUDA_VISIBLE_DEVICES=3 python3 visualize_snn_predictions.py \
      ../data/raw_scaled/ \
      ../data/processed/experiments/snn_baseline_weights/ \
      --gpu 0 --week week_32-box_3 --fps 30

  # Or point at Axel's best SNN checkpoint:
  CUDA_VISIBLE_DEVICES=3 python3 visualize_snn_predictions.py \
      ../data/raw_scaled/ \
      ~/DA150X-Hybrid-SNN-ViT-Event-Detection/data/model_output/scaled/3-11-15-30/ \
      --gpu 0 --week week_32-box_3 --fps 30

  # All weeks:
  CUDA_VISIBLE_DEVICES=3 python3 visualize_snn_predictions.py \
      ../data/raw_scaled/ \
      ../data/processed/experiments/snn_baseline_weights/ \
      --gpu 0 --fps 30
"""

import argparse
import os
import re
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import cv2
from pathlib import Path

from norse.torch import LILinearCell
from norse.torch.module.lif import LIFCell, LIFParameters

# =============================================================================
# HYPERPARAMETERS (must match training)
# =============================================================================
tau_mem = 180
sequence_length = 60
overlap = 25
layer_nr = 500

# Class config
CLASS_NAMES = ['person', 'car', 'bus', 'truck']
CLASS_COLORS_BGR = [
    (0, 0, 255),     # person = red
    (255, 0, 0),     # car = blue
    (0, 255, 0),     # bus = green
    (0, 255, 255),   # truck = yellow
]
CLASS_COLORS_RGB = [
    (255, 0, 0),
    (0, 0, 255),
    (0, 255, 0),
    (255, 255, 0),
]


# =============================================================================
# SNN MODEL DEFINITION (must match SNN_final_model.py exactly)
# =============================================================================

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


# =============================================================================
# VIDEO GENERATION HELPERS
# =============================================================================

def find_best_checkpoint(model_dir):
    pth_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
    best_file, best_loss = None, float('inf')
    for f in pth_files:
        match = re.search(r'-(\d+)-(\d+\.?\d*).pth', f)
        if match:
            val_loss = float(match.group(2))
            if val_loss < best_loss:
                best_loss = val_loss
                best_file = f
    return os.path.join(model_dir, best_file) if best_file else None


def heatmap_to_color(heatmap_np, color_rgb, vmax=None):
    if vmax is None:
        vmax = max(heatmap_np.max(), 0.001)
    normalized = np.clip(heatmap_np / vmax, 0, 1)
    overlay = np.zeros((heatmap_np.shape[0], heatmap_np.shape[1], 3), dtype=np.float32)
    for c in range(3):
        overlay[:, :, c] = normalized * color_rgb[c] / 255.0
    return overlay, normalized


def extract_peaks(heatmap_np, threshold=0.03, min_distance=5):
    try:
        from scipy.ndimage import maximum_filter, label
    except ImportError:
        return []
    local_max = maximum_filter(heatmap_np, size=min_distance * 2 + 1)
    peaks = (heatmap_np == local_max) & (heatmap_np > threshold)
    labeled, num_features = label(peaks)
    centers = []
    for i in range(1, num_features + 1):
        ys, xs = np.where(labeled == i)
        cy, cx = ys.mean(), xs.mean()
        val = heatmap_np[int(cy), int(cx)]
        centers.append((cx, cy, val))
    return centers


def draw_legend(frame, x_start, y_start):
    for i, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS_BGR)):
        y = y_start + i * 20
        cv2.rectangle(frame, (x_start, y), (x_start + 12, y + 12), color, -1)
        cv2.putText(frame, name, (x_start + 18, y + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


# =============================================================================
# VIDEO 1: PREDICTION OVERLAY
# =============================================================================

def generate_prediction_overlay(events, predictions, output_path, fps=30):
    num_frames = events.shape[0]
    h, w = events.shape[1], events.shape[2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for frame_idx in range(num_frames):
        event = events[frame_idx].numpy()
        event_norm = np.clip(event / max(event.max(), 1), 0, 1)
        base = (event_norm * 255).astype(np.uint8)
        base_bgr = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)

        combined_overlay = np.zeros((h, w, 3), dtype=np.float32)

        for c in range(4):
            pred = predictions[c][frame_idx].numpy()
            pred_resized = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)
            overlay, mask = heatmap_to_color(pred_resized, CLASS_COLORS_RGB[c],
                                              vmax=max(pred_resized.max(), 0.01))
            combined_overlay += overlay

        combined_overlay = np.clip(combined_overlay, 0, 1)
        blend = (base_bgr.astype(np.float32) / 255 * 0.5 + combined_overlay * 0.5)
        blend = (np.clip(blend, 0, 1) * 255).astype(np.uint8)
        blend = cv2.cvtColor(blend, cv2.COLOR_RGB2BGR)

        draw_legend(blend, 10, 10)
        cv2.putText(blend, f'SNN Baseline | Frame {frame_idx}/{num_frames}',
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        out.write(blend)

    out.release()
    print(f"    Saved: {output_path}")


# =============================================================================
# VIDEO 2: SIDE-BY-SIDE (ground truth vs SNN prediction)
# =============================================================================

def generate_side_by_side(events, targets, predictions, output_path, fps=30):
    num_frames = events.shape[0]
    panel_w, panel_h = 128, 128
    total_w = panel_w * 4
    row_h = panel_h
    header_h = 30
    frame_w = total_w
    frame_h = header_h + row_h * 2 + 160

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))

    for frame_idx in range(num_frames):
        canvas = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)

        cv2.putText(canvas, 'Ground Truth', (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(canvas, 'SNN Prediction (FC heads)', (10, 20 + row_h),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

        for c in range(4):
            x_off = c * panel_w

            # Target
            tgt = targets[c][frame_idx].numpy()
            tgt_norm = np.clip(tgt / max(tgt.max(), 0.03), 0, 1)
            tgt_color = np.zeros((64, 64, 3), dtype=np.uint8)
            for ch in range(3):
                tgt_color[:, :, ch] = (tgt_norm * CLASS_COLORS_RGB[c][ch]).astype(np.uint8)
            tgt_resized = cv2.resize(tgt_color, (panel_w, panel_h))
            canvas[header_h:header_h + row_h, x_off:x_off + panel_w] = cv2.cvtColor(
                tgt_resized, cv2.COLOR_RGB2BGR)

            # Prediction
            pred = predictions[c][frame_idx].numpy()
            pred_vmax = max(pred.max(), 0.01)
            pred_norm = np.clip(pred / pred_vmax, 0, 1)
            pred_color = np.zeros((64, 64, 3), dtype=np.uint8)
            for ch in range(3):
                pred_color[:, :, ch] = (pred_norm * CLASS_COLORS_RGB[c][ch]).astype(np.uint8)
            pred_resized = cv2.resize(pred_color, (panel_w, panel_h))
            canvas[header_h + row_h:header_h + row_h * 2, x_off:x_off + panel_w] = cv2.cvtColor(
                pred_resized, cv2.COLOR_RGB2BGR)

            cv2.putText(canvas, CLASS_NAMES[c], (x_off + 5, header_h - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, CLASS_COLORS_BGR[c], 1)

        # Event frame at bottom
        event = events[frame_idx].numpy()
        event_norm = np.clip(event / max(event.max(), 1), 0, 1)
        event_uint8 = (event_norm * 255).astype(np.uint8)
        event_resized = cv2.resize(event_uint8, (frame_w, 160))
        event_bgr = cv2.cvtColor(event_resized, cv2.COLOR_GRAY2BGR)
        canvas[header_h + row_h * 2:, :] = event_bgr

        cv2.putText(canvas, f'{frame_idx}/{num_frames}',
                    (frame_w - 80, frame_h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

        out.write(canvas)

    out.release()
    print(f"    Saved: {output_path}")


# =============================================================================
# VIDEO 3: PEAK DETECTION WITH BOUNDING BOXES
# =============================================================================

def generate_bbox_video(events, predictions, output_path, fps=30):
    try:
        from scipy.ndimage import maximum_filter, label
    except ImportError:
        print("    Skipping bbox video (scipy not installed)")
        return

    num_frames = events.shape[0]
    h, w = events.shape[1], events.shape[2]

    # SNN predictions are much more diffuse than ViT, so thresholds
    # need to be relative to the prediction range.
    # SNN max (mean): person=2.15, car=4.10, bus=1.92, truck=0.91
    CLASS_THRESHOLDS = [1.5, 3.0, 1.5, 0.7]  # person, car, bus, truck

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    scale_x = w / 64.0
    scale_y = h / 64.0

    for frame_idx in range(num_frames):
        event = events[frame_idx].numpy()
        event_norm = np.clip(event / max(event.max(), 1), 0, 1)
        base = (event_norm * 255).astype(np.uint8)
        frame_bgr = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)

        for c in range(4):
            pred = predictions[c][frame_idx].numpy()
            peaks = extract_peaks(pred, threshold=CLASS_THRESHOLDS[c], min_distance=10)

            for (cx, cy, val) in peaks:
                cx_scaled = int(cx * scale_x)
                cy_scaled = int(cy * scale_y)
                box_half = int(max(8, val * 300) * scale_x / 64)

                x1 = max(0, cx_scaled - box_half)
                y1 = max(0, cy_scaled - box_half)
                x2 = min(w - 1, cx_scaled + box_half)
                y2 = min(h - 1, cy_scaled + box_half)

                color = CLASS_COLORS_BGR[c]
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame_bgr, f'{CLASS_NAMES[c]} {val:.2f}',
                            (x1, max(y1 - 5, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

        draw_legend(frame_bgr, 10, 10)
        cv2.putText(frame_bgr, f'SNN Baseline | Frame {frame_idx}/{num_frames}',
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        out.write(frame_bgr)

    out.release()
    print(f"    Saved: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate SNN baseline prediction videos")
    parser.add_argument("data_dir", help="Path to training data (e.g. ../data/raw_scaled/)")
    parser.add_argument("model_dir", help="Path to model dir with .pth checkpoints")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--week", type=str, default=None,
                        help="Specific week (e.g. week_32-box_3). Default: all.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: <model_dir>/videos/)")

    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(args.gpu)

    # Load model
    checkpoint_path = find_best_checkpoint(args.model_dir)
    if not checkpoint_path:
        print("ERROR: No checkpoint found")
        return

    print(f"Loading SNN model: {checkpoint_path}")
    model = SNN().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    output_dir = args.output_dir or os.path.join(args.model_dir, 'videos')
    os.makedirs(output_dir, exist_ok=True)

    data_dirs = ["week_32-box_3", "week_33-box_2", "week_34-box_1",
                 "week_35-box_2", "week_36-box_3"]

    if args.week:
        data_dirs = [args.week]

    for dir_name in data_dirs:
        dir_path = os.path.join(args.data_dir, dir_name)
        if not os.path.exists(dir_path):
            print(f"Skipping {dir_name} (not found)")
            continue

        pt_files = [f for f in os.listdir(dir_path) if f.endswith('.pt')]
        for pt_file in pt_files:
            fpath = os.path.join(dir_path, pt_file)
            print(f"\nProcessing {dir_name}/{pt_file}...")

            # Load data
            data = torch.load(fpath)
            for i in range(5):
                if not data[i].is_coalesced():
                    data[i] = data[i].coalesce()

            events = data[0].to_dense().float()
            targets = [data[i + 1].to_dense().float() for i in range(4)]

            # Center crop to 200x200 (matching training)
            events_cropped = TF.center_crop(events, (200, 200))
            targets_cropped = [
                TF.resize(TF.center_crop(t.unsqueeze(1), (50, 50)), (64, 64)).squeeze(1)
                for t in targets
            ]

            num_frames = events_cropped.shape[0]

            # Run inference
            print(f"  Running SNN inference on {num_frames} frames...")
            all_preds = [[], [], [], []]

            start = time.time()
            # 11 membrane states for SNN
            mem_states = tuple(None for _ in range(11))

            with torch.no_grad():
                for frame_idx in range(num_frames):
                    input_frame = events_cropped[frame_idx].unsqueeze(0).unsqueeze(0).to(device)
                    o1, o2, o3, o4, mem_states = model(input_frame, mem_states)

                    # SNN outputs are [1, 4096], reshape to [64, 64]
                    all_preds[0].append(o1.view(64, 64).cpu())
                    all_preds[1].append(o2.view(64, 64).cpu())
                    all_preds[2].append(o3.view(64, 64).cpu())
                    all_preds[3].append(o4.view(64, 64).cpu())

                    if (frame_idx + 1) % 1000 == 0:
                        elapsed = time.time() - start
                        fps_actual = (frame_idx + 1) / elapsed
                        print(f"    {frame_idx + 1}/{num_frames} ({fps_actual:.0f} fps)")

            predictions = [torch.stack(p) for p in all_preds]

            elapsed = time.time() - start
            print(f"  Inference done: {elapsed:.1f}s ({num_frames / elapsed:.0f} fps)")

            # Generate videos
            prefix = dir_name.replace('/', '-')

            print(f"  Generating SNN prediction overlay video...")
            generate_prediction_overlay(
                events_cropped, predictions,
                os.path.join(output_dir, f'{prefix}_snn_predictions.mp4'),
                fps=args.fps,
            )

            print(f"  Generating SNN side-by-side comparison video...")
            generate_side_by_side(
                events_cropped, targets_cropped, predictions,
                os.path.join(output_dir, f'{prefix}_snn_comparison.mp4'),
                fps=args.fps,
            )

            print(f"  Generating SNN bounding box detection video...")
            generate_bbox_video(
                events_cropped, predictions,
                os.path.join(output_dir, f'{prefix}_snn_bboxes.mp4'),
                fps=args.fps,
            )

    print(f"\nAll videos saved to: {output_dir}/")


if __name__ == '__main__':
    main()
