"""
ViT Prediction Video Generator
DA150X - KTH Royal Institute of Technology
Authors: Axel Prander & Viggo Jahr

Generates visualization videos from trained SNN+ViT model predictions:
  1. Prediction overlay: color-coded heatmaps on event frames
  2. Side-by-side: ground truth vs prediction per class
  3. Peak detection: bounding boxes from heatmap peaks on event frames

Usage:
  # All videos for one clip:
  CUDA_VISIBLE_DEVICES=3 python3 visualize_vit_predictions.py \
      ../data/raw_scaled/ \
      data/processed/vit/vit-3-13-18-55/ \
      --gpu 0

  # Specific week only:
  CUDA_VISIBLE_DEVICES=3 python3 visualize_vit_predictions.py \
      ../data/raw_scaled/ \
      data/processed/vit/vit-3-13-18-55/ \
      --gpu 0 --week week_32-box_3

  # Custom fps and output dir:
  CUDA_VISIBLE_DEVICES=3 python3 visualize_vit_predictions.py \
      ../data/raw_scaled/ \
      data/processed/vit/vit-3-13-18-55/ \
      --gpu 0 --fps 30 --output_dir ../data/visualizations/vit/
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
import timm

# =============================================================================
# HYPERPARAMETERS (must match training)
# =============================================================================
tau_mem = 180
sequence_length = 60
overlap = 25

# Class config
CLASS_NAMES = ['person', 'car', 'bus', 'truck']
# BGR colors for OpenCV
CLASS_COLORS_BGR = [
    (0, 0, 255),     # person = red
    (255, 0, 0),     # car = blue
    (0, 255, 0),     # bus = green
    (0, 255, 255),   # truck = yellow
]
CLASS_COLORS_RGB = [
    (255, 0, 0),     # person = red
    (0, 0, 255),     # car = blue
    (0, 255, 0),     # bus = green
    (255, 255, 0),   # truck = yellow
]


# =============================================================================
# MODEL DEFINITION (must match SNN_ViT_model.py)
# =============================================================================

class ViTHeatmapHead(nn.Module):
    def __init__(self, in_channels=8, grid_size=20, num_classes=4, output_size=64):
        super().__init__()
        self.grid_size = grid_size
        vit = timm.create_model('vit_tiny_patch16_224', pretrained=False)
        embed_dim = vit.embed_dim
        self.blocks = vit.blocks[:4]
        self.norm = vit.norm
        self.patch_embed = nn.Linear(in_channels, embed_dim)
        self.pos_embed = nn.Parameter(
            torch.randn(1, grid_size * grid_size, embed_dim) * 0.02
        )
        self.to_spatial = nn.Sequential(nn.Linear(embed_dim, 64), nn.GELU())
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.GELU(),
            nn.Conv2d(16, num_classes, 3, padding=1),
            nn.AdaptiveAvgPool2d(output_size),
        )
        del vit

    def forward(self, spk3):
        B, C, H, W = spk3.shape
        x = spk3.flatten(2).transpose(1, 2)
        x = self.patch_embed(x) + self.pos_embed
        x = self.blocks(x)
        x = self.norm(x)
        x = self.to_spatial(x)
        x = x.transpose(1, 2).reshape(B, 64, self.grid_size, self.grid_size)
        return self.decoder(x)


class SNNViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=7, stride=2, padding=0)
        self.bn1 = nn.BatchNorm2d(8)
        self.lif1 = LIFCell(p=LIFParameters(tau_mem_inv=tau_mem))
        self.conv2 = nn.Conv2d(8, 8, kernel_size=5, stride=2, padding=0)
        self.bn2 = nn.BatchNorm2d(8)
        self.lif2 = LIFCell(p=LIFParameters(tau_mem_inv=tau_mem))
        self.conv3 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(8)
        self.lif3 = LIFCell(p=LIFParameters(tau_mem_inv=tau_mem))
        self.maxpool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(p=0.3)
        self.vit_head = ViTHeatmapHead(in_channels=8, grid_size=20,
                                        num_classes=4, output_size=64)

    def forward(self, x, mem_states):
        batch_size, C, W, H = x.shape
        x = (x != 0).float()
        mem1, mem2, mem3 = mem_states
        v1 = self.bn1(self.conv1(x))
        spk1, mem1 = self.lif1(v1, mem1)
        v2 = self.dropout1(self.bn2(self.conv2(self.maxpool(spk1))))
        spk2, mem2 = self.lif2(v2, mem2)
        v3 = self.dropout1(self.bn3(self.conv3(spk2)))
        spk3, mem3 = self.lif3(v3, mem3)
        heatmaps = self.vit_head(spk3)
        return heatmaps[:, 0], heatmaps[:, 1], heatmaps[:, 2], heatmaps[:, 3], (mem1, mem2, mem3)


# =============================================================================
# VIDEO GENERATION HELPERS
# =============================================================================

def find_best_checkpoint(model_dir):
    """Find checkpoint with lowest val loss."""
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


def heatmap_to_color(heatmap_np, color_rgb, alpha=0.6, vmax=None):
    """Convert a single-channel heatmap to a colored RGBA overlay."""
    if vmax is None:
        vmax = max(heatmap_np.max(), 0.001)
    normalized = np.clip(heatmap_np / vmax, 0, 1)

    overlay = np.zeros((heatmap_np.shape[0], heatmap_np.shape[1], 3), dtype=np.float32)
    for c in range(3):
        overlay[:, :, c] = normalized * color_rgb[c] / 255.0

    return overlay, normalized


def extract_peaks(heatmap_np, threshold=0.03, min_distance=3):
    """Extract local maxima from heatmap as detection centers."""
    from scipy.ndimage import maximum_filter, label

    # Find local maxima
    local_max = maximum_filter(heatmap_np, size=min_distance * 2 + 1)
    peaks = (heatmap_np == local_max) & (heatmap_np > threshold)

    # Label connected components
    labeled, num_features = label(peaks)

    centers = []
    for i in range(1, num_features + 1):
        ys, xs = np.where(labeled == i)
        cy, cx = ys.mean(), xs.mean()
        val = heatmap_np[int(cy), int(cx)]
        centers.append((cx, cy, val))

    return centers


def draw_legend(frame, x_start, y_start):
    """Draw class color legend on frame."""
    for i, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS_BGR)):
        y = y_start + i * 20
        cv2.rectangle(frame, (x_start, y), (x_start + 12, y + 12), color, -1)
        cv2.putText(frame, name, (x_start + 18, y + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


# =============================================================================
# VIDEO 1: PREDICTION OVERLAY
# =============================================================================

def generate_prediction_overlay(events, predictions, output_path, fps=30):
    """
    Overlay predicted heatmaps (color-coded per class) on event frames.
    Similar to visualize_data.py but using model predictions.
    """
    num_frames = events.shape[0]
    h, w = events.shape[1], events.shape[2]

    # Scale predictions to event frame resolution
    pred_h, pred_w = predictions[0].shape[1], predictions[0].shape[2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for frame_idx in range(num_frames):
        # Base event frame (grayscale → BGR)
        event = events[frame_idx].numpy()
        event_norm = np.clip(event / max(event.max(), 1), 0, 1)
        base = (event_norm * 255).astype(np.uint8)
        base_bgr = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)

        # Overlay each class heatmap
        combined_overlay = np.zeros((h, w, 3), dtype=np.float32)

        for c in range(4):
            pred = predictions[c][frame_idx].numpy()
            # Resize prediction to event frame size
            pred_resized = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)

            overlay, mask = heatmap_to_color(pred_resized, CLASS_COLORS_RGB[c],
                                              vmax=max(pred_resized.max(), 0.01))
            combined_overlay += overlay

        # Blend
        combined_overlay = np.clip(combined_overlay, 0, 1)
        blend = (base_bgr.astype(np.float32) / 255 * 0.5 +
                 combined_overlay * 0.5)
        blend = (np.clip(blend, 0, 1) * 255).astype(np.uint8)

        # Convert RGB overlay to BGR for OpenCV
        blend = cv2.cvtColor(blend, cv2.COLOR_RGB2BGR)

        # Legend
        draw_legend(blend, 10, 10)

        # Frame counter
        cv2.putText(blend, f'Frame {frame_idx}/{num_frames}',
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 1)

        out.write(blend)

    out.release()
    print(f"    Saved: {output_path}")


# =============================================================================
# VIDEO 2: SIDE-BY-SIDE (ground truth vs prediction)
# =============================================================================

def generate_side_by_side(events, targets, predictions, output_path, fps=30):
    """
    Side-by-side comparison: left = ground truth heatmaps, right = predictions.
    Event frame on top, 4 class pairs below.
    """
    num_frames = events.shape[0]

    # Layout: 640 wide, event on top, then 4 class rows
    panel_w, panel_h = 128, 128  # per heatmap panel
    total_w = panel_w * 4  # 4 classes side by side
    row_h = panel_h
    header_h = 30

    # Two rows: targets and predictions, 4 classes each
    frame_w = total_w
    frame_h = header_h + row_h * 2 + 160  # event frame at bottom

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))

    for frame_idx in range(num_frames):
        canvas = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)

        # Header
        cv2.putText(canvas, 'Ground Truth', (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(canvas, 'ViT Prediction', (10, 20 + row_h),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

        for c in range(4):
            x_off = c * panel_w

            # Target heatmap
            tgt = targets[c][frame_idx].numpy()
            tgt_norm = np.clip(tgt / max(tgt.max(), 0.03), 0, 1)
            tgt_color = np.zeros((64, 64, 3), dtype=np.uint8)
            for ch in range(3):
                tgt_color[:, :, ch] = (tgt_norm * CLASS_COLORS_RGB[c][ch]).astype(np.uint8)
            tgt_resized = cv2.resize(tgt_color, (panel_w, panel_h))
            canvas[header_h:header_h + row_h, x_off:x_off + panel_w] = cv2.cvtColor(
                tgt_resized, cv2.COLOR_RGB2BGR)

            # Prediction heatmap
            pred = predictions[c][frame_idx].numpy()
            pred_vmax = max(pred.max(), 0.01)
            pred_norm = np.clip(pred / pred_vmax, 0, 1)
            pred_color = np.zeros((64, 64, 3), dtype=np.uint8)
            for ch in range(3):
                pred_color[:, :, ch] = (pred_norm * CLASS_COLORS_RGB[c][ch]).astype(np.uint8)
            pred_resized = cv2.resize(pred_color, (panel_w, panel_h))
            canvas[header_h + row_h:header_h + row_h * 2, x_off:x_off + panel_w] = cv2.cvtColor(
                pred_resized, cv2.COLOR_RGB2BGR)

            # Class label
            cv2.putText(canvas, CLASS_NAMES[c],
                        (x_off + 5, header_h - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        CLASS_COLORS_BGR[c], 1)

        # Event frame at bottom
        event = events[frame_idx].numpy()
        event_norm = np.clip(event / max(event.max(), 1), 0, 1)
        event_uint8 = (event_norm * 255).astype(np.uint8)
        event_resized = cv2.resize(event_uint8, (frame_w, 160))
        event_bgr = cv2.cvtColor(event_resized, cv2.COLOR_GRAY2BGR)
        canvas[header_h + row_h * 2:, :] = event_bgr

        # Frame counter
        cv2.putText(canvas, f'{frame_idx}/{num_frames}',
                    (frame_w - 80, frame_h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

        out.write(canvas)

    out.release()
    print(f"    Saved: {output_path}")


# =============================================================================
# VIDEO 3: PEAK DETECTION WITH BOUNDING BOXES
# =============================================================================

def generate_bbox_video(events, predictions, output_path, fps=30, threshold=0.05):
    """
    Extract peaks from predicted heatmaps, draw bounding boxes on event frames.
    This shows the model's detections as a traditional detector would display them.

    Uses per-class thresholds based on each class's prediction range:
      person: peaks at ~0.4  → threshold 0.3
      car:    peaks at ~3.5  → threshold 2.5
      bus:    peaks at ~1.9  → threshold 1.2
      truck:  peaks at ~0.4  → threshold 0.3

    These can be overridden by passing a single threshold (applied uniformly).
    """
    try:
        from scipy.ndimage import maximum_filter, label
    except ImportError:
        print("    Skipping bbox video (scipy not installed)")
        return

    num_frames = events.shape[0]
    h, w = events.shape[1], events.shape[2]

    # Per-class thresholds tuned to prediction magnitude ranges
    # Adjust these if detections are too noisy or too sparse
    CLASS_THRESHOLDS = [0.6, 4.0, 2.0, 0.5]  # person, car, bus, truck

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # Scale factors from 64x64 heatmap to event frame resolution
    scale_x = w / 64.0
    scale_y = h / 64.0

    for frame_idx in range(num_frames):
        # Base event frame
        event = events[frame_idx].numpy()
        event_norm = np.clip(event / max(event.max(), 1), 0, 1)
        base = (event_norm * 255).astype(np.uint8)
        frame_bgr = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)

        for c in range(4):
            pred = predictions[c][frame_idx].numpy()

            # Find peaks with per-class threshold and wider min_distance
            peaks = extract_peaks(pred, threshold=CLASS_THRESHOLDS[c], min_distance=10)

            for (cx, cy, val) in peaks:
                # Scale to event frame coordinates
                cx_scaled = int(cx * scale_x)
                cy_scaled = int(cy * scale_y)

                # Estimate box size from heatmap blob
                # Use a fixed size proportional to confidence
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

        # Legend
        draw_legend(frame_bgr, 10, 10)
        cv2.putText(frame_bgr, f'Frame {frame_idx}/{num_frames}',
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 1)

        out.write(frame_bgr)

    out.release()
    print(f"    Saved: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate ViT prediction videos")
    parser.add_argument("data_dir", help="Path to training data (e.g. ../data/raw_scaled/)")
    parser.add_argument("model_dir", help="Path to model dir with .pth checkpoints")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--week", type=str, default=None,
                        help="Specific week to process (e.g. week_32-box_3). Default: all.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: <model_dir>/videos/)")
    parser.add_argument("--bbox_threshold", type=float, default=0.05,
                        help="Peak detection threshold for bbox video (default: 0.05)")

    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(args.gpu)

    # Load model
    checkpoint_path = find_best_checkpoint(args.model_dir)
    if not checkpoint_path:
        print("ERROR: No checkpoint found")
        return

    print(f"Loading model: {checkpoint_path}")
    model = SNNViT().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Output directory
    output_dir = args.output_dir or os.path.join(args.model_dir, 'videos')
    os.makedirs(output_dir, exist_ok=True)

    # Data directories
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

            events = data[0].to_dense().float()  # [5400, 256, 256]
            targets = [data[i + 1].to_dense().float() for i in range(4)]

            # Center crop events to 200x200 (matching training)
            events_cropped = TF.center_crop(events, (200, 200))
            targets_cropped = [
                TF.resize(TF.center_crop(t.unsqueeze(1), (50, 50)), (64, 64)).squeeze(1)
                for t in targets
            ]

            num_frames = events_cropped.shape[0]

            # Run inference on all frames
            print(f"  Running inference on {num_frames} frames...")
            all_preds = [[], [], [], []]  # per class

            start = time.time()
            mem_states = (None, None, None)

            with torch.no_grad():
                for frame_idx in range(num_frames):
                    input_frame = events_cropped[frame_idx].unsqueeze(0).unsqueeze(0).to(device)
                    o1, o2, o3, o4, mem_states = model(input_frame, mem_states)

                    all_preds[0].append(o1.view(64, 64).cpu())
                    all_preds[1].append(o2.view(64, 64).cpu())
                    all_preds[2].append(o3.view(64, 64).cpu())
                    all_preds[3].append(o4.view(64, 64).cpu())

                    if (frame_idx + 1) % 1000 == 0:
                        elapsed = time.time() - start
                        fps_actual = (frame_idx + 1) / elapsed
                        print(f"    {frame_idx + 1}/{num_frames} ({fps_actual:.0f} fps)")

            # Stack predictions
            predictions = [torch.stack(p) for p in all_preds]  # 4 x [5400, 64, 64]

            elapsed = time.time() - start
            print(f"  Inference done: {elapsed:.1f}s ({num_frames / elapsed:.0f} fps)")

            # Generate videos
            prefix = dir_name.replace('/', '-')

            print(f"  Generating prediction overlay video...")
            generate_prediction_overlay(
                events_cropped, predictions,
                os.path.join(output_dir, f'{prefix}_vit_predictions.mp4'),
                fps=args.fps,
            )

            print(f"  Generating side-by-side comparison video...")
            generate_side_by_side(
                events_cropped, targets_cropped, predictions,
                os.path.join(output_dir, f'{prefix}_vit_comparison.mp4'),
                fps=args.fps,
            )

            print(f"  Generating bounding box detection video...")
            generate_bbox_video(
                events_cropped, predictions,
                os.path.join(output_dir, f'{prefix}_vit_bboxes.mp4'),
                fps=args.fps,
                threshold=args.bbox_threshold,
            )

    print(f"\nAll videos saved to: {output_dir}/")


if __name__ == '__main__':
    main()
