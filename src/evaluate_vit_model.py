"""
Evaluation Script for SNN+ViT Hybrid Model
DA150X - KTH Royal Institute of Technology
Authors: Axel Prander & Viggo Jahr

Evaluates trained SNN+ViT checkpoints using the same metrics as the SNN
baseline evaluation. Produces MSE loss, precision/recall/F1 per class,
and visual comparison images (target vs predicted heatmaps).

Based on evaluate_model.py (SNN baseline) with these changes:
  - Uses SNNViT class instead of SNN (ViT head, not FC heads)
  - 3 membrane states instead of 11
  - Imports ViTHeatmapHead from timm-based architecture
  - Same center-crop logic (200x200 events, 50x50 targets)

Usage:
  CUDA_VISIBLE_DEVICES=3 python3 evaluate_vit_model.py \
      data/raw_scaled/ \
      src/data/processed/vit/vit-3-13-18-48/ \
      --gpu 0 --save_images

  Or specify a checkpoint explicitly:
  CUDA_VISIBLE_DEVICES=3 python3 evaluate_vit_model.py \
      data/raw_scaled/ \
      src/data/processed/vit/vit-3-13-18-48/ \
      --gpu 0 --save_images \
      --checkpoint multiclass-adamw-42-85.1234.pth
"""

import argparse
import json
import os
import re
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from pathlib import Path

# Norse imports for SNN backbone
from norse.torch import LILinearCell
from norse.torch.module.lif import LIFCell, LIFParameters

# timm for ViT
import timm

# =============================================================================
# HYPERPARAMETERS (must match training)
# =============================================================================
tau_mem = 180
batch_size = 1  # Evaluate one sequence at a time
sequence_length = 60
overlap = 25

# =============================================================================
# MODEL DEFINITION (must match SNN_ViT_model.py exactly)
# =============================================================================

class ViTHeatmapHead(nn.Module):
    def __init__(self, in_channels=8, grid_size=20,
                 num_classes=4, output_size=64):
        super().__init__()
        self.grid_size = grid_size
        self.num_classes = num_classes

        vit = timm.create_model('vit_tiny_patch16_224', pretrained=False)
        embed_dim = vit.embed_dim  # 192

        # Only first 4 layers (must match training)
        self.blocks = vit.blocks[:4]
        self.norm = vit.norm

        self.patch_embed = nn.Linear(in_channels, embed_dim)
        self.pos_embed = nn.Parameter(
            torch.randn(1, grid_size * grid_size, embed_dim) * 0.02
        )

        self.to_spatial = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.GELU(),
            nn.Conv2d(16, num_classes, kernel_size=3, padding=1),
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
        x = self.decoder(x)
        return x


class SNNViT(nn.Module):
    def __init__(self):
        super(SNNViT, self).__init__()
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

        self.vit_head = ViTHeatmapHead(
            in_channels=8, grid_size=20,
            num_classes=4, output_size=64,
        )

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

        person = heatmaps[:, 0]
        car = heatmaps[:, 1]
        bus = heatmaps[:, 2]
        truck = heatmaps[:, 3]

        return person, car, bus, truck, (mem1, mem2, mem3)


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def find_best_checkpoint(model_dir, checkpoint_name=None):
    """Find the checkpoint with the lowest validation loss."""
    if checkpoint_name:
        path = os.path.join(model_dir, checkpoint_name)
        if os.path.exists(path):
            return path
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    pth_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
    if not pth_files:
        raise FileNotFoundError(f"No .pth files found in {model_dir}")

    # Parse val loss from filename: multiclass-adamw-{epoch}-{val_loss}.pth
    best_file = None
    best_loss = float('inf')
    for f in pth_files:
        match = re.search(r'-(\d+)-(\d+\.?\d*).pth', f)
        if match:
            val_loss = float(match.group(2))
            if val_loss < best_loss:
                best_loss = val_loss
                best_file = f

    if best_file is None:
        best_file = sorted(pth_files)[-1]  # Fallback: last alphabetically

    print(f"Best checkpoint: {best_file} (val_loss={best_loss:.4f})")
    return os.path.join(model_dir, best_file)


def center_crop_events(frames, crop_size=200):
    """Center crop event frames from 256x256 to 200x200 (matching training augmentation)."""
    return TF.center_crop(frames, (crop_size, crop_size))


def center_crop_targets(targets, crop_from=64, crop_to=50, resize_to=64):
    """Center crop targets from 64x64 to 50x50, then resize back to 64x64."""
    cropped = TF.center_crop(targets, (crop_to, crop_to))
    resized = TF.resize(cropped, (resize_to, resize_to))
    return resized


def compute_detection_metrics(pred, target, threshold=0.03):
    """Compute TP, FP, FN for heatmap detection at a given threshold."""
    pred_binary = (pred > threshold).float()
    target_binary = (target > 0).float()

    tp = (pred_binary * target_binary).sum().item()
    fp = (pred_binary * (1 - target_binary)).sum().item()
    fn = ((1 - pred_binary) * target_binary).sum().item()

    return tp, fp, fn


def save_comparison_image(event_frame, targets, predictions, frame_idx, save_dir):
    """Save a visual comparison: target heatmaps vs predicted heatmaps + input."""
    class_names = ['Person', 'Car', 'Bus', 'Truck']

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))

    for i in range(4):
        # Row 1: Target heatmaps
        axes[0, i].imshow(targets[i].cpu().numpy(), cmap='hot', vmin=0, vmax=0.03)
        axes[0, i].set_title(f'Target: {class_names[i]}', fontsize=10)
        axes[0, i].axis('off')

        # Row 2: Predicted heatmaps
        pred_max = predictions[i].cpu().numpy().max()
        axes[1, i].imshow(predictions[i].cpu().numpy(), cmap='hot',
                          vmin=0, vmax=max(pred_max, 0.001))
        axes[1, i].set_title(f'Pred: {class_names[i]} (max={pred_max:.4f})', fontsize=10)
        axes[1, i].axis('off')

    # Row 3: Input event frame (span all 4 columns)
    for i in range(4):
        if i == 0:
            axes[2, i].imshow(event_frame.cpu().numpy(), cmap='gray')
            axes[2, i].set_title('Input Event Frame', fontsize=10)
        axes[2, i].axis('off')

    plt.suptitle(f'Frame {frame_idx}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'eval_frame_{frame_idx:05d}.png'),
                dpi=100, bbox_inches='tight')
    plt.close()


# =============================================================================
# MAIN EVALUATION
# =============================================================================

def evaluate(data_dir, model_dir, device, save_images=False, checkpoint_name=None):
    # Find and load model
    checkpoint_path = find_best_checkpoint(model_dir, checkpoint_name)
    model = SNNViT().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    print(f"Loaded model from: {checkpoint_path}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Set up output directory
    eval_dir = os.path.join(model_dir, 'evaluation')
    os.makedirs(eval_dir, exist_ok=True)
    if save_images:
        img_dir = os.path.join(eval_dir, 'images')
        os.makedirs(img_dir, exist_ok=True)

    # Scan data directories
    data_dirs = [
        "week_32-box_3", "week_33-box_2", "week_34-box_1",
        "week_35-box_2", "week_36-box_3",
    ]

    loss_fn = nn.MSELoss()

    # Accumulators
    total_loss = 0
    class_losses = [0, 0, 0, 0]  # person, car, bus, truck
    class_tp = [0, 0, 0, 0]
    class_fp = [0, 0, 0, 0]
    class_fn = [0, 0, 0, 0]
    total_frames = 0

    # Per-class prediction statistics
    class_max_vals = [[], [], [], []]
    class_mean_vals = [[], [], [], []]
    class_nonzero_pcts = [[], [], [], []]

    class_names = ['person', 'car', 'bus', 'truck']

    print("\nEvaluating...")
    start_time = time.time()

    with torch.no_grad():
        for dir_name in data_dirs:
            dir_path = os.path.join(data_dir, dir_name)
            if not os.path.exists(dir_path):
                print(f"  Skipping {dir_name} (not found)")
                continue

            pt_files = sorted([f for f in os.listdir(dir_path) if f.endswith('.pt')])
            for pt_file in pt_files:
                fpath = os.path.join(dir_path, pt_file)
                print(f"  Processing {dir_name}/{pt_file}...")

                # Load data
                data = torch.load(fpath)
                for i in range(5):
                    if not data[i].is_coalesced():
                        data[i] = data[i].coalesce()

                events = data[0].to_dense().float()  # [5400, 256, 256]
                targets = []
                for i in range(4):
                    targets.append(data[i + 1].to_dense().float())  # [5400, 64, 64]

                num_frames = events.shape[0]
                clip_frames = 0

                # Process in sequences
                start_idx = 0
                while start_idx + sequence_length <= num_frames:
                    # Extract sequence
                    event_seq = events[start_idx:start_idx + sequence_length]  # [60, 256, 256]
                    target_seq = [t[start_idx:start_idx + sequence_length] for t in targets]  # 4 x [60, 64, 64]

                    # Center crop (matching training augmentation)
                    event_seq = center_crop_events(event_seq, 200)  # [60, 200, 200]
                    target_seq = [center_crop_targets(t.unsqueeze(1), 64, 50, 64).squeeze(1)
                                  for t in target_seq]  # 4 x [60, 64, 64]

                    # Run through model
                    mem_states = (None, None, None)

                    for step in range(sequence_length):
                        input_frame = event_seq[step].unsqueeze(0).unsqueeze(0).to(device)
                        # [1, 1, 200, 200]

                        out_person, out_car, out_bus, out_truck, mem_states = model(
                            input_frame, mem_states
                        )

                        # Only evaluate after overlap (matching training)
                        if step >= overlap:
                            outputs = [
                                out_person.view(64, 64),
                                out_car.view(64, 64),
                                out_bus.view(64, 64),
                                out_truck.view(64, 64),
                            ]

                            frame_targets = [t[step].to(device) for t in target_seq]

                            # MSE loss per class
                            for c in range(4):
                                class_losses[c] += loss_fn(outputs[c], frame_targets[c]).item()

                                # Detection metrics
                                tp, fp, fn = compute_detection_metrics(
                                    outputs[c], frame_targets[c], threshold=0.03
                                )
                                class_tp[c] += tp
                                class_fp[c] += fp
                                class_fn[c] += fn

                                # Prediction statistics
                                pred_np = outputs[c].cpu().numpy()
                                class_max_vals[c].append(pred_np.max())
                                class_mean_vals[c].append(pred_np.mean())
                                class_nonzero_pcts[c].append(
                                    (pred_np != 0).sum() / pred_np.size * 100
                                )

                            total_frames += 1
                            clip_frames += 1

                            # Save comparison images periodically
                            if save_images and total_frames % 200 == 0:
                                save_comparison_image(
                                    event_seq[step],
                                    frame_targets,
                                    outputs,
                                    total_frames,
                                    img_dir,
                                )

                    start_idx += sequence_length - overlap

                print(f"    Evaluated {clip_frames} frames")

    elapsed = time.time() - start_time
    print(f"\nEvaluation complete: {total_frames} frames in {elapsed:.1f}s")

    # ==========================================================================
    # COMPUTE AND PRINT RESULTS
    # ==========================================================================

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS — SNN+ViT HYBRID MODEL")
    print("=" * 70)

    # Average MSE
    avg_total = sum(class_losses) / total_frames
    print(f"\nAverage MSE Loss: {avg_total:.3f}")
    for c in range(4):
        print(f"  {class_names[c]:8s}: {class_losses[c] / total_frames:.3f}")

    # Prediction statistics
    print(f"\nPrediction Output Statistics:")
    print(f"  {'Class':8s} | {'Max (mean)':>10s} | {'Max (max)':>9s} | {'Mean (mean)':>11s} | {'Nonzero %':>9s}")
    print(f"  {'-'*8:8s}-+-{'-'*10:10s}-+-{'-'*9:9s}-+-{'-'*11:11s}-+-{'-'*9:9s}")
    for c in range(4):
        print(f"  {class_names[c]:8s} | {np.mean(class_max_vals[c]):10.3f} | "
              f"{np.max(class_max_vals[c]):9.3f} | "
              f"{np.mean(class_mean_vals[c]):11.3f} | "
              f"{np.mean(class_nonzero_pcts[c]):8.1f}%")

    # Detection metrics
    print(f"\nDetection Metrics (threshold=0.03):")
    print(f"  {'Class':8s} | {'TP':>7s} | {'FP':>10s} | {'FN':>7s} | {'Prec':>5s} | {'Recall':>6s} | {'F1':>5s}")
    print(f"  {'-'*8:8s}-+-{'-'*7:7s}-+-{'-'*10:10s}-+-{'-'*7:7s}-+-{'-'*5:5s}-+-{'-'*6:6s}-+-{'-'*5:5s}")

    results = {}
    for c in range(4):
        prec = class_tp[c] / (class_tp[c] + class_fp[c]) if (class_tp[c] + class_fp[c]) > 0 else 0
        rec = class_tp[c] / (class_tp[c] + class_fn[c]) if (class_tp[c] + class_fn[c]) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        print(f"  {class_names[c]:8s} | {int(class_tp[c]):7d} | {int(class_fp[c]):10d} | "
              f"{int(class_fn[c]):7d} | {prec:.3f} | {rec:6.3f} | {f1:.3f}")

        results[class_names[c]] = {
            'mse': class_losses[c] / total_frames,
            'tp': int(class_tp[c]), 'fp': int(class_fp[c]), 'fn': int(class_fn[c]),
            'precision': prec, 'recall': rec, 'f1': f1,
            'max_mean': float(np.mean(class_max_vals[c])),
            'max_max': float(np.max(class_max_vals[c])),
            'mean_mean': float(np.mean(class_mean_vals[c])),
            'nonzero_pct': float(np.mean(class_nonzero_pcts[c])),
        }

    # Save results JSON
    results_path = os.path.join(eval_dir, 'eval_results.json')
    results['total_frames'] = total_frames
    results['avg_mse'] = avg_total
    results['checkpoint'] = checkpoint_path
    results['model'] = 'SNN_ViT'

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    if save_images:
        print(f"Comparison images saved to: {img_dir}")


# =============================================================================
# CLI
# =============================================================================

parser = argparse.ArgumentParser(
    description="Evaluate SNN+ViT hybrid model",
    usage="%(prog)s <data_dir/> <model_dir/> [options]",
)
parser.add_argument("data_dir",
                    help="Path to training data (e.g. data/raw_scaled/)")
parser.add_argument("model_dir",
                    help="Path to model output dir containing .pth files")
parser.add_argument("--gpu", type=int, default=0,
                    help="CUDA device ID (default: 0)")
parser.add_argument("--save_images", action="store_true",
                    help="Save visual comparison images every 200 frames")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Specific checkpoint filename (default: auto-find best)")

args = parser.parse_args()

device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(args.gpu)

evaluate(args.data_dir, args.model_dir, device, args.save_images, args.checkpoint)
