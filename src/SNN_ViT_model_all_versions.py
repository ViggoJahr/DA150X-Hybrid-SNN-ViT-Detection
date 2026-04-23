"""SNN + ViT Hybrid Model for Event-Based Traffic Detection
DA150X - KTH Royal Institute of Technology
Authors: Axel Prander & Viggo Jahr

This script replaces the FC classification heads in SNN_final_model.py
with a pre-trained ViT-Tiny head from timm. The SNN conv backbone
(conv1-conv3 + LIF neurons) is kept unchanged.

Supported Architectures:
  - v1: ViT (4 layers) + Linear Patch Embedding
  - v2: ViT (4 layers) + Multi-Scale Patch Embedding (MSPE)
  - v3: ViT (2 layers) + Multi-Scale Patch Embedding (MSPE)

Usage:
  Assuming you are running this script from the 'src/' directory.
  The model reads training files from '../data/raw_scaled/' and 
  saves all checkpoints and JSON stats to '../data/processed/experiments/'.

  # Train version 1 (default) with frozen transformer (Phase 1)
  CUDA_VISIBLE_DEVICES=3 python3 SNN_ViT_train.py \
      ../data/raw_scaled/ ../data/processed/experiments/ \
      --gpu 0 --epoch 10 --phase 1

  # Fine-tune version 2 (Phase 2) from a previous checkpoint
  CUDA_VISIBLE_DEVICES=3 python3 SNN_ViT_train.py \
      ../data/raw_scaled/ ../data/processed/experiments/ \
      --gpu 0 --epoch 40 --phase 2 --version v2 \
      --checkpoint ../data/processed/experiments/<run_dir>/multiclass-adamw-<epoch>-<loss>.pth
"""

import argparse
import datetime
import os
from pathlib import Path
import random
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from data_loading import get_data
from norse.torch import LILinearCell
from norse.torch.module.lif import LIFCell, LIFParameters
import torch.nn.functional as F
import numpy as np
import gc
import json
import timm

from torch.optim.lr_scheduler import ReduceLROnPlateau

# =============================================================================
# HYPERPARAMETERS
# =============================================================================
sequence_length, overlap, batch_size = 60, 25, 12
num_inputs = 256 * 256
num_outputs = 64 * 64  # 4096

w_decay = 1e-4
lr = 1e-4
lr_pretrained = 1e-5  # Lower LR for pre-trained transformer blocks

loss_function = nn.MSELoss()
tau_mem = 180



# =============================================================================
# ViT HEATMAP HEAD
# =============================================================================

class MultiScalePatchEmbed(nn.Module):
    """
    Multi-Scale Patch Embedding (MSPE) module.
    Replaces the standard linear projection with three parallel convolutions
    (1x1, 3x3, 5x5) to capture both fine details and larger structures before
    the Transformer processes the patches.
    """ 
    def __init__(self, in_channels=8, embed_dim=192):
        super().__init__()
        # Divide the embedding dimension evenly across three scales
        out_dim = embed_dim // 3 
        
        # Three parallel convolutional "magnifying glasses"
        self.conv_small = nn.Conv2d(in_channels, out_dim, kernel_size=1, padding=0)
        self.conv_mid   = nn.Conv2d(in_channels, out_dim, kernel_size=3, padding=1)
        self.conv_large = nn.Conv2d(in_channels, out_dim, kernel_size=5, padding=2)

    def forward(self, x):
        # Input 'x' is spk3 from SNN: [Batch, 8, 20, 20]
        x_small = self.conv_small(x)
        x_mid = self.conv_mid(x)
        x_large = self.conv_large(x)
        
        # Concatenate along the channel dimension -> [B, 192, 20, 20]
        x_multi = torch.cat([x_small, x_mid, x_large], dim=1) 
        
        # Flatten for the Transformer: [B, 192, 20, 20] -> [B, 400, 192]
        x_flat = x_multi.flatten(2).transpose(1, 2)
        return x_flat

class ViTHeatmapHead(nn.Module):
    """
    Vision Transformer head that decodes SNN spiking features into heatmaps.
    Supports versions v1, v2, and v3 based on depth and patch embedding strategy.

    """
    def __init__(self, in_channels=8, grid_size=20, num_classes=4, output_size=64, version='v1'):
        super().__init__()
        self.grid_size = grid_size
        self.num_classes = num_classes
        self.version = version

        vit = timm.create_model('vit_tiny_patch16_224', pretrained=True)
        embed_dim = vit.embed_dim  # 192

        # Depth: 4 layers for v1 & v2. 2 layers for v3.
        depth = 4 if version in ['v1', 'v2'] else 2
        self.blocks = vit.blocks[:depth]
        self.norm = vit.norm

        # Patch embedding: Linear for v1. Multi-Scale for v2 & v3.
        if version == 'v1':
            self.patch_embed = nn.Linear(in_channels, embed_dim)
        else:
            self.patch_embed = MultiScalePatchEmbed(in_channels, embed_dim)
        
        self.pos_embed = nn.Parameter(torch.randn(1, grid_size * grid_size, embed_dim) * 0.02)

        # Dimension reduction
        self.to_spatial = nn.Sequential(nn.Linear(embed_dim, 64), nn.GELU())

        # Decoder to upsample back to 64x64 heatmaps
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GELU(),
            nn.Conv2d(16, num_classes, kernel_size=3, padding=1),
            # Guarantees exact 64x64 output regardless of intermediate dimensions
            nn.AdaptiveAvgPool2d(output_size), 
        )
        del vit

    def forward(self, spk3):
        B, C, H, W = spk3.shape

        # --- Patch Embedding ---
        # v1 uses nn.Linear which expects the channel dimension last: [Batch, Num_Patches, Channels].
        # Therefore, we flatten the spatial dimensions (H, W) and transpose.
        # v2/v3 use nn.Conv2d (inside MSPE), which expects the original spatial format: [Batch, Channels, H, W].
        if self.version == 'v1':
            x = spk3.flatten(2).transpose(1, 2)
            x = self.patch_embed(x)
        else:
            x = self.patch_embed(spk3)

        
        # Transformer Blocks
        x = x + self.pos_embed
        x = self.blocks(x)
        x = self.norm(x)

        # Reshape back to spatial dimensions (20x20)
        x = self.to_spatial(x)
        x = x.transpose(1, 2).reshape(B, 64, self.grid_size, self.grid_size)

        # Final decoding to heatmaps
        x = self.decoder(x)
        return x

# =============================================================================
# HYBRID SNN + ViT MODEL
# =============================================================================

class SNNViT(nn.Module):
    """
    Hybrid SNN-ViT model (v3) for event-based vehicle detection.

    The SNN conv backbone (conv1-conv3 with LIF neurons) is identical to
    SNN_final_model.py. The FC classification heads are replaced with a
    single ViT head that outputs all 4 class heatmaps at once.
    """

    def __init__(self, version='v1'):
        super(SNNViT, self).__init__()

        # --- SNN Conv Backbone (unchanged from SNN_final_model.py) ---
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

        # --- ViT Classification Head (replaces 4 FC branches) ---
        self.vit_head = ViTHeatmapHead(
            in_channels=8,
            grid_size=20,
            num_classes=4,
            output_size=64,
            version=version
        )

    def forward(self, x, mem_states):
        batch_size, C, W, H = x.shape
        x = (x != 0).float()  # Ensure binary input

        mem1, mem2, mem3 = mem_states

        # --- Conv backbone (identical to SNN_final_model.py) ---
        v1 = self.bn1(self.conv1(x))
        spk1, mem1 = self.lif1(v1, mem1)

        v2 = self.dropout1(self.bn2(self.conv2(self.maxpool(spk1))))
        spk2, mem2 = self.lif2(v2, mem2)

        v3 = self.dropout1(self.bn3(self.conv3(spk2)))
        spk3, mem3 = self.lif3(v3, mem3)

        # --- ViT head (replaces spk3_flat + 4 FC branches) ---
        # spk3: [batch, 8, 20, 20] → heatmaps: [batch, 4, 64, 64]
        heatmaps = self.vit_head(spk3)

        person = heatmaps[:, 0]   # [batch, 64, 64]
        car = heatmaps[:, 1]      # [batch, 64, 64]
        bus = heatmaps[:, 2]      # [batch, 64, 64]
        truck = heatmaps[:, 3]    # [batch, 64, 64]

        return person, car, bus, truck, (mem1, mem2, mem3)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def save_data(model, output_dir, train_loss, validation_loss, val_accuracy,
              tau, epoch, lr_list, save_model, version):
    file_name = os.path.join(output_dir, "multiclass-adamw")

    if save_model:
        torch.save(model.state_dict(),
                   f"{file_name}-{epoch}-{validation_loss[-1][0]}.pth")

    data = {
        "Epoch": epoch,
        "Tau": tau,
        "w_decay": w_decay,
        "lr": lr,
        "lr_pretrained": lr_pretrained,
        "model": f"SNN_ViT_{version}",
        "train_loss": train_loss,
        "validation_loss": validation_loss,
        "lr_schedule": lr_list,
        "validation_accuracy": val_accuracy,
    }

    with open(f"{file_name}.json", "w") as f:
        json.dump(data, f)

    if save_model:
        print(f"Model saved to \x1b[1m{file_name}-{epoch}-{validation_loss[-1][0]}.pth\x1b[22m")
    print(f"Stats saved to \x1b[1m{file_name}.json\x1b[22m")


def loss_fn(output_frame, target_frame):
    return loss_function(output_frame, target_frame * 1000)


def pretty_time(seconds):
    if not seconds:
        return "0s"
    seconds = int(seconds)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    measures = ((hours, "h"), (minutes, "m"), (seconds, "s"))
    return " ".join([f"{count}{unit}" for (count, unit) in measures if count])


def chunker(seq, size):
    return (seq[pos: pos + size] for pos in range(0, len(seq), size))


# =============================================================================
# DATA DIRECTORIES
# =============================================================================
data_dirs = [
    "week_32-box_3",
    "week_33-box_2",
    "week_34-box_1",
    "week_35-box_2",
    "week_36-box_3",
]


# =============================================================================
# TRAINING FUNCTION
# =============================================================================

def start_training(training_data_dir, output_dir, num_epochs, phase, checkpoint_path, snn_backbone_path=None, version='v1'):

    cur_time = datetime.datetime.now()
    output_dir = os.path.join(
        output_dir, f"vit-{cur_time.month}-{cur_time.day}-{cur_time.hour}-{cur_time.minute}"
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SNNViT(version=version).to(device)

    # INJECT PRE-TRAINED SNN BACKBONE
    if snn_backbone_path and os.path.exists(snn_backbone_path):
        print(f"Loading pre-trained SNN backbone from: {snn_backbone_path}")
        old_state_dict = torch.load(snn_backbone_path, map_location=device)
        
        # Filter out the FC layers. Keep only conv and batch norm weights
        pretrained_dict = {k: v for k, v in old_state_dict.items() if 'conv' in k or 'bn' in k}
        
        # Overwrite the random conv weights with the pre-trained ones
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print(f"Successfully injected {len(pretrained_dict)} pre-trained SNN layers.")

    # Load checkpoint if provided (for phase 2)
    if checkpoint_path:
        print(f"Loading checkpoint: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # =========================================================================
    # PHASE-DEPENDENT TRAINING SETUP
    # =========================================================================
    if phase == 1:
        print("=" * 60)
        print("PHASE 1: Frozen transformer AND Frozen SNN — training adapters only")
        print("=" * 60)

        # Freeze pre-trained transformer blocks
        for param in model.vit_head.blocks.parameters():
            param.requires_grad = False
        for param in model.vit_head.norm.parameters():
            param.requires_grad = False
        
        # Freeze the pre-trained SNN backbone
        for name, param in model.named_parameters():
            if 'conv' in name or 'bn' in name:
                param.requires_grad = False

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=w_decay, eps=1e-8
        )

    elif phase == 2:
        print("=" * 60)
        print("PHASE 2: Full fine-tuning — all layers unfrozen")
        print("=" * 60)

        # Unfreeze everything
        for param in model.parameters():
            param.requires_grad = True

        # Separate parameter groups: lower LR for pre-trained blocks
        pretrained_params = []
        new_params = []

        for name, param in model.named_parameters():
            if 'vit_head.blocks' in name or 'vit_head.norm' in name:
                pretrained_params.append(param)
            else:
                new_params.append(param)

        optimizer = torch.optim.AdamW([
            {'params': new_params, 'lr': lr},
            {'params': pretrained_params, 'lr': lr_pretrained},
        ], weight_decay=w_decay, eps=1e-8)

        print(f"  New params LR:         {lr}")
        print(f"  Pre-trained params LR: {lr_pretrained}")

    else:
        # Default: train everything at same LR (simple mode)
        print("=" * 60)
        print("TRAINING: All layers, single learning rate")
        print("=" * 60)
        for param in model.parameters():
            param.requires_grad = True
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=w_decay, eps=1e-8
        )

    scheduler = ReduceLROnPlateau(optimizer, "min", patience=3, factor=0.5)

    # Print parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen parameters:    {frozen_params:,}")
    print(f"  Scheduler patience:   3 (increased from baseline's 0)")
    print("=" * 60)

    # =========================================================================
    # DATA LOADING
    # =========================================================================
    data_files = []
    for curr_dir in data_dirs:
        dir_path = os.path.join(training_data_dir, curr_dir)
        if os.path.exists(dir_path):
            data_files.extend([
                os.path.join(dir_path, f)
                for f in os.listdir(dir_path)
                if f.endswith(".pt")
            ])

    print(f"Found {len(data_files)} training files")
    random.shuffle(data_files)

    # =========================================================================
    # TRAINING LOOP
    # =========================================================================
    train_loss_list = []
    val_loss_list = []
    lr_list = []
    val_acc_list = []
    best_val = 999999

    for epoch in range(num_epochs):

        # --- TRAINING ---
        total_train_loss = 0
        person_train_loss = 0
        car_train_loss = 0
        buss_train_loss = 0
        truck_train_loss = 0
        num_train_batches = 0

        model.train()
        start = time.time()

        for i, data_paths in enumerate(chunker(data_files, 4)):
            data = get_data(data_paths)

            if data is not None:
                train_data, val_data = data

                print(
                    f"\r\x1b[2KEpoch {epoch + 1} | Chunk \x1b[1m{i}/{len(data_files) // 4}\x1b[22m Train | "
                    f"loss: \x1b[1m{total_train_loss / (num_train_batches + 1e-8):.3f}\x1b[22m "
                    f"person \x1b[1m{person_train_loss / (num_train_batches + 1e-8):.3f}\x1b[22m | "
                    f"car \x1b[1m{car_train_loss / (num_train_batches + 1e-8):.3f}\x1b[22m | "
                    f"bus \x1b[1m{buss_train_loss / (num_train_batches + 1e-8):.3f}\x1b[22m | "
                    f"truck \x1b[1m{truck_train_loss / (num_train_batches + 1e-8):.3f}\x1b[22m | "
                    f"\x1b[1m{pretty_time(time.time() - start)}\x1b[22m",
                    end="",
                )

                for i, (frames, targets) in enumerate(train_data):
                    # Only 3 membrane states now (not 11)
                    mem_states = (None, None, None)

                    optimizer.zero_grad()
                    frames, targets = frames.to(device), targets.to(device)

                    loss = 0
                    person_loss = 0
                    car_loss = 0
                    buss_loss = 0
                    truck_loss = 0

                    for step in range(sequence_length):
                        input_frame = frames[:, step].unsqueeze(1)
                        output1, output2, output3, output4, mem_states = model(
                            input_frame, mem_states
                        )

                        if step >= overlap:
                            final_output1 = output1.view(-1, 64, 64)
                            final_output2 = output2.view(-1, 64, 64)
                            final_output3 = output3.view(-1, 64, 64)
                            final_output4 = output4.view(-1, 64, 64)

                            person_loss += (
                                0.7 * 2
                                * loss_fn(final_output1, targets[:, 0, step])
                                / (sequence_length - overlap)
                            )
                            car_loss += (
                                0.5 * 2
                                * loss_fn(final_output2, targets[:, 1, step])
                                / (sequence_length - overlap)
                            )
                            buss_loss += (
                                4 * 6
                                * loss_fn(final_output3, targets[:, 2, step])
                                / (sequence_length - overlap)
                            )
                            truck_loss += (
                                4.2 * 6
                                * loss_fn(final_output4, targets[:, 3, step])
                                / (sequence_length - overlap)
                            )

                    loss = person_loss + car_loss + buss_loss + truck_loss

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 20.0)
                    optimizer.step()

                    total_train_loss += loss.item()
                    person_train_loss += person_loss.item()
                    car_train_loss += car_loss.item()
                    buss_train_loss += buss_loss.item()
                    truck_train_loss += truck_loss.item()
                    num_train_batches += 1

        # --- VALIDATION ---
        model.eval()
        total_val_loss = 0
        person_val_loss = 0
        car_val_loss = 0
        buss_val_loss = 0
        truck_val_loss = 0
        val_acc = 0
        num_test_batches = 0

        with torch.no_grad():
            for i, data_paths in enumerate(chunker(data_files, 4)):
                data = get_data(data_paths)

                if data is not None:
                    train_data, val_data = data

                    print(
                        f"\r\x1b[2KEpoch {epoch + 1} | Chunk \x1b[1m{i}/{len(data_files) // 4}\x1b[22m Val | "
                        f"loss: \x1b[1m{total_val_loss / (num_test_batches + 1e-8):.3f}\x1b[22m "
                        f"person \x1b[1m{person_val_loss / (num_test_batches + 1e-8):.3f}\x1b[22m | "
                        f"car \x1b[1m{car_val_loss / (num_test_batches + 1e-8):.3f}\x1b[22m | "
                        f"bus \x1b[1m{buss_val_loss / (num_test_batches + 1e-8):.3f}\x1b[22m | "
                        f"truck \x1b[1m{truck_val_loss / (num_test_batches + 1e-8):.3f}\x1b[22m | "
                        f"\x1b[1m{pretty_time(time.time() - start)}\x1b[22m",
                        end="",
                    )

                    for i, (frames, targets) in enumerate(val_data):
                        mem_states = (None, None, None)
                        frames, targets = frames.to(device), targets.to(device)

                        loss = 0
                        person_loss = 0
                        car_loss = 0
                        buss_loss = 0
                        truck_loss = 0

                        for step in range(sequence_length):
                            input_frame = frames[:, step].unsqueeze(1)
                            output1, output2, output3, output4, mem_states = model(
                                input_frame, mem_states
                            )

                            final_output1 = output1.view(-1, 64, 64)
                            final_output2 = output2.view(-1, 64, 64)
                            final_output3 = output3.view(-1, 64, 64)
                            final_output4 = output4.view(-1, 64, 64)

                            person_loss += (
                                0.7 * 2
                                * loss_fn(final_output1, targets[:, 0, step])
                                / (sequence_length - overlap)
                            )
                            car_loss += (
                                0.5 * 2
                                * loss_fn(final_output2, targets[:, 1, step])
                                / (sequence_length - overlap)
                            )
                            buss_loss += (
                                4 * 6 
                                * loss_fn(final_output3, targets[:, 2, step])
                                / (sequence_length - overlap)
                            )
                            truck_loss += (
                                4.2 * 6
                                * loss_fn(final_output4, targets[:, 3, step])
                                / (sequence_length - overlap)
                            )

                        loss = person_loss + car_loss + buss_loss + truck_loss

                        total_val_loss += loss.item()
                        person_val_loss += person_loss.item()
                        car_val_loss += car_loss.item()
                        buss_val_loss += buss_loss.item()
                        truck_val_loss += truck_loss.item()
                        num_test_batches += 1

        scheduler.step(total_val_loss / num_test_batches)
        del data
        gc.collect()

        # --- EPOCH SUMMARY ---
        epoch_time = time.time() - start
        print(
            f"\x1b[2K\x1b[0GEpoch {epoch + 1} | "
            f"train: \x1b[1m{total_train_loss / num_train_batches:.3f}\x1b[22m | "
            f"val: \x1b[1m{total_val_loss / num_test_batches:.3f}\x1b[22m | "
            f"person \x1b[1m{person_val_loss / num_test_batches:.3f}\x1b[22m | "
            f"car \x1b[1m{car_val_loss / num_test_batches:.3f}\x1b[22m | "
            f"bus \x1b[1m{buss_val_loss / num_test_batches:.3f}\x1b[22m | "
            f"truck \x1b[1m{truck_val_loss / num_test_batches:.3f}\x1b[22m | "
            f"\x1b[1m{pretty_time(epoch_time)}\x1b[22m"
        )

        # --- LOGGING ---
        train_loss_list.append([
            np.round(total_train_loss / num_train_batches, 4),
            np.round(person_train_loss / num_train_batches, 4),
            np.round(car_train_loss / num_train_batches, 4),
            np.round(buss_train_loss / num_train_batches, 4),
            np.round(truck_train_loss / num_train_batches, 4),
        ])
        val_loss_list.append([
            np.round(total_val_loss / num_test_batches, 4),
            np.round(person_val_loss / num_test_batches, 4),
            np.round(car_val_loss / num_test_batches, 4),
            np.round(buss_val_loss / num_test_batches, 4),
            np.round(truck_val_loss / num_test_batches, 4),
        ])
        lr_list.append(scheduler.get_last_lr())
        val_acc_list.append(np.round(val_acc / num_test_batches, 4))

        # --- SAVE ---
        is_best = total_val_loss / num_test_batches < best_val
        if is_best:
            best_val = total_val_loss / num_test_batches

        save_data(
            model, output_dir, train_loss_list, val_loss_list, val_acc_list, 
            tau_mem, epoch, lr_list, 
            is_best, version
        )

    # Save final model
    save_data(
        model, output_dir, train_loss_list, 
        val_loss_list, val_acc_list, tau_mem, 
        epoch, lr_list, True, version
    )

# =============================================================================
# CLI
# =============================================================================
parser = argparse.ArgumentParser(
    description="SNN + ViT Hybrid Trainer for traffic monitoring",
    usage="%(prog)s <training_data/> <output_dir/> [options]",
)
parser.add_argument("input_dir",
                    help="Path to training data (e.g. ../data/raw_scaled/)")
parser.add_argument("output_dir",
                    help="Path to save model checkpoints (e.g. ../data/processed/experiments/)")
parser.add_argument("--epoch", default=10, type=int,
                    help="Number of epochs (default: 10)")
parser.add_argument("--gpu", type=int, default=0,
                    help="CUDA device ID (default: 0)")
parser.add_argument("--phase", type=int, default=0, choices=[0, 1, 2],
                    help="Training phase: 0=all layers same LR, "
                         "1=frozen transformer, 2=full fine-tune with dual LR")
parser.add_argument("--version", type=str, default="v1", choices=["v1", "v2", "v3"],
                    help="Architecture version: v1=ViT(4)+Linear, v2=ViT(4)+MSPE, v3=ViT(2)+MSPE (default: v1)")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Path to checkpoint .pth to resume from (for phase 2)")
parser.add_argument("--snn_backbone", type=str, default=None,
                    help="Path to pre-trained SNN baseline .pth file")

if __name__ == "__main__":
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu)

    start_training(args.input_dir, args.output_dir, args.epoch, args.phase, args.checkpoint, args.snn_backbone, args.version)