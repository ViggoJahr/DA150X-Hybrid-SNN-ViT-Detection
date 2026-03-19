"""
SNN + ViT Hybrid Model for Event-Based Vehicle Detection
DA150X - KTH Royal Institute of Technology
Authors: Axel Prander & Viggo Jahr

This script replaces the FC classification heads in SNN_final_model.py
with a pre-trained ViT-Tiny head from timm. The SNN conv backbone
(conv1-conv3 + LIF neurons) is kept unchanged.

Architecture:
  Input: 200x200 event frame (binary)
    → Conv backbone (3 conv + LIF layers) → spk3: [batch, 8, 20, 20]
    → ViT head (pre-trained ViT-Tiny transformer blocks)
    → 4 heatmaps: [batch, 4, 64, 64] (person, car, bus, truck)

Usage:
  # Phase 1: frozen transformer, train adapters only
  CUDA_VISIBLE_DEVICES=3 python3 SNN_ViT_model.py \
      data/training_output_scaled/ data/model_output/vit/ \
      --gpu 0 --epoch 10 --phase 1

  # Phase 2: unfreeze everything, fine-tune
  CUDA_VISIBLE_DEVICES=3 python3 SNN_ViT_model.py \
      data/training_output_scaled/ data/model_output/vit/ \
      --gpu 0 --epoch 40 --phase 2 \
      --checkpoint data/model_output/vit/<run_dir>/multiclass-adamw-<epoch>-<loss>.pth
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

class ViTHeatmapHead(nn.Module):
    """
    Vision Transformer head that replaces the FC classification branches.

    Takes the SNN conv backbone output spk3 [batch, 8, 20, 20] and produces
    4-class heatmaps [batch, 4, 64, 64].

    Uses pre-trained ViT-Tiny transformer blocks (12 layers, 192 dim, 3 heads)
    from timm. The patch embedding and output decoder are custom.

    Pre-trained blocks: ~5.5M params (from ImageNet)
    Custom layers: ~150K params (trained from scratch)
    Total: ~5.7M params

    Compare to FC heads: ~15.5M params
    """

    def __init__(self, in_channels=8, grid_size=20,
                 num_classes=4, output_size=64):
        super().__init__()
        self.grid_size = grid_size
        self.num_classes = num_classes

        # --- Load pre-trained ViT-Tiny ---
        vit = timm.create_model('vit_tiny_patch16_224', pretrained=True)
        embed_dim = vit.embed_dim  # 192

        # KEEP: pre-trained transformer blocks + final norm
        self.blocks = vit.blocks[:4]   # 12 transformer encoder layers SKA DET INTE VARA 4 BARA?
        self.norm = vit.norm       # LayerNorm

        # CUSTOM: patch embedding (8 spike channels → 192 dim)
        # Replaces vit.patch_embed which expects 3-channel 16x16 RGB patches
        self.patch_embed = nn.Linear(in_channels, embed_dim)

        # CUSTOM: positional embeddings for 400 patches (20x20 grid)
        # Replaces vit.pos_embed which is sized for 196 patches (14x14)
        self.pos_embed = nn.Parameter(
            torch.randn(1, grid_size * grid_size, embed_dim) * 0.02
        )

        # CUSTOM: decode transformer output back to spatial heatmaps
        # Replaces vit.head which is FC → 1000 ImageNet classes
        self.to_spatial = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
        )

        self.decoder = nn.Sequential(
            # 20x20 → 40x40
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.GELU(),
            # 40x40 → 80x80
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.GELU(),
            # 80x80 → 80x80 (channel reduction to num_classes)
            nn.Conv2d(16, num_classes, kernel_size=3, padding=1),
            # 80x80 → 64x64
            nn.AdaptiveAvgPool2d(output_size),
        )

        # Free the parts we don't need
        del vit

    def forward(self, spk3):
        """
        Args:
            spk3: [batch, 8, 20, 20] spike tensor from conv backbone

        Returns:
            heatmaps: [batch, 4, 64, 64] predicted heatmaps
                      channel 0=person, 1=car, 2=bus, 3=truck
        """
        B, C, H, W = spk3.shape

        # Flatten spatial positions into a sequence of patches
        # [B, 8, 20, 20] → [B, 400, 8]
        x = spk3.flatten(2).transpose(1, 2)

        # Project to transformer dimension + add positional info
        # [B, 400, 8] → [B, 400, 192]
        x = self.patch_embed(x) + self.pos_embed

        # Run through pre-trained transformer blocks
        # [B, 400, 192] → [B, 400, 192]
        x = self.blocks(x)
        x = self.norm(x)

        # Project down and reshape back to spatial grid
        # [B, 400, 192] → [B, 400, 64] → [B, 64, 20, 20]
        x = self.to_spatial(x)
        x = x.transpose(1, 2).reshape(B, 64, self.grid_size, self.grid_size)

        # Upsample to output resolution
        # [B, 64, 20, 20] → [B, 4, 64, 64]
        x = self.decoder(x)

        return x


# =============================================================================
# HYBRID SNN + ViT MODEL
# =============================================================================

class SNNViT(nn.Module):
    """
    Hybrid SNN-ViT model for event-based vehicle detection.

    The SNN conv backbone (conv1-conv3 with LIF neurons) is identical to
    SNN_final_model.py. The FC classification heads are replaced with a
    single ViT head that outputs all 4 class heatmaps at once.

    Backbone: ~2.6K params (conv layers + batch norm)
    ViT Head: ~5.7M params (pre-trained transformer + custom adapter)
    Total:    ~5.7M params

    Compare to original SNN with FC heads: ~15.5M params
    """

    def __init__(self):
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
              tau, epoch, lr_list, save_model):
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
        "model": "SNN_ViT",
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

def start_training(training_data_dir, output_dir, num_epochs, phase, checkpoint_path, snn_backbone_path=None):

    cur_time = datetime.datetime.now()
    output_dir = os.path.join(
        output_dir, f"vit-{cur_time.month}-{cur_time.day}-{cur_time.hour}-{cur_time.minute}"
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SNNViT().to(device)

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
            model, output_dir, train_loss_list, val_loss_list,
            val_acc_list, tau_mem, epoch, lr_list, is_best,
        )

    # Save final model
    save_data(
        model, output_dir, train_loss_list, val_loss_list,
        val_acc_list, tau_mem, epoch, lr_list, True,
    )


# =============================================================================
# CLI
# =============================================================================
parser = argparse.ArgumentParser(
    description="SNN + ViT Hybrid Trainer for traffic monitoring",
    usage="%(prog)s <training_data/> <output_dir/> [options]",
)
parser.add_argument("input_dir",
                    help="Path to training data (e.g. data/training_output_scaled/)")
parser.add_argument("output_dir",
                    help="Path to save model checkpoints")
parser.add_argument("--epoch", default=10, type=int,
                    help="Number of epochs (default: 10)")
parser.add_argument("--gpu", type=int, default=0,
                    help="CUDA device ID (default: 0)")
parser.add_argument("--phase", type=int, default=0, choices=[0, 1, 2],
                    help="Training phase: 0=all layers same LR, "
                         "1=frozen transformer, 2=full fine-tune with dual LR")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Path to checkpoint .pth to resume from (for phase 2)")
parser.add_argument("--snn_backbone", type=str, default=None,
                    help="Path to pre-trained SNN baseline .pth file")

args = parser.parse_args()
torch.cuda.set_device(args.gpu)

start_training(args.input_dir, args.output_dir, args.epoch,
               args.phase, args.checkpoint, args.snn_backbone)
