#!/usr/bin/env python3
"""
Experiment Runner for DA150X SNN/ViT Training
KTH Royal Institute of Technology
Authors: Axel Prander & Viggo Jahr

Queues and runs multiple training experiments overnight with different
hyperparameters. Each experiment gets its own output directory, logs, and
JSON stats. If one run crashes, the runner continues to the next.

Features:
  - Define experiments as a list of configs
  - Each run gets a unique named directory
  - Full stdout/stderr logging per run
  - Crash-resilient: continues to next experiment on failure
  - Summary table printed at the end
  - Supports both SNN baseline and ViT models
  - GPU selection per experiment
  - Optional checkpoint resuming for ViT Phase 2

Usage:
  # Edit the EXPERIMENTS list below, then:
  CUDA_VISIBLE_DEVICES=3 python3 run_experiments.py --gpu 0

  # Or run in tmux for overnight:
  tmux new -s experiments
  CUDA_VISIBLE_DEVICES=3 python3 run_experiments.py --gpu 0
  # Ctrl+B then D

  # Check results next day:
  python3 run_experiments.py --summary_only

How to customize:
  Scroll down to the EXPERIMENTS list and edit/add/remove entries.
  Each experiment is a dictionary with the settings you want to change.
  Anything not specified uses the defaults at the top.
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
import time
import copy
from pathlib import Path
import tqdm

# =============================================================================
# DEFAULT HYPERPARAMETERS
# =============================================================================
# These are used unless overridden per experiment.

DEFAULTS = {
    # Model selection
    "model": "vit",               # "snn" or "vit"

    # Paths (relative to where you run from)
    "data_dir": "../data/raw_scaled/",
    "output_base": "../data/processed/experiments/",

    # Training
    "epochs": 50,
    "lr": 1e-4,
    "lr_pretrained": 1e-5,        # Only used for ViT phase 2
    "batch_size": 12,
    "w_decay": 1e-4,
    "dropout1": 0.3,
    "dropout2": 0.3,
    "tau_mem": 180,
    "sequence_length": 60,
    "overlap": 25,
    "grad_clip": 20.0,
    "scheduler_patience": 3,

    # ViT specific
    "vit_layers": 4,              # Number of transformer layers (of 12)
    "phase": 0,                   # 0=single LR, 1=frozen, 2=dual LR
    "checkpoint": None,           # Path to checkpoint for resuming

    # Loss weights
    "person_weight": 1.4,         # 0.7 * 2
    "car_weight": 1.0,            # 0.5 * 2
    "bus_weight": 24.0,           # 4 * 6
    "truck_weight": 25.2,         # 4.2 * 6
}


# =============================================================================
# EXPERIMENTS TO RUN
# =============================================================================
# Each dict overrides specific defaults. Only include what you want to change.
# Experiments run in order. Give each a unique "name".
#
# EDIT THIS LIST to set up your overnight runs.
# =============================================================================

EXPERIMENTS = [
    # -----------------------------------------------------------------
    # EXAMPLE 1: ViT Phase 0, lr=1e-4 (baseline ViT run)
    # -----------------------------------------------------------------
    {
        "name": "vit_phase0_lr1e-4",
        "model": "vit",
        "phase": 0,
        "lr": 1e-4,
        "epochs": 50,
        "vit_layers": 4,
        "batch_size": 12,
    },

    # -----------------------------------------------------------------
    # EXAMPLE 2: ViT Phase 0, lr=5e-4 (higher LR comparison)
    # -----------------------------------------------------------------
    {
        "name": "vit_phase0_lr5e-4",
        "model": "vit",
        "phase": 0,
        "lr": 5e-4,
        "epochs": 50,
        "vit_layers": 4,
        "batch_size": 12,
    },

    # -----------------------------------------------------------------
    # EXAMPLE 3: ViT with 2 layers (smaller, can use bigger batch)
    # -----------------------------------------------------------------
    # {
    #     "name": "vit_2layers_lr1e-4",
    #     "model": "vit",
    #     "phase": 0,
    #     "lr": 1e-4,
    #     "epochs": 50,
    #     "vit_layers": 2,
    #     "batch_size": 16,
    # },

    # -----------------------------------------------------------------
    # EXAMPLE 4: SNN baseline with lower dropout
    # -----------------------------------------------------------------
    # {
    #     "name": "snn_dropout02",
    #     "model": "snn",
    #     "lr": 5e-4,
    #     "epochs": 50,
    #     "dropout1": 0.2,
    #     "dropout2": 0.2,
    #     "batch_size": 24,
    # },

    # -----------------------------------------------------------------
    # EXAMPLE 5: ViT Phase 2 from checkpoint
    # -----------------------------------------------------------------
    # {
    #     "name": "vit_phase2_from_p1",
    #     "model": "vit",
    #     "phase": 2,
    #     "lr": 1e-4,
    #     "lr_pretrained": 1e-5,
    #     "epochs": 100,
    #     "checkpoint": "src/data/processed/vit/vit-3-13-18-24/multiclass-adamw-8-134.6753.pth",
    # },

    # -----------------------------------------------------------------
    # EXAMPLE 6: ViT with equal class weights (no bus/truck boost)
    # -----------------------------------------------------------------
    # {
    #     "name": "vit_equal_weights",
    #     "model": "vit",
    #     "phase": 0,
    #     "lr": 1e-4,
    #     "epochs": 50,
    #     "person_weight": 1.0,
    #     "car_weight": 1.0,
    #     "bus_weight": 1.0,
    #     "truck_weight": 1.0,
    # },
]


# =============================================================================
# SCRIPT GENERATOR
# =============================================================================
# Generates a temporary Python training script with the exact hyperparameters
# for each experiment, then runs it as a subprocess.

def generate_vit_script(config, script_path):
    """Generate a ViT training script with the given config."""
    script = f'''#!/usr/bin/env python3
"""Auto-generated by run_experiments.py — {config["name"]}"""
import argparse, datetime, os, random, time, gc, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from data_loading import get_data
from norse.torch import LILinearCell
from norse.torch.module.lif import LIFCell, LIFParameters
from torch.optim.lr_scheduler import ReduceLROnPlateau
import timm

# === HYPERPARAMETERS (set by experiment runner) ===
sequence_length = {config["sequence_length"]}
overlap = {config["overlap"]}
batch_size = {config["batch_size"]}
w_decay = {config["w_decay"]}
lr = {config["lr"]}
lr_pretrained = {config["lr_pretrained"]}
tau_mem = {config["tau_mem"]}
loss_function = nn.MSELoss()
VIT_LAYERS = {config["vit_layers"]}
PHASE = {config["phase"]}
GRAD_CLIP = {config["grad_clip"]}
SCHEDULER_PATIENCE = {config["scheduler_patience"]}
PERSON_W = {config["person_weight"]}
CAR_W = {config["car_weight"]}
BUS_W = {config["bus_weight"]}
TRUCK_W = {config["truck_weight"]}
DROPOUT1 = {config["dropout1"]}
DROPOUT2 = {config["dropout2"]}

class ViTHeatmapHead(nn.Module):
    def __init__(self, in_channels=8, grid_size=20, num_classes=4, output_size=64):
        super().__init__()
        self.grid_size = grid_size
        vit = timm.create_model('vit_tiny_patch16_224', pretrained=True)
        embed_dim = vit.embed_dim
        self.blocks = vit.blocks[:VIT_LAYERS]
        self.norm = vit.norm
        self.patch_embed = nn.Linear(in_channels, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, grid_size*grid_size, embed_dim) * 0.02)
        self.to_spatial = nn.Sequential(nn.Linear(embed_dim, 64), nn.GELU())
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.GELU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), nn.GELU(),
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
        self.dropout1 = nn.Dropout(p=DROPOUT1)
        self.vit_head = ViTHeatmapHead(in_channels=8, grid_size=20, num_classes=4, output_size=64)

    def forward(self, x, mem_states):
        bs, C, W, H = x.shape
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

def loss_fn(output, target, step, cid):
    return loss_function(output, target * 1000)

data_dirs = ["week_32-box_3","week_33-box_2","week_34-box_1","week_35-box_2","week_36-box_3"]

def chunker(seq, size):
    return (seq[pos:pos+size] for pos in range(0, len(seq), size))

def pretty_time(s):
    if not s: return "0s"
    s = int(s)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return " ".join([f"{{c}}{{u}}" for c, u in [(h,"h"),(m,"m"),(s,"s")] if c])

training_data_dir = "{config["data_dir"]}"
output_dir = "{config["_output_dir"]}"
num_epochs = {config["epochs"]}
checkpoint_path = {repr(config["checkpoint"])}

Path(output_dir).mkdir(parents=True, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SNNViT().to(device)

if checkpoint_path:
    print(f"Loading checkpoint: {{checkpoint_path}}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

if PHASE == 1:
    for p in model.vit_head.blocks.parameters(): p.requires_grad = False
    for p in model.vit_head.norm.parameters(): p.requires_grad = False
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=w_decay, eps=1e-8)
elif PHASE == 2:
    pretrained_p, new_p = [], []
    for n, p in model.named_parameters():
        (pretrained_p if 'vit_head.blocks' in n or 'vit_head.norm' in n else new_p).append(p)
    optimizer = torch.optim.AdamW([{{"params": new_p, "lr": lr}}, {{"params": pretrained_p, "lr": lr_pretrained}}], weight_decay=w_decay, eps=1e-8)
else:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=w_decay, eps=1e-8)

scheduler = ReduceLROnPlateau(optimizer, "min", patience=SCHEDULER_PATIENCE, factor=0.5)
total_p = sum(p.numel() for p in model.parameters())
train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model: SNNViT | Params: {{total_p:,}} total, {{train_p:,}} trainable | Phase {{PHASE}}")
print(f"LR: {{lr}} | LR_pretrained: {{lr_pretrained}} | ViT layers: {{VIT_LAYERS}} | Batch: {{batch_size}}")
print(f"Class weights: person={{PERSON_W}} car={{CAR_W}} bus={{BUS_W}} truck={{TRUCK_W}}")
print("=" * 60)

data_files = []
for d in data_dirs:
    dp = os.path.join(training_data_dir, d)
    if os.path.exists(dp):
        data_files.extend([os.path.join(dp, f) for f in os.listdir(dp) if f.endswith(".pt")])
print(f"Found {{len(data_files)}} training files")
random.shuffle(data_files)

train_loss_list, val_loss_list, lr_list, val_acc_list = [], [], [], []
best_val = 999999

for epoch in range(num_epochs):
    total_tl, p_tl, c_tl, b_tl, t_tl, n_tb = 0,0,0,0,0,0
    model.train()
    start = time.time()
    for i, dp in enumerate(chunker(data_files, 4)):
        data = get_data(dp)
        if data is not None:
            for _, (frames, targets) in enumerate(data[0]):
                mem = (None, None, None)
                optimizer.zero_grad()
                frames, targets = frames.to(device), targets.to(device)
                pl, cl, bl, tl = 0,0,0,0
                for step in range(sequence_length):
                    o1,o2,o3,o4, mem = model(frames[:,step].unsqueeze(1), mem)
                    if step >= overlap:
                        pl += PERSON_W * loss_fn(o1.view(-1,64,64), targets[:,0,step], step, 0) / (sequence_length - overlap)
                        cl += CAR_W * loss_fn(o2.view(-1,64,64), targets[:,1,step], step, 2) / (sequence_length - overlap)
                        bl += BUS_W * loss_fn(o3.view(-1,64,64), targets[:,2,step], step, 5) / (sequence_length - overlap)
                        tl += TRUCK_W * loss_fn(o4.view(-1,64,64), targets[:,3,step], step, 7) / (sequence_length - overlap)
                loss = pl + cl + bl + tl
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                total_tl += loss.item(); p_tl += pl.item(); c_tl += cl.item(); b_tl += bl.item(); t_tl += tl.item()
                n_tb += 1

    total_vl, p_vl, c_vl, b_vl, t_vl, n_vb = 0,0,0,0,0,0
    model.eval()
    with torch.no_grad():
        for i, dp in enumerate(chunker(data_files, 4)):
            data = get_data(dp)
            if data is not None:
                for _, (frames, targets) in enumerate(data[1]):
                    mem = (None, None, None)
                    frames, targets = frames.to(device), targets.to(device)
                    pl, cl, bl, tl = 0,0,0,0
                    for step in range(sequence_length):
                        o1,o2,o3,o4, mem = model(frames[:,step].unsqueeze(1), mem)
                        pl += PERSON_W * loss_fn(o1.view(-1,64,64), targets[:,0,step], step, 0) / (sequence_length - overlap)
                        cl += CAR_W * loss_fn(o2.view(-1,64,64), targets[:,1,step], step, 2) / (sequence_length - overlap)
                        bl += BUS_W * loss_fn(o3.view(-1,64,64), targets[:,2,step], step, 5) / (sequence_length - overlap)
                        tl += TRUCK_W * loss_fn(o4.view(-1,64,64), targets[:,3,step], step, 7) / (sequence_length - overlap)
                    loss = pl + cl + bl + tl
                    total_vl += loss.item(); p_vl += pl.item(); c_vl += cl.item(); b_vl += bl.item(); t_vl += tl.item()
                    n_vb += 1

    scheduler.step(total_vl / n_vb)
    del data; gc.collect()
    et = time.time() - start
    vl = total_vl/n_vb
    print(f"Epoch {{epoch+1}} | train: {{total_tl/n_tb:.3f}} | val: {{vl:.3f}} | person {{p_vl/n_vb:.3f}} | car {{c_vl/n_vb:.3f}} | bus {{b_vl/n_vb:.3f}} | truck {{t_vl/n_vb:.3f}} | {{pretty_time(et)}}")

    train_loss_list.append([round(total_tl/n_tb,4), round(p_tl/n_tb,4), round(c_tl/n_tb,4), round(b_tl/n_tb,4), round(t_tl/n_tb,4)])
    val_loss_list.append([round(vl,4), round(p_vl/n_vb,4), round(c_vl/n_vb,4), round(b_vl/n_vb,4), round(t_vl/n_vb,4)])
    lr_list.append(scheduler.get_last_lr())
    val_acc_list.append(0)

    fn = os.path.join(output_dir, "multiclass-adamw")
    is_best = vl < best_val
    if is_best: best_val = vl
    save_d = {{"Epoch": epoch, "Tau": tau_mem, "w_decay": w_decay, "lr": lr, "model": "SNN_ViT",
              "vit_layers": VIT_LAYERS, "phase": PHASE, "batch_size": batch_size,
              "class_weights": {{"person": PERSON_W, "car": CAR_W, "bus": BUS_W, "truck": TRUCK_W}},
              "train_loss": train_loss_list, "validation_loss": val_loss_list, "lr_schedule": lr_list,
              "validation_accuracy": val_acc_list}}
    with open(f"{{fn}}.json", "w") as f: json.dump(save_d, f)
    if is_best:
        torch.save(model.state_dict(), f"{{fn}}-{{epoch}}-{{vl:.4f}}.pth")
        print(f"  Best model saved: {{fn}}-{{epoch}}-{{vl:.4f}}.pth")

# Save final
fn = os.path.join(output_dir, "multiclass-adamw")
torch.save(model.state_dict(), f"{{fn}}-{{epoch}}-{{vl:.4f}}.pth")
print(f"\\nTraining complete. Best val_loss: {{best_val:.4f}}")
'''
    with open(script_path, 'w') as f:
        f.write(script)


def generate_snn_script(config, script_path):
    """Generate an SNN baseline training script with the given config."""
    script = f'''#!/usr/bin/env python3
"""Auto-generated by run_experiments.py — {config["name"]}"""
import argparse, datetime, os, random, time, gc, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from data_loading import get_data
from norse.torch import LILinearCell
from norse.torch.module.lif import LIFCell, LIFParameters
from torch.optim.lr_scheduler import ReduceLROnPlateau

sequence_length = {config["sequence_length"]}
overlap = {config["overlap"]}
batch_size = {config["batch_size"]}
w_decay = {config["w_decay"]}
lr = {config["lr"]}
tau_mem = {config["tau_mem"]}
loss_function = nn.MSELoss()
GRAD_CLIP = {config["grad_clip"]}
SCHEDULER_PATIENCE = {config["scheduler_patience"]}
PERSON_W = {config["person_weight"]}
CAR_W = {config["car_weight"]}
BUS_W = {config["bus_weight"]}
TRUCK_W = {config["truck_weight"]}
DROPOUT1 = {config["dropout1"]}
DROPOUT2 = {config["dropout2"]}
layer_nr = 500

class SNN(nn.Module):
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
        self.dropout1 = nn.Dropout(p=DROPOUT1)
        self.dropout2 = nn.Dropout(p=DROPOUT2)

    def forward(self, x, mem_states):
        bs, C, W, H = x.shape
        x = (x != 0).float()
        mem1,mem2,mem3,m51,m52,m61,m62,m71,m72,m81,m82 = mem_states
        v1 = self.bn1(self.conv1(x)); spk1, mem1 = self.lif1(v1, mem1)
        v2 = self.dropout1(self.bn2(self.conv2(self.maxpool(spk1)))); spk2, mem2 = self.lif2(v2, mem2)
        v3 = self.dropout1(self.bn3(self.conv3(spk2))); spk3, mem3 = self.lif3(v3, mem3)
        flat = spk3.view(bs, -1)
        v5 = self.dropout2(self.fcperson1(flat)); s51, m51 = self.lif4(v5, m51)
        v5 = self.dropout2(self.fcperson2(s51)); s52, m52 = self.lifperson(v5, m52)
        v6 = self.dropout2(self.fccar1(flat)); s61, m61 = self.lif5(v6, m61)
        v6 = self.dropout2(self.fccar2(s61)); s62, m62 = self.lifcar(v6, m62)
        v7 = self.dropout2(self.fcbus1(flat)); s71, m71 = self.lif6(v7, m71)
        v7 = self.dropout2(self.fcbus2(s71)); s72, m72 = self.lifbus(v7, m72)
        v8 = self.dropout2(self.fctruck1(flat)); s81, m81 = self.lif7(v8, m81)
        v8 = self.dropout2(self.fctruck2(s81)); s82, m82 = self.liftruck(v8, m82)
        return s52, s62, s72, s82, (mem1,mem2,mem3,m51,m52,m61,m62,m71,m72,m81,m82)

def loss_fn(output, target, step, cid):
    return loss_function(output, target * 1000)

data_dirs = ["week_32-box_3","week_33-box_2","week_34-box_1","week_35-box_2","week_36-box_3"]

def chunker(seq, size):
    return (seq[pos:pos+size] for pos in range(0, len(seq), size))

def pretty_time(s):
    if not s: return "0s"
    s = int(s)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return " ".join([f"{{c}}{{u}}" for c, u in [(h,"h"),(m,"m"),(s,"s")] if c])

training_data_dir = "{config["data_dir"]}"
output_dir = "{config["_output_dir"]}"
num_epochs = {config["epochs"]}

Path(output_dir).mkdir(parents=True, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SNN().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=w_decay, eps=1e-8)
scheduler = ReduceLROnPlateau(optimizer, "min", patience=SCHEDULER_PATIENCE, factor=0.5)
total_p = sum(p.numel() for p in model.parameters())
print(f"Model: SNN (FC heads) | Params: {{total_p:,}}")
print(f"LR: {{lr}} | Batch: {{batch_size}} | Dropout: {{DROPOUT1}}/{{DROPOUT2}}")
print(f"Class weights: person={{PERSON_W}} car={{CAR_W}} bus={{BUS_W}} truck={{TRUCK_W}}")
print("=" * 60)

data_files = []
for d in data_dirs:
    dp = os.path.join(training_data_dir, d)
    if os.path.exists(dp):
        data_files.extend([os.path.join(dp, f) for f in os.listdir(dp) if f.endswith(".pt")])
print(f"Found {{len(data_files)}} training files")
random.shuffle(data_files)

train_loss_list, val_loss_list, lr_list, val_acc_list = [], [], [], []
best_val = 999999

for epoch in range(num_epochs):
    total_tl, p_tl, c_tl, b_tl, t_tl, n_tb = 0,0,0,0,0,0
    model.train()
    start = time.time()
    for i, dp in enumerate(chunker(data_files, 4)):
        data = get_data(dp)
        if data is not None:
            for _, (frames, targets) in enumerate(data[0]):
                mem = tuple(None for _ in range(11))
                optimizer.zero_grad()
                frames, targets = frames.to(device), targets.to(device)
                pl, cl, bl, tl = 0,0,0,0
                for step in range(sequence_length):
                    o1,o2,o3,o4, mem = model(frames[:,step].unsqueeze(1), mem)
                    if step >= overlap:
                        pl += PERSON_W * loss_fn(o1.view(-1,64,64), targets[:,0,step], step, 0) / (sequence_length - overlap)
                        cl += CAR_W * loss_fn(o2.view(-1,64,64), targets[:,1,step], step, 2) / (sequence_length - overlap)
                        bl += BUS_W * loss_fn(o3.view(-1,64,64), targets[:,2,step], step, 5) / (sequence_length - overlap)
                        tl += TRUCK_W * loss_fn(o4.view(-1,64,64), targets[:,3,step], step, 7) / (sequence_length - overlap)
                loss = pl + cl + bl + tl
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                total_tl += loss.item(); p_tl += pl.item(); c_tl += cl.item(); b_tl += bl.item(); t_tl += tl.item()
                n_tb += 1

    total_vl, p_vl, c_vl, b_vl, t_vl, n_vb = 0,0,0,0,0,0
    model.eval()
    with torch.no_grad():
        for i, dp in enumerate(chunker(data_files, 4)):
            data = get_data(dp)
            if data is not None:
                for _, (frames, targets) in enumerate(data[1]):
                    mem = tuple(None for _ in range(11))
                    frames, targets = frames.to(device), targets.to(device)
                    pl, cl, bl, tl = 0,0,0,0
                    for step in range(sequence_length):
                        o1,o2,o3,o4, mem = model(frames[:,step].unsqueeze(1), mem)
                        pl += PERSON_W * loss_fn(o1.view(-1,64,64), targets[:,0,step], step, 0) / (sequence_length - overlap)
                        cl += CAR_W * loss_fn(o2.view(-1,64,64), targets[:,1,step], step, 2) / (sequence_length - overlap)
                        bl += BUS_W * loss_fn(o3.view(-1,64,64), targets[:,2,step], step, 5) / (sequence_length - overlap)
                        tl += TRUCK_W * loss_fn(o4.view(-1,64,64), targets[:,3,step], step, 7) / (sequence_length - overlap)
                    loss = pl + cl + bl + tl
                    total_vl += loss.item(); p_vl += pl.item(); c_vl += cl.item(); b_vl += bl.item(); t_vl += tl.item()
                    n_vb += 1

    scheduler.step(total_vl / n_vb)
    del data; gc.collect()
    et = time.time() - start
    vl = total_vl/n_vb
    print(f"Epoch {{epoch+1}} | train: {{total_tl/n_tb:.3f}} | val: {{vl:.3f}} | person {{p_vl/n_vb:.3f}} | car {{c_vl/n_vb:.3f}} | bus {{b_vl/n_vb:.3f}} | truck {{t_vl/n_vb:.3f}} | {{pretty_time(et)}}")

    train_loss_list.append([round(total_tl/n_tb,4), round(p_tl/n_tb,4), round(c_tl/n_tb,4), round(b_tl/n_tb,4), round(t_tl/n_tb,4)])
    val_loss_list.append([round(vl,4), round(p_vl/n_vb,4), round(c_vl/n_vb,4), round(b_vl/n_vb,4), round(t_vl/n_vb,4)])
    lr_list.append(scheduler.get_last_lr())
    val_acc_list.append(0)

    fn = os.path.join(output_dir, "multiclass-adamw")
    is_best = vl < best_val
    if is_best: best_val = vl
    save_d = {{"Epoch": epoch, "Tau": tau_mem, "w_decay": w_decay, "lr": lr, "model": "SNN_FC",
              "batch_size": batch_size, "dropout": [DROPOUT1, DROPOUT2],
              "class_weights": {{"person": PERSON_W, "car": CAR_W, "bus": BUS_W, "truck": TRUCK_W}},
              "train_loss": train_loss_list, "validation_loss": val_loss_list, "lr_schedule": lr_list,
              "validation_accuracy": val_acc_list}}
    with open(f"{{fn}}.json", "w") as f: json.dump(save_d, f)
    if is_best:
        torch.save(model.state_dict(), f"{{fn}}-{{epoch}}-{{vl:.4f}}.pth")
        print(f"  Best model saved: {{fn}}-{{epoch}}-{{vl:.4f}}.pth")

fn = os.path.join(output_dir, "multiclass-adamw")
torch.save(model.state_dict(), f"{{fn}}-{{epoch}}-{{vl:.4f}}.pth")
print(f"\\nTraining complete. Best val_loss: {{best_val:.4f}}")
'''
    with open(script_path, 'w') as f:
        f.write(script)


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

def run_experiments(gpu_id):
    """Run all experiments sequentially."""
    results = []
    total_start = time.time()

    print("=" * 70)
    print(f"EXPERIMENT RUNNER — {len(EXPERIMENTS)} experiments queued")
    print(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"GPU: {gpu_id}")
    print("=" * 70)

    for exp_idx, exp_config in enumerate(EXPERIMENTS):
        # Merge with defaults
        config = copy.deepcopy(DEFAULTS)
        config.update(exp_config)

        name = config["name"]
        config["_output_dir"] = os.path.join(config["output_base"], name)

        print(f"\n{'='*70}")
        print(f"EXPERIMENT {exp_idx + 1}/{len(EXPERIMENTS)}: {name}")
        print(f"{'='*70}")
        print(f"  Model:      {config['model']}")
        print(f"  Epochs:     {config['epochs']}")
        print(f"  LR:         {config['lr']}")
        if config['model'] == 'vit':
            print(f"  LR (pre):   {config['lr_pretrained']}")
            print(f"  ViT layers: {config['vit_layers']}")
            print(f"  Phase:      {config['phase']}")
        print(f"  Batch:      {config['batch_size']}")
        print(f"  Dropout:    {config['dropout1']}/{config['dropout2']}")
        print(f"  Output:     {config['_output_dir']}")

        # Generate script
        os.makedirs("_tmp_experiments", exist_ok=True)
        script_path = f"_tmp_experiments/exp_{name}.py"

        if config["model"] == "vit":
            generate_vit_script(config, script_path)
        else:
            generate_snn_script(config, script_path)

        # Create log file
        log_dir = config["_output_dir"]
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "training.log")

        # Run as subprocess
        exp_start = time.time()
        try:
            print(f"  Running... (log: {log_path})")            
            pbar = tqdm(total=config["epochs"], desc="Training", unit="epoch", leave=False, dynamic_ncols=True)

            with open(log_path, 'w') as log_file:
                # Use Popen to read output in real-time
                proc = subprocess.Popen(
                    [sys.executable, script_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1  # Line buffered
                )
                
                # Listen to the script's output as it runs
                for line in proc.stdout:
                    # 1. Write everything to the log file silently
                    log_file.write(line)
                    log_file.flush()
                    
                    # 2. Update the progress bar when an Epoch finishes
                    if line.startswith("Epoch "):
                        pbar.update(1)
                        # Bonus: Extract the validation loss and stick it on the progress bar!
                        try:
                            val_str = [p for p in line.split("|") if "val:" in p][0]
                            val_loss = val_str.split(":")[1].strip()
                            pbar.set_postfix(val_loss=val_loss)
                        except Exception:
                            pass
                
                proc.wait()
                pbar.close()

            exp_time = time.time() - exp_start
            success = proc.returncode == 0

            # Load results if available
            json_path = os.path.join(config["_output_dir"], "multiclass-adamw.json")
            best_val = None
            n_epochs = 0
            if os.path.exists(json_path):
                with open(json_path) as f:
                    jdata = json.load(f)
                    n_epochs = len(jdata.get("validation_loss", []))
                    if jdata.get("validation_loss"):
                        best_val = min(v[0] for v in jdata["validation_loss"])

            status = "OK" if success else "FAILED"
            results.append({
                "name": name,
                "status": status,
                "epochs": n_epochs,
                "best_val": best_val,
                "time": exp_time,
                "config": config,
            })

            if success:
                print(f"  DONE in {exp_time/60:.1f} min | {n_epochs} epochs | best_val={best_val:.2f}" if best_val else f"  DONE in {exp_time/60:.1f} min")
            else:
                print(f"  FAILED (return code {proc.returncode}) after {exp_time/60:.1f} min")
                print(f"  Check log: {log_path}")

        except Exception as e:
            exp_time = time.time() - exp_start
            print(f"  CRASHED: {e}")
            results.append({
                "name": name,
                "status": f"CRASH: {str(e)[:100]}",
                "epochs": 0,
                "best_val": None,
                "time": exp_time,
                "config": config,
            })

    # =========================================================================
    # SUMMARY
    # =========================================================================
    total_time = time.time() - total_start

    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"{'Name':<30s} | {'Status':>7s} | {'Epochs':>6s} | {'Best Val':>9s} | {'Time':>8s}")
    print(f"{'-'*30}-+-{'-'*7}-+-{'-'*6}-+-{'-'*9}-+-{'-'*8}")

    for r in results:
        val_str = f"{r['best_val']:.2f}" if r['best_val'] else "N/A"
        time_str = f"{r['time']/60:.1f}m"
        print(f"{r['name']:<30s} | {r['status']:>7s} | {r['epochs']:>6d} | {val_str:>9s} | {time_str:>8s}")

    print(f"\nTotal time: {total_time/3600:.1f} hours")
    print(f"Finished: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Save summary JSON
    summary_path = os.path.join(DEFAULTS["output_base"], "experiment_summary.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    summary = []
    for r in results:
        s = {k: v for k, v in r.items() if k != 'config'}
        s['lr'] = r['config']['lr']
        s['model'] = r['config']['model']
        s['batch_size'] = r['config']['batch_size']
        if r['config']['model'] == 'vit':
            s['vit_layers'] = r['config']['vit_layers']
            s['phase'] = r['config']['phase']
        summary.append(s)

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved: {summary_path}")


def show_summary():
    """Show results from previous experiment runs."""
    summary_path = os.path.join(DEFAULTS["output_base"], "experiment_summary.json")
    if not os.path.exists(summary_path):
        print("No experiment summary found. Run experiments first.")
        return

    with open(summary_path) as f:
        summary = json.load(f)

    print("=" * 70)
    print("PREVIOUS EXPERIMENT RESULTS")
    print("=" * 70)
    print(f"{'Name':<30s} | {'Status':>7s} | {'Epochs':>6s} | {'Best Val':>9s} | {'Time':>8s}")
    print(f"{'-'*30}-+-{'-'*7}-+-{'-'*6}-+-{'-'*9}-+-{'-'*8}")

    for r in summary:
        val_str = f"{r['best_val']:.2f}" if r.get('best_val') else "N/A"
        time_str = f"{r['time']/60:.1f}m" if r.get('time') else "N/A"
        print(f"{r['name']:<30s} | {r['status']:>7s} | {r.get('epochs', 0):>6d} | {val_str:>9s} | {time_str:>8s}")


# =============================================================================
# CLI
# =============================================================================

parser = argparse.ArgumentParser(
    description="Run queued training experiments",
    usage="%(prog)s [--gpu GPU_ID] [--summary_only]",
)
parser.add_argument("--gpu", type=int, default=0,
                    help="CUDA device ID (default: 0)")
parser.add_argument("--summary_only", action="store_true",
                    help="Just print results from previous runs")

args = parser.parse_args()

if args.summary_only:
    show_summary()
else:
    run_experiments(args.gpu)
