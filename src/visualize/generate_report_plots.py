#!/usr/bin/env python3
import os
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================
# Set style for academic reporting
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 16,            # General text
    'axes.titlesize': 20,       # Subplot titles
    'axes.labelsize': 18,       # X and Y axis labels
    'xtick.labelsize': 14,      # X axis tick numbers
    'ytick.labelsize': 14,      # Y axis tick numbers
    'legend.fontsize': 14,      # Legend text
    'figure.titlesize': 24,     # Main suptitle
    'figure.dpi': 300           # High resolution
})
OUTPUT_DIR = "visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define the paths to your standardized log files
EXPERIMENTS = {
    "../../data/processed/experiments/snn-baseline-run2/4-27-9-35/training.log": {
        "name": "Baseline",
        "color": "#7f8c8d",
        "style": "--"
    },
    "../../data/processed/experiments/v1_phase2_clean/training.log": {
        "name": "v1 (Linear, 4L)",
        "color": "#e74c3c",
        "style": "-"
    },
    "../../data/processed/experiments/v2_phase2/training.log": {
        "name": "v2 (MSPE, 4L)",
        "color": "#2980b9",
        "style": "-"
    },
    "../../data/processed/experiments/v3_phase2_clean/training.log": {
        "name": "v3 (MSPE, 2L)",
        "color": "#27ae60",
        "style": "-"
    }
}

# ==========================================
# DYNAMIC DATA EXTRACTION
# ==========================================
def parse_log_file(filepath):
    """Parses the standardized table format to extract train and val metrics."""
    train_curve = []
    val_curve = []
    
    if not os.path.exists(filepath):
        print(f"Warning: File not found -> {filepath}")
        return None

    with open(filepath, 'r') as f:
        for line in f:
            parts = [p.strip() for p in line.split('|')]
            # Check if it's a valid data row (starts with integer epoch)
            if len(parts) >= 7:
                try:
                    epoch = int(parts[0])
                    train_loss = float(parts[1]) # <-- Extracted Train Loss
                    val_loss = float(parts[2])
                    
                    train_curve.append(train_loss)
                    val_curve.append(val_loss)
                except ValueError:
                    # Skips headers and divider lines
                    continue
                    
    if not val_curve:
        return None
        
    best_loss = min(val_curve)
    
    return {
        "train_curve": train_curve,
        "val_curve": val_curve,
        "best_loss": best_loss
    }

# Build the dynamic data dictionary
parsed_data = {}
for path, meta in EXPERIMENTS.items():
    extracted = parse_log_file(path)
    if extracted:
        parsed_data[meta["name"]] = {**meta, **extracted}

if not parsed_data:
    print("Error: No data could be parsed. Check your file paths.")
    exit(1)

# ==========================================
# PLOT 1: All Validation Curves (Combined)
# ==========================================
plt.figure(figsize=(12, 8))
for name, info in parsed_data.items():
    epochs = range(1, len(info["val_curve"]) + 1)
    plt.plot(epochs, info["val_curve"], 
             label=name, color=info["color"], linestyle=info["style"], 
             linewidth=3 if name != "Baseline" else 2.5)

if "Baseline" in parsed_data:
    baseline_best = parsed_data["Baseline"]["best_loss"]
    plt.axhline(y=baseline_best, color=parsed_data["Baseline"]["color"], 
                linestyle=':', alpha=1, label=f"Baseline Best ({baseline_best:.1f})")

plt.title("Validation Loss Convergence", fontweight='bold', pad=10)
plt.xlabel("Epoch", labelpad=7)
plt.ylabel("MSE Loss", labelpad=7)
plt.legend(frameon=True, shadow=True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "1_convergence_all_val.png"))
plt.close()
print(f"Generated -> 1_convergence_all_val.png")

# ==========================================
# PLOT 2: Train vs Validation (2x2 Grid)
# ==========================================
# Create a 2x2 grid of subplots
fig, axs = plt.subplots(2, 2, figsize=(16, 12))
axs = axs.flatten() # Flatten to loop over them easily

for i, (name, info) in enumerate(parsed_data.items()):
    ax = axs[i]
    epochs = range(1, len(info["train_curve"]) + 1)
    
    # Plot Training Loss
    ax.plot(epochs, info["train_curve"], 
            label='Train Loss', color='gray', linestyle='--', linewidth=2.5)
    # Plot Validation Loss
    ax.plot(epochs, info["val_curve"], 
            label='Val Loss', color=info["color"], linewidth=3)
    
    # Formatting the subplot
    ax.set_title(f"{name} (Best Val: {info['best_loss']:.1f})", fontweight='bold', pad=10)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.legend(loc='upper right', frameon=True)
    ax.grid(True, linestyle=':', alpha=1)

plt.tight_layout()
# Adjust top to make room for suptitle
plt.subplots_adjust(top=0.92) 
plt.savefig(os.path.join(OUTPUT_DIR, "2_train_vs_val_grid.png"))
plt.close()
print(f"Generated -> 2_train_vs_val_grid.png")

print(f"\n✅ Success! Plots saved to {os.path.abspath(OUTPUT_DIR)}")
