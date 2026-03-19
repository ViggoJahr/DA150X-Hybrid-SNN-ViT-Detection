"""
Combined Training Curve Plotter: SNN Baseline vs SNN+ViT
DA150X - KTH Royal Institute of Technology
Authors: Axel Prander & Viggo Jahr

Generates publication-quality comparison figures for the thesis.
Loads training stats from JSON files produced by both SNN_final_model.py
and SNN_ViT_model.py.

Usage:
  # Auto-detect from both repos:
  python3 plot_snn_vs_vit.py

  # Or specify paths manually:
  python3 plot_snn_vs_vit.py \
      --snn_json path/to/snn/multiclass-adamw.json \
      --vit_p1_json path/to/vit_phase1/multiclass-adamw.json \
      --vit_p2_json path/to/vit_phase2/multiclass-adamw.json \
      --output_dir figures/

Output: PNG figures saved to output directory.
"""

import argparse
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server

# =============================================================================
# STYLE CONFIGURATION (matching plot_training.py thesis style)
# =============================================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.3,
})

CLASS_NAMES = ['person', 'car', 'bus', 'truck']
CLASS_COLORS = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']  # red, blue, green, orange


def load_json(path):
    """Load training stats JSON file."""
    if not os.path.exists(path):
        print(f"WARNING: File not found: {path}")
        return None
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def extract_losses(data):
    """Extract val and train loss arrays from JSON data.

    Returns:
        epochs: list of epoch numbers (1-indexed)
        val_total: list of total val loss per epoch
        val_classes: dict of class_name -> list of val loss per epoch
        train_total: list of total train loss per epoch
        train_classes: dict of class_name -> list of train loss per epoch
    """
    val_loss = data.get('validation_loss', [])
    train_loss = data.get('train_loss', [])

    epochs = list(range(1, len(val_loss) + 1))

    val_total = [v[0] for v in val_loss]
    train_total = [t[0] for t in train_loss]

    val_classes = {}
    train_classes = {}
    for i, name in enumerate(CLASS_NAMES):
        val_classes[name] = [v[i + 1] for v in val_loss]
        train_classes[name] = [t[i + 1] for t in train_loss]

    return epochs, val_total, val_classes, train_total, train_classes


def find_json_files():
    """Try to auto-discover JSON files from known locations."""
    search_paths = {
        'snn_run2': [
            'data/model_output/scaled/3-11-15-30/multiclass-adamw.json',
            '../DA150X-Hybrid-SNN-ViT-Event-Detection/data/model_output/scaled/3-11-15-30/multiclass-adamw.json',
        ],
        'snn_run3': [
            'data/model_output/scaled/3-11-16-5/multiclass-adamw.json',
            '../DA150X-Hybrid-SNN-ViT-Event-Detection/data/model_output/scaled/3-11-16-5/multiclass-adamw.json',
        ],
        'vit_phase1': [
            'src/data/processed/vit/vit-3-13-18-24/multiclass-adamw.json',
            'data/processed/vit/vit-3-13-18-24/multiclass-adamw.json',
        ],
        'vit_phase2': [
            'src/data/processed/vit/vit-3-13-18-48/multiclass-adamw.json',
            'data/processed/vit/vit-3-13-18-48/multiclass-adamw.json',
        ],
    }

    found = {}
    for name, paths in search_paths.items():
        for p in paths:
            if os.path.exists(p):
                found[name] = p
                print(f"  Found {name}: {p}")
                break

    return found


# =============================================================================
# FIGURE GENERATORS
# =============================================================================

def fig1_total_val_loss_comparison(runs, output_dir):
    """
    Figure 1: Total validation loss across all runs on one plot.
    Shows SNN baseline(s) vs ViT phases.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {
        'snn_run2': '#7f8c8d',
        'snn_run3': '#2c3e50',
        'vit_phase1': '#e67e22',
        'vit_phase2': '#e74c3c',
    }
    labels = {
        'snn_run2': 'SNN Baseline (Run 2, lr=1e-4)',
        'snn_run3': 'SNN Baseline (Run 3, lr=5e-4)',
        'vit_phase1': 'SNN+ViT Phase 1 (frozen)',
        'vit_phase2': 'SNN+ViT Phase 2 (fine-tune)',
    }
    linestyles = {
        'snn_run2': '--',
        'snn_run3': '-.',
        'vit_phase1': ':',
        'vit_phase2': '-',
    }

    for name, data in runs.items():
        epochs, val_total, _, _, _ = extract_losses(data)
        ax.plot(epochs, val_total, label=labels.get(name, name),
                color=colors.get(name, 'black'),
                linestyle=linestyles.get(name, '-'),
                linewidth=2)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Total Validation Loss: SNN Baseline vs SNN+ViT')
    ax.legend(loc='upper right')

    path = os.path.join(output_dir, 'fig_comparison_total_val_loss.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def fig2_per_class_comparison(runs, output_dir):
    """
    Figure 2: Per-class validation loss comparison.
    2x2 grid, one subplot per class.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    colors = {
        'snn_run2': '#7f8c8d',
        'snn_run3': '#2c3e50',
        'vit_phase1': '#e67e22',
        'vit_phase2': '#e74c3c',
    }
    labels = {
        'snn_run2': 'SNN (Run 2)',
        'snn_run3': 'SNN (Run 3)',
        'vit_phase1': 'ViT P1',
        'vit_phase2': 'ViT P2',
    }

    for c, class_name in enumerate(CLASS_NAMES):
        ax = axes[c]
        for name, data in runs.items():
            epochs, _, val_classes, _, _ = extract_losses(data)
            ax.plot(epochs, val_classes[class_name],
                    label=labels.get(name, name),
                    color=colors.get(name, 'black'),
                    linewidth=1.5)

        ax.set_title(f'{class_name.capitalize()} Validation Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend(fontsize=8)

    plt.suptitle('Per-Class Validation Loss: SNN vs ViT', fontsize=14, fontweight='bold')
    plt.tight_layout()

    path = os.path.join(output_dir, 'fig_comparison_per_class.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def fig3_train_vs_val(runs, output_dir):
    """
    Figure 3: Train vs validation loss for each model.
    Shows overfitting behavior comparison.
    """
    # Only plot runs with enough data
    plot_runs = {k: v for k, v in runs.items() if len(v.get('validation_loss', [])) > 5}

    n = len(plot_runs)
    if n == 0:
        print("  Skipping fig3 (not enough data)")
        return

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    titles = {
        'snn_run2': 'SNN Run 2 (lr=1e-4)',
        'snn_run3': 'SNN Run 3 (lr=5e-4)',
        'vit_phase1': 'ViT Phase 1 (frozen)',
        'vit_phase2': 'ViT Phase 2 (fine-tune)',
    }

    for i, (name, data) in enumerate(plot_runs.items()):
        ax = axes[i]
        epochs, val_total, _, train_total, _ = extract_losses(data)

        ax.plot(epochs, train_total, label='Train', color='#3498db', linewidth=1.5)
        ax.plot(epochs, val_total, label='Val', color='#e74c3c', linewidth=1.5)

        # Shade the gap
        ax.fill_between(epochs, train_total, val_total, alpha=0.1, color='red')

        ax.set_title(titles.get(name, name))
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()

        # Annotate gap at last epoch
        gap = val_total[-1] - train_total[-1]
        ax.annotate(f'Gap: {gap:.1f}',
                     xy=(epochs[-1], (val_total[-1] + train_total[-1]) / 2),
                     fontsize=9, ha='right')

    plt.suptitle('Overfitting Analysis: Train vs Validation', fontsize=14, fontweight='bold')
    plt.tight_layout()

    path = os.path.join(output_dir, 'fig_comparison_overfitting.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def fig4_final_epoch_bar_chart(runs, output_dir):
    """
    Figure 4: Grouped bar chart of per-class loss at final epoch.
    Direct comparison of where each model ended up.
    """
    run_names = list(runs.keys())
    labels = {
        'snn_run2': 'SNN (Run 2)',
        'snn_run3': 'SNN (Run 3)',
        'vit_phase1': 'ViT P1',
        'vit_phase2': 'ViT P2',
    }

    # Extract final-epoch per-class losses
    final_losses = {}
    for name, data in runs.items():
        _, _, val_classes, _, _ = extract_losses(data)
        final_losses[name] = [val_classes[c][-1] for c in CLASS_NAMES]

    x = np.arange(len(CLASS_NAMES))
    width = 0.8 / len(run_names)

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, name in enumerate(run_names):
        offset = (i - len(run_names) / 2 + 0.5) * width
        bars = ax.bar(x + offset, final_losses[name], width,
                      label=labels.get(name, name),
                      color=CLASS_COLORS[i % len(CLASS_COLORS)],
                      alpha=0.8)

        # Add value labels on bars
        for bar, val in zip(bars, final_losses[name]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Class')
    ax.set_ylabel('Validation Loss (final epoch)')
    ax.set_title('Per-Class Loss at Final Epoch')
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in CLASS_NAMES])
    ax.legend()

    path = os.path.join(output_dir, 'fig_comparison_final_loss_bar.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def fig5_parameter_efficiency(runs, output_dir):
    """
    Figure 5: Scatter plot — best val loss vs number of parameters.
    Shows that ViT achieves comparable/better loss with far fewer params.
    """
    # Hardcoded param counts (from architecture)
    param_counts = {
        'snn_run2': 15_598_640,
        'snn_run3': 15_598_640,
        'vit_phase1': 117_188,      # trainable only
        'vit_phase2': 1_897_028,
    }
    labels = {
        'snn_run2': 'SNN Run 2',
        'snn_run3': 'SNN Run 3',
        'vit_phase1': 'ViT P1\n(117K trainable)',
        'vit_phase2': 'ViT P2\n(1.9M total)',
    }
    colors = {
        'snn_run2': '#7f8c8d',
        'snn_run3': '#2c3e50',
        'vit_phase1': '#e67e22',
        'vit_phase2': '#e74c3c',
    }

    fig, ax = plt.subplots(figsize=(8, 6))

    for name, data in runs.items():
        _, val_total, _, _, _ = extract_losses(data)
        best_val = min(val_total)
        params = param_counts.get(name, 0)

        ax.scatter(params / 1e6, best_val, s=150, color=colors.get(name, 'black'),
                   zorder=5, edgecolor='white', linewidth=1.5)
        ax.annotate(labels.get(name, name),
                     xy=(params / 1e6, best_val),
                     xytext=(10, 10), textcoords='offset points',
                     fontsize=9,
                     arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

    ax.set_xlabel('Parameters (millions)')
    ax.set_ylabel('Best Validation Loss')
    ax.set_title('Parameter Efficiency: Fewer Params, Better Loss?')
    ax.set_xscale('log')

    path = os.path.join(output_dir, 'fig_comparison_param_efficiency.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def fig6_loss_composition_comparison(runs, output_dir):
    """
    Figure 6: Pie charts showing loss composition (% per class) at final epoch.
    One pie per model.
    """
    plot_runs = {k: v for k, v in runs.items()}
    n = len(plot_runs)
    if n == 0:
        return

    titles = {
        'snn_run2': 'SNN Run 2',
        'snn_run3': 'SNN Run 3',
        'vit_phase1': 'ViT Phase 1',
        'vit_phase2': 'ViT Phase 2',
    }

    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4))
    if n == 1:
        axes = [axes]

    for i, (name, data) in enumerate(plot_runs.items()):
        ax = axes[i]
        _, _, val_classes, _, _ = extract_losses(data)
        final = [val_classes[c][-1] for c in CLASS_NAMES]
        total = sum(final)
        pcts = [v / total * 100 for v in final]

        wedges, texts, autotexts = ax.pie(
            pcts,
            labels=[c.capitalize() for c in CLASS_NAMES],
            autopct='%1.0f%%',
            colors=CLASS_COLORS,
            startangle=90,
        )
        ax.set_title(titles.get(name, name), fontsize=11, fontweight='bold')

    plt.suptitle('Loss Composition by Class (Final Epoch)', fontsize=14, fontweight='bold')
    plt.tight_layout()

    path = os.path.join(output_dir, 'fig_comparison_loss_composition.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Plot SNN vs ViT training curves for thesis",
    )
    parser.add_argument("--snn_json", type=str, default=None,
                        help="Path to SNN baseline JSON (best run)")
    parser.add_argument("--snn_run3_json", type=str, default=None,
                        help="Path to SNN Run 3 JSON")
    parser.add_argument("--vit_p1_json", type=str, default=None,
                        help="Path to ViT Phase 1 JSON")
    parser.add_argument("--vit_p2_json", type=str, default=None,
                        help="Path to ViT Phase 2 JSON")
    parser.add_argument("--output_dir", type=str, default="figures/",
                        help="Directory to save figures (default: figures/)")

    args = parser.parse_args()

    # Auto-discover if not specified
    print("Searching for training data...")
    found = find_json_files()

    # Override with CLI args
    if args.snn_json:
        found['snn_run2'] = args.snn_json
    if args.snn_run3_json:
        found['snn_run3'] = args.snn_run3_json
    if args.vit_p1_json:
        found['vit_phase1'] = args.vit_p1_json
    if args.vit_p2_json:
        found['vit_phase2'] = args.vit_p2_json

    if not found:
        print("ERROR: No JSON files found. Specify paths with --snn_json etc.")
        sys.exit(1)

    # Load data
    runs = {}
    for name, path in found.items():
        data = load_json(path)
        if data and data.get('validation_loss'):
            runs[name] = data
            epochs = len(data['validation_loss'])
            best = min(v[0] for v in data['validation_loss'])
            print(f"  Loaded {name}: {epochs} epochs, best val_loss={best:.2f}")

    if not runs:
        print("ERROR: No valid training data loaded.")
        sys.exit(1)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate figures
    print(f"\nGenerating figures in {args.output_dir}/...")
    fig1_total_val_loss_comparison(runs, args.output_dir)
    fig2_per_class_comparison(runs, args.output_dir)
    fig3_train_vs_val(runs, args.output_dir)
    fig4_final_epoch_bar_chart(runs, args.output_dir)
    fig5_parameter_efficiency(runs, args.output_dir)
    fig6_loss_composition_comparison(runs, args.output_dir)

    print(f"\nDone! {len(os.listdir(args.output_dir))} figures saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
