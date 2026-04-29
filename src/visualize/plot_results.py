import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Set academic plotting style
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 12, 
    'axes.labelsize': 14, 
    'axes.titlesize': 16,
    'figure.titlesize': 18
})

def parse_training_log(filepath):
    """
    Parses the training.log file to extract validation loss per epoch.
    Expected format: Epoch | Train | Val | ...
    """
    epochs = []
    val_losses = []
    
    if not os.path.exists(filepath):
        print(f"Warning: Log file not found at {filepath}")
        return epochs, val_losses

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.split('|')
            # Check if the line is a data row (starts with a digit)
            if len(parts) > 2 and parts[0].strip().isdigit():
                epochs.append(int(parts[0].strip()))
                val_losses.append(float(parts[2].strip()))
                
    return epochs, val_losses

def plot_convergence(experiments, output_dir):
    """
    Generates a line plot comparing Validation Loss across different models.
    Useful for demonstrating model stability and convergence speed.
    """
    plt.figure(figsize=(10, 6))
    
    # Professional color palette
    colors = ['#E63946', '#1D3557', '#457B9D', '#2A9D8F'] 
    
    for i, (label, folder) in enumerate(experiments.items()):
        log_path = os.path.join(folder, "training.log")
        epochs, val_losses = parse_training_log(log_path)
        
        if epochs:
            plt.plot(epochs, val_losses, label=label, linewidth=2.5, color=colors[i % len(colors)])

    plt.title("Model Convergence: Validation Loss Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.legend(title="Architecture", frameon=True)
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "convergence_comparison.png")
    plt.savefig(save_path, dpi=300)
    print(f"Successfully saved convergence plot to: {save_path}")
    plt.close()

def plot_class_recall(experiments, output_dir, threshold="0.05"):
    """
    Generates a grouped bar chart for Recall per class from evaluation JSON files.
    Demonstrates model performance on underrepresented classes (e.g., Bus, Truck).
    """
    data = []
    classes = ['person', 'car', 'bus', 'truck']
    
    for label, folder in experiments.items():
        json_path = os.path.join(folder, "evaluation", "eval_results.json")
        
        if not os.path.exists(json_path):
            print(f"Warning: Evaluation data missing for {label}")
            continue
            
        with open(json_path, 'r') as f:
            eval_data = json.load(f)
            
        # Extract metrics for the specified confidence threshold
        metrics = eval_data.get("detection_metrics", {}).get(threshold, {})
        
        for cls in classes:
            if cls in metrics:
                data.append({
                    "Model": label,
                    "Class": cls.capitalize(),
                    "Recall (%)": metrics[cls]["recall"] * 100
                })
                
    if not data:
        print("No evaluation data found to plot.")
        return
        
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(12, 7))
    sns.barplot(data=df, x="Class", y="Recall (%)", hue="Model", palette="viridis")
    
    plt.title(f"Class-specific Recall (Confidence Threshold: {threshold})")
    plt.xlabel("Object Category")
    plt.ylabel("Recall (True Positive Rate %)")
    plt.ylim(0, 105)
    plt.legend(title="Architecture", loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f"recall_per_class_th{threshold.replace('.', '')}.png")
    plt.savefig(save_path, dpi=300)
    print(f"Successfully saved recall plot to: {save_path}")
    plt.close()

if __name__ == "__main__":
    # --- CONFIGURATION ---
    # Define experiment directory paths relative to the project root
    base_dir = "data/processed/experiments"
    output_directory = "visualizations"
    
    # Mapping of Display Name : Folder Path
    # Using 'snn-baseline-run2' as the representative baseline as discussed
    experiment_paths = {
        "SNN Baseline": os.path.join(base_dir, "snn-baseline-run2", "4-27-9-35"),
        "v1 (Naïve Hybrid)": os.path.join(base_dir, "v1_phase2_clean"),
        "v2 (HsVT - FPN)": os.path.join(base_dir, "v2_phase2"),
        "v3 (Diet-ViT)": os.path.join(base_dir, "v3_phase2_clean")
    }
    
    print("Initializing report visualization generation...")
    plot_convergence(experiment_paths, output_directory)
    plot_class_recall(experiment_paths, output_directory, threshold="0.05")
    print("Process complete. Files are available in the 'visualizations' folder.")
