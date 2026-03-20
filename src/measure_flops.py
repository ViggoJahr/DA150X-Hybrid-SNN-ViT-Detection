import torch
import torch.nn as nn
import calflops

# Import the model architectures
from SNN_final_model import SNN
from SNN_ViT_model import SNNViT

# =====================================================================
# WRAPPER CLASSES
# Profiling tools like calflops expect standard inputs (e.g., images).
# Since our spiking models require a tuple of memory states (mem_states) 
# during the forward pass, we wrap the models to initialize and hide 
# these states internally. This allows the profiler to run smoothly.
# =====================================================================

class SNN_Profiler_Wrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = SNN()

    def forward(self, x):
        # Initialize 11 empty memory states for the SNN baseline
        mem_states = tuple([None] * 11)
        return self.model(x, mem_states)

class ViT_Profiler_Wrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = SNNViT()

    def forward(self, x):
        # Initialize 3 empty memory states for the ViT hybrid
        mem_states = (None, None, None)
        return self.model(x, mem_states)

if __name__ == "__main__":
    print("Calculating computational cost (FLOPs/MACs) and parameters...\n")

    # Define input shape matching the center-cropped event frames
    # (Batch size: 1, Channels: 1, Height: 200, Width: 200)
    input_shape = (1, 1, 200, 200)

    # -----------------------------------------------------------------
    # 1. Evaluate SNN Baseline
    # -----------------------------------------------------------------
    snn_wrapper = SNN_Profiler_Wrapper()
    print("--- SNN Baseline (Eliasson & Persson) ---")
    
    flops_snn, macs_snn, params_snn = calflops.calculate_flops(
        model=snn_wrapper, 
        input_shape=input_shape,
        output_as_string=True,
        output_precision=3,
        print_results=False  # Disabled default print to format it custom below
    )
    print(f"FLOPs: {flops_snn} | MACs: {macs_snn} | Params: {params_snn}\n")

    # -----------------------------------------------------------------
    # 2. Evaluate Hybrid SNN+ViT
    # -----------------------------------------------------------------
    vit_wrapper = ViT_Profiler_Wrapper()
    print("--- Hybrid SNN+ViT (Proposed Architecture) ---")
    
    flops_vit, macs_vit, params_vit = calflops.calculate_flops(
        model=vit_wrapper, 
        input_shape=input_shape,
        output_as_string=True,
        output_precision=3,
        print_results=False
    )
    print(f"FLOPs: {flops_vit} | MACs: {macs_vit} | Params: {params_vit}\n")