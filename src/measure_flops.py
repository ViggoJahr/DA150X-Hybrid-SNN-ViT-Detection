import torch
import torch.nn as nn
import calflops

# Import the model architectures (Updated to use the unified script)
from SNN_final_model import SNN
from SNN_ViT_model_all_versions import SNNViT

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

class ViT_Profiler_Wrapper_v1(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize v1: Linear Patch Embedding, 4 Layers
        self.model = SNNViT(version='v1')

    def forward(self, x):
        # Initialize 3 empty memory states for the ViT hybrid
        mem_states = (None, None, None)
        return self.model(x, mem_states)

class ViT_Profiler_Wrapper_v2(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize v2: MSPE, 4 Layers
        self.model = SNNViT(version='v2')

    def forward(self, x):
        mem_states = (None, None, None)
        return self.model(x, mem_states)

class ViT_Profiler_Wrapper_v3(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize v3: MSPE, 2 Layers
        self.model = SNNViT(version='v3')

    def forward(self, x):
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
    # 2. Evaluate Hybrid SNN+ViT v.1
    # -----------------------------------------------------------------
    vit_wrapper_v1 = ViT_Profiler_Wrapper_v1()
    print("--- Hybrid SNN+ViT (Proposed Architecture v1 - Linear Patch) ---")

    flops_vit1, macs_vit1, params_vit1 = calflops.calculate_flops(
        model=vit_wrapper_v1,
        input_shape=input_shape,
        output_as_string=True,
        output_precision=3,
        print_results=False
    )
    print(f"FLOPs: {flops_vit1} | MACs: {macs_vit1} | Params: {params_vit1}\n")

    # -----------------------------------------------------------------
    # 3. Evaluate Hybrid SNN+ViT v.2
    # -----------------------------------------------------------------
    vit_wrapper_v2 = ViT_Profiler_Wrapper_v2()
    print("--- Hybrid SNN+ViT (Proposed Architecture v2 - MSPE 4 Layers) ---")

    flops_vit2, macs_vit2, params_vit2 = calflops.calculate_flops(
        model=vit_wrapper_v2,
        input_shape=input_shape,
        output_as_string=True,
        output_precision=3,
        print_results=False
    )
    print(f"FLOPs: {flops_vit2} | MACs: {macs_vit2} | Params: {params_vit2}\n")

    # -----------------------------------------------------------------
    # 4. Evaluate Hybrid SNN+ViT v.3
    # -----------------------------------------------------------------
    vit_wrapper_v3 = ViT_Profiler_Wrapper_v3()
    print("--- Hybrid SNN+ViT (Proposed Architecture v3 - MSPE 2 Layers) ---")

    flops_vit3, macs_vit3, params_vit3 = calflops.calculate_flops(
        model=vit_wrapper_v3,
        input_shape=input_shape,
        output_as_string=True,
        output_precision=3,
        print_results=False
    )
    print(f"FLOPs: {flops_vit3} | MACs: {macs_vit3} | Params: {params_vit3}\n")
