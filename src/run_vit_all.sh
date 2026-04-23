#!/bin/bash
# Run all 4 ViT experiments sequentially on GPU 3
# Launch in tmux: tmux new -s vit
# CUDA_VISIBLE_DEVICES=3 bash run_vit_all.sh

set -e
cd /raid/home/da150x143/DA150X-Hybrid-SNN-ViT-Detection/src

SNN_BACKBONE="../data/processed/experiments/v1/snn_baseline_weights/multiclass-adamw-39-91.4787.pth"

echo "============================================"
echo "STEP 1/4: v1 Phase 1 (frozen, 20 epochs)"
echo "============================================"
python3 SNN_ViT_model.py \
    ../data/raw_scaled/ ../data/processed/experiments/v1_phase1/ \
    --gpu 0 --epoch 20 --phase 1 \
    --snn_backbone "$SNN_BACKBONE"

# Find best checkpoint from phase 1
V1_P1_BEST=$(ls -t ../data/processed/experiments/v1_phase1/vit-*/multiclass-adamw-*.pth 2>/dev/null | head -1)
echo "v1 Phase 1 best: $V1_P1_BEST"

echo "============================================"
echo "STEP 2/4: v1 Phase 2 (unfrozen, 40 epochs)"
echo "============================================"
python3 SNN_ViT_model.py \
    ../data/raw_scaled/ ../data/processed/experiments/v1_phase2/ \
    --gpu 0 --epoch 40 --phase 2 \
    --checkpoint "$V1_P1_BEST"

echo "============================================"
echo "STEP 3/4: v2.1 Phase 1 (frozen, 20 epochs)"
echo "============================================"
python3 SNN_ViT_model_v2.py --version v2.1 \
    ../data/raw_scaled/ ../data/processed/experiments/v2_phase1/ \
    --gpu 0 --epoch 20 --phase 1 \
    --snn_backbone "$SNN_BACKBONE"

# Find best checkpoint from v2 phase 1
V2_P1_BEST=$(ls -t ../data/processed/experiments/v2_phase1/vit-*/multiclass-adamw-*.pth 2>/dev/null | head -1)
echo "v2.1 Phase 1 best: $V2_P1_BEST"

echo "============================================"
echo "STEP 4/4: v2.1 Phase 2 (unfrozen, 40 epochs)"
echo "============================================"
python3 SNN_ViT_model_v2.py --version v2.1 \
    ../data/raw_scaled/ ../data/processed/experiments/v2_phase2/ \
    --gpu 0 --epoch 40 --phase 2 \
    --checkpoint "$V2_P1_BEST"

echo "============================================"
echo "ALL DONE!"
echo "============================================"
