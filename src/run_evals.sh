#!/bin/bash
# run_final_evals.sh

DATA_DIR="../data/raw_scaled/"
EXP_DIR="../data/processed/experiments"
LOG_FILE="centroid_eval_debug.log"

echo "=== Starting Centroid Evaluation Sequence: $(date) ===" | tee -a $LOG_FILE

# 1. Baseline
echo "Processing Baseline..." | tee -a $LOG_FILE
CUDA_VISIBLE_DEVICES=2 python3 evaluate_model_centroid.py $DATA_DIR $EXP_DIR/snn-baseline-run2/4-27-9-35/ --baseline --gpu 0 2>&1 | tee -a $LOG_FILE

# 2. Hybrid v1
echo "Processing v1..." | tee -a $LOG_FILE
CUDA_VISIBLE_DEVICES=2 python3 evaluate_model_centroid.py $DATA_DIR $EXP_DIR/v1_phase2_clean/ --version v1 --gpu 0 2>&1 | tee -a $LOG_FILE

# 3. Hybrid v2
echo "Processing v2..." | tee -a $LOG_FILE
CUDA_VISIBLE_DEVICES=2 python3 evaluate_model_centroid.py $DATA_DIR $EXP_DIR/v2_phase2/ --version v2 --gpu 0 2>&1 | tee -a $LOG_FILE

# 4. Hybrid v3
echo "Processing v3..." | tee -a $LOG_FILE
CUDA_VISIBLE_DEVICES=2 python3 evaluate_model_centroid.py $DATA_DIR $EXP_DIR/v3_phase2_clean/ --version v3 --gpu 0 2>&1 | tee -a $LOG_FILE

echo "=== All evaluations complete: $(date) ===" | tee -a $LOG_FILE
