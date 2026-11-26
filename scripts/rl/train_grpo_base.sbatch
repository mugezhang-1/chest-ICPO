#!/bin/bash
#SBATCH --job-name=qwen35-grpo-base
#SBATCH --partition=debug-quad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=256G
#SBATCH --time=00:30:00
#SBATCH --account PAS3526
#SBATCH --output=logs/qwen35-grpo-base.out

set -euo pipefail

# NVIDIA-only environment: remove ROCm visibility variables
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

# ============================================================
# Chest-ICPO: Qwen3.5-9B + Fresh LoRA + Vanilla GRPO
# ============================================================

PROJECT_ROOT=/users/PAS2912/yusenpeng/chest-ICPO
VERL_ROOT=/fs/scratch/PAS3526/ICPO_5539/verl

# Environment
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=/fs/scratch/PAS3526/ICPO_5539/uv_cache
export HF_HOME=/fs/scratch/PAS3526/ICPO_5539/hf_cache
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1

export OMP_NUM_THREADS=4
export NGPUS=1

# Dataset
export TRAIN_FILE=/fs/scratch/PAS3526/ICPO_5539/RL_dataset/train_smoke.parquet
export VAL_FILE=/fs/scratch/PAS3526/ICPO_5539/RL_dataset/val_smoke.parquet

mkdir -p "$HF_HOME"

cd "$PROJECT_ROOT"

# GPU diagnostics
echo "===== SLURM JOB ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPUs:"
nvidia-smi

echo "===== GPU ENVIRONMENT ====="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-unset}"
echo "HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-unset}"

echo "===== STARTING GRPO ====="

# Reduce informational logs
export VLLM_LOGGING_LEVEL=ERROR
export TRANSFORMERS_VERBOSITY=error
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false

# Reduce Ray startup/logging noise
export RAY_COLOR_PREFIX=0
export RAY_DEDUP_LOGS=1

# Suppress Python UserWarning/FutureWarning messages
export PYTHONWARNINGS="ignore::FutureWarning,ignore::UserWarning"

bash scripts/rl/train_grpo_base.sh

echo "===== GRPO FINISHED ====="
