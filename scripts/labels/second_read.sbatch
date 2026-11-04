#!/bin/bash
# Second readers (labels v2, R4) on one 4-GPU node. Submit from Pitzer:
#   sbatch -M cardinal --export=NONE,READER=gpt-oss-120b,REQUESTS=pilot_val_v1 scripts/labels/second_read.sbatch
#   sbatch -M ascend   --export=NONE,READER=qwen3.8-27b,REQUESTS=pilot_val_v1 scripts/labels/second_read.sbatch
# --export=NONE matters: the submitting cluster's modules must not leak into the job.
#
#SBATCH --job-name=icpo-reader
#SBATCH --account=PAS2836
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --gpus-per-node=4
#SBATCH --output=/fs/scratch/PAS2836/mugezhang/icpo/logs/%x-%j.out

set -euo pipefail
READER=${READER:?set READER}; REQUESTS=${REQUESTS:?set REQUESTS}
REPO=/fs/ess/PAS2836/mugezhang/code/ChestICPO
# Start from a clean, explicit environment regardless of where the job was submitted from.
unset CC CXX FC F77 F90 CPATH LIBRARY_PATH LD_LIBRARY_PATH PYTHONPATH CUDA_HOME
export PATH=/usr/local/bin:/usr/bin:/bin CC=gcc CXX=g++
VENV=/fs/scratch/PAS2836/mugezhang/envs/icpo-vllm
PY=$VENV/bin/python
export PATH=$VENV/bin:$PATH   # vLLM/FlashInfer JIT builds call `ninja` from PATH
# keep every compile/kernel cache off the (full) home directory
C=/fs/scratch/PAS2836/mugezhang/cache
export VLLM_CACHE_ROOT=$C/vllm TORCHINDUCTOR_CACHE_DIR=$C/inductor TRITON_CACHE_DIR=$C/triton \
       FLASHINFER_WORKSPACE_BASE=$C/flashinfer XDG_CACHE_HOME=$C/xdg
cd "$REPO"
export HF_HOME=/fs/ess/PAS2836/mugezhang/hf_cache HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1
# Use vLLM's PyTorch sampler: FlashInfer's would be JIT-built with the node's own nvcc (other CUDA version)
export VLLM_USE_FLASHINFER_SAMPLER=0
# Batch-invariant mode is OFF by default: on A100 it made repeat runs differ more (2.1% vs 0.66% of items)
# and ran 35% slower (docs/exp1_micxr.md). Set BATCH_INVARIANT=1 only to re-test that.
export VLLM_BATCH_INVARIANT=${BATCH_INVARIANT:-0}
echo "host $(hostname)  commit $(git rev-parse HEAD)"
echo "gcc $(gcc -dumpversion)"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
$PY -c "import torch; print('torch', torch.__version__, 'cuda ok:', torch.cuda.is_available())"
$PY scripts/labels/second_read.py --reader "$READER" --requests "$REQUESTS" ${EXTRA:-}
