#!/bin/bash
#SBATCH --job-name=ICPO_generate_reasoning_traces_w_answer_1000
#SBATCH --output=ICPO_generate_reasoning_traces_w_answer_1000.out
#SBATCH --account=PAS2836
#SBATCH --partition=nextgen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=15:00:00

module load miniconda3/24.1.2-py310
conda activate ICPO

export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CUDAHOSTCXX=/usr/bin/g++
export NVCC_CCBIN=/usr/bin/g++
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export VLLM_USE_FLASHINFER_SAMPLER=0


MODEL="qwen3.5-9b-think"
DATA_ROOT="/fs/scratch/PAS3526/ICPO_5539/chesticpo_v2_1024"

# 0 = full training dataset; 100 = first 100 QA bundles
LIMIT=1000

cd /users/PAS2912/yusenpeng/chest-ICPO/
    
# With ground-truth answer (2.0 - 2.2)
echo "Starting WITH-answer inference: $(date)"
python scripts/eval/run_new_dataset.py \
    --model "$MODEL" \
    --jsonl "$DATA_ROOT/train.jsonl" \
    --image-root "$DATA_ROOT/images" \
    --split train \
    --answer-mode with \
    --limit "$LIMIT" \
    --allow-dirty

echo "Finished WITH-answer inference: $(date)"
echo "All six variants generated successfully."

conda deactivate
