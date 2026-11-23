#!/bin/bash
#SBATCH --job-name=ICPO_EVAL_test
#SBATCH --output=ICPO_EVAL_test.out
#SBATCH --account=PAS3526
#SBATCH --partition=nextgen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=20:00:00

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

SPLIT="test"

cd /users/PAS2912/yusenpeng/chest-ICPO/

echo "Starting $SPLIT inference: $(date)"
python scripts/eval/run_new_dataset.py \
    --model "$MODEL" \
    --jsonl "$DATA_ROOT/$SPLIT.jsonl" \
    --image-root "$DATA_ROOT/images" \
    --split "$SPLIT" \
    --answer-mode without \
    --allow-dirty
echo "Finished $SPLIT inference: $(date)"
conda deactivate
