#!/bin/bash
#SBATCH --job-name=ICPO_EVAL_test_GRPO_120steps
#SBATCH --output=ICPO_EVAL_test_GRPO_120steps.out
#SBATCH --account=PAS3526
#SBATCH --partition=nextgen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00

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

############ ORIGINAL HuggingFace MODEL ############
# python scripts/eval/run_new_dataset.py \
#     --model "$MODEL" \
#     --jsonl "$DATA_ROOT/$SPLIT.jsonl" \
#     --image-root "$DATA_ROOT/images" \
#     --split "$SPLIT" \
#     --answer-mode without \
#     --allow-dirty

############ RL model with LoRA ############

# step 1: extract LoRA adapter from the checkpoint (DO IT ONCE per checkpoint)
# VERL_ROOT=/fs/scratch/PAS3526/ICPO_5539/verl
# CKPT=/fs/scratch/PAS3526/ICPO_5539/grpo_outputs/base_lora_FULL/global_step_119
# cd "$VERL_ROOT"
# .venv/bin/python -m verl.model_merger merge \
#     --backend fsdp \
#     --local_dir "$CKPT/actor" \
#     --target_dir "$CKPT/exported"

# step 2: run evaluation with the extracted LoRA adapter
cd /users/PAS2912/yusenpeng/chest-ICPO/
python scripts/eval/run_new_dataset.py \
    --model "$MODEL" \
    --jsonl "$DATA_ROOT/$SPLIT.jsonl" \
    --image-root "$DATA_ROOT/images" \
    --split "$SPLIT" \
    --answer-mode without \
    --allow-dirty \
    --lora-path /fs/scratch/PAS3526/ICPO_5539/grpo_outputs/base_lora_FULL/global_step_119/exported/lora_adapter \
    --tag grpo_step120

echo "Finished $SPLIT inference: $(date)"
conda deactivate
