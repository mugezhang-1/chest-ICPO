
#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Chest-ICPO: Vanilla GRPO with Fresh LoRA
#
# Base:     Qwen/Qwen3.5-9B
# Training: GRPO + fresh LoRA (no SFT checkpoint)
# Backend:  veRL FSDP2 + vLLM
# Reward:   Exact-match answer accuracy
#
# DRY_RUN=1 bash scripts/rl/train_grpo_base.sh
# ============================================================

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

VERL_ROOT=/fs/scratch/PAS3526/ICPO_5539/verl
PROJECT_ROOT=/users/PAS2912/yusenpeng/chest-ICPO
DATA_ROOT=/fs/scratch/PAS3526/ICPO_5539/RL_dataset

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-9B}"

TRAIN_FILE="${TRAIN_FILE:-$DATA_ROOT/train_smoke.parquet}"
VAL_FILE="${VAL_FILE:-$DATA_ROOT/val_smoke.parquet}"

# OUTPUT_DIR=/fs/scratch/PAS3526/ICPO_5539/grpo_outputs/base_lora_SMOKE
OUTPUT_DIR=/fs/scratch/PAS3526/ICPO_5539/grpo_outputs/base_lora_FULL

PYTHON="$VERL_ROOT/.venv/bin/python"

# ------------------------------------------------------------
# Training configuration
# ------------------------------------------------------------

NGPUS="${NGPUS:-1}"

TRAIN_BATCH_SIZE=4
PPO_MINI_BATCH_SIZE=4
PPO_MICRO_BATCH_SIZE_PER_GPU=1

ROLLOUT_N=4
ROLLOUT_TP=1
ROLLOUT_GPU_MEM_UTIL=0.35

MAX_PROMPT_LENGTH=8192
MAX_RESPONSE_LENGTH=2048
MAX_MODEL_LEN=$(($MAX_PROMPT_LENGTH + $MAX_RESPONSE_LENGTH))

LORA_RANK=8
LORA_ALPHA=16

ACTOR_LR=1e-6

TOTAL_EPOCHS=1
SAVE_FREQ=50
TEST_FREQ=50

# ------------------------------------------------------------
# Environment
# ------------------------------------------------------------

export UV_CACHE_DIR=/fs/scratch/PAS3526/ICPO_5539/uv_cache
export HF_HOME=/fs/scratch/PAS3526/ICPO_5539/hf_cache

export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "$OUTPUT_DIR" "$HF_HOME"

# ------------------------------------------------------------
# Validate paths
# ------------------------------------------------------------

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: veRL Python not found: $PYTHON"
    exit 1
fi

for f in \
    "$TRAIN_FILE" \
    "$VAL_FILE" \
    "$PROJECT_ROOT/scripts/rl/reward.py"; do

    if [[ ! -r "$f" ]]; then
        echo "ERROR: Missing or unreadable file: $f"
        exit 1
    fi
done

# ------------------------------------------------------------
# LoRA configuration
#
# Match LANGUAGE MODEL projections only.
# No LoRA on vision encoder or aligner.
#
# [.] avoids regex backslash escaping problems.
# Embedded double quotes are preserved for Hydra.
# ------------------------------------------------------------

LORA_TARGET='actor_rollout_ref.model.target_modules="^model[.]language_model[.].*[.](v_proj|gate_proj|o_proj|out_proj|up_proj|in_proj_b|k_proj|q_proj|in_proj_qkv|in_proj_z|in_proj_a|down_proj)$"'

# ------------------------------------------------------------
# veRL Hydra overrides
# ------------------------------------------------------------

ARGS=(
    # Algorithm
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=false

    # Dataset
    "data.train_files=$TRAIN_FILE"
    "data.val_files=$VAL_FILE"
    data.image_key=images
    "data.train_batch_size=$TRAIN_BATCH_SIZE"
    "data.max_prompt_length=$MAX_PROMPT_LENGTH"
    "data.max_response_length=$MAX_RESPONSE_LENGTH"
    data.filter_overlong_prompts=true
    data.truncation=error

    # Model: original Hugging Face model
    "actor_rollout_ref.model.path=$BASE_MODEL"
    "actor_rollout_ref.model.lora_rank=$LORA_RANK"
    "actor_rollout_ref.model.lora_alpha=$LORA_ALPHA"
    actor_rollout_ref.model.lora_adapter_path=null

    # LoRA targets
    "$LORA_TARGET"
    actor_rollout_ref.model.exclude_modules=null

    # Memory optimizations
    actor_rollout_ref.model.enable_gradient_checkpointing=true
    actor_rollout_ref.model.use_remove_padding=true

    # Actor
    actor_rollout_ref.actor.strategy=fsdp2
    "actor_rollout_ref.actor.optim.lr=$ACTOR_LR"
    "actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE"
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU"
    actor_rollout_ref.actor.use_kl_loss=false
    actor_rollout_ref.actor.fsdp_config.param_offload=true
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true

    # vLLM rollout
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.load_format=dummy
    "actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP"
    "actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEM_UTIL"
    "actor_rollout_ref.rollout.n=$ROLLOUT_N"
    '+actor_rollout_ref.rollout.engine_kwargs.vllm.reasoning_config={reasoning_start_str:"<think>",reasoning_end_str:"</think>"}'
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.enable_chunked_prefill=true
    actor_rollout_ref.rollout.layered_summon=false
    actor_rollout_ref.rollout.enforce_eager=true

    # Log-probability microbatches
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1


    # Reward
    "reward.custom_reward_function.path=$PROJECT_ROOT/scripts/rl/reward.py"
    reward.custom_reward_function.name=compute_score

    # Trainer
    trainer.project_name=chest_icpo
    trainer.experiment_name=qwen35_9b_base_grpo_lora
    'trainer.logger=["console"]'
    "trainer.n_gpus_per_node=$NGPUS"
    trainer.nnodes=1
    "trainer.total_epochs=$TOTAL_EPOCHS"
    "trainer.default_local_dir=$OUTPUT_DIR"
    "trainer.save_freq=$SAVE_FREQ"
    "trainer.test_freq=$TEST_FREQ"
    "trainer.rollout_data_dir=$OUTPUT_DIR/rollouts"
    "trainer.validation_data_dir=$OUTPUT_DIR/validation"
    trainer.resume_mode=disable
    trainer.val_before_train=False
)

# Allow additional Hydra overrides from command line
ARGS+=("$@")

# ------------------------------------------------------------
# Diagnostics
# ------------------------------------------------------------

echo "=========================================="
echo "Chest-ICPO GRPO"
echo "=========================================="
echo "Model:          $BASE_MODEL"
echo "Initialization: Fresh LoRA"
echo "LoRA rank:      $LORA_RANK"
echo "LoRA alpha:     $LORA_ALPHA"
echo "GPUs:           $NGPUS"
echo "Train dataset:  $TRAIN_FILE"
echo "Val dataset:    $VAL_FILE"
echo "Output:         $OUTPUT_DIR"
echo "=========================================="

cd "$VERL_ROOT"

# ------------------------------------------------------------
# Validate Hydra configuration without GPU
# ------------------------------------------------------------

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "Running Hydra configuration check..."

    "$PYTHON" -m verl.trainer.main_ppo \
        "${ARGS[@]}" \
        --cfg job

    echo "Hydra configuration check completed."
    exit 0
fi

# ------------------------------------------------------------
# Validate GPU allocation
# ------------------------------------------------------------

VISIBLE_GPUS=$("$PYTHON" - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)

if [[ "$VISIBLE_GPUS" -lt "$NGPUS" ]]; then
    echo "ERROR: Requested $NGPUS GPUs, but only $VISIBLE_GPUS visible."
    exit 1
fi

# ------------------------------------------------------------
# Launch GRPO
# ------------------------------------------------------------

echo "Starting veRL GRPO..."

"$PYTHON" -m verl.trainer.main_ppo \
    "${ARGS[@]}"
