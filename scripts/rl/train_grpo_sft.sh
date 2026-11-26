#!/usr/bin/env bash
set -euo pipefail

VERL_ROOT=/fs/scratch/PAS3526/ICPO_5539/verl
PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
DATA_ROOT=/fs/scratch/PAS3526/ICPO_5539/RL_dataset

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-9B}"
SFT_ADAPTER="${SFT_ADAPTER:-}"

TRAIN_FILE="${TRAIN_FILE:-$DATA_ROOT/train_smoke.parquet}"
VAL_FILE="${VAL_FILE:-$DATA_ROOT/val.parquet}"

NGPUS="${NGPUS:-4}"
OUTPUT_DIR=/fs/scratch/PAS3526/ICPO_5539/grpo_outputs

export UV_CACHE_DIR=/fs/scratch/PAS3526/ICPO_5539/uv_cache
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false

# Don't silently train from a fresh adapter.
if [[ -z "$SFT_ADAPTER" ]]; then
    echo "ERROR: Set SFT_ADAPTER to the accessible SFT checkpoint."
    exit 1
fi

for f in "$TRAIN_FILE" "$VAL_FILE" \
         "$PROJECT_ROOT/scripts/rl/reward.py" \
         "$SFT_ADAPTER/adapter_config.json" \
         "$SFT_ADAPTER/adapter_model.safetensors"; do
    if [[ ! -r "$f" ]]; then
        echo "ERROR: Missing or unreadable: $f"
        exit 1
    fi
done

mkdir -p "$OUTPUT_DIR"
cd "$VERL_ROOT"

uv run --frozen --all-packages --extra vllm --extra fsdp \
    python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=false \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$VAL_FILE" \
    data.image_key=images \
    data.train_batch_size=4 \
    data.max_prompt_length=8192 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=true \
    data.truncation=error \
    actor_rollout_ref.model.path="$BASE_MODEL" \
    actor_rollout_ref.model.lora_adapter_path="$SFT_ADAPTER" \
    actor_rollout_ref.model.lora_rank=8 \
    actor_rollout_ref.model.lora_alpha=16 \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.enable_chunked_prefill=true \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=false \
    reward.custom_reward_function.path="$PROJECT_ROOT/scripts/rl/reward.py" \
    reward.custom_reward_function.name=compute_score \
    trainer.project_name=chest_icpo \
    trainer.experiment_name=qwen35_9b_grpo_lora \
    trainer.logger='["console"]' \
    trainer.n_gpus_per_node="$NGPUS" \
    trainer.nnodes=1 \
    trainer.total_epochs=1 \
    trainer.default_local_dir="$OUTPUT_DIR" \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    "$@"
