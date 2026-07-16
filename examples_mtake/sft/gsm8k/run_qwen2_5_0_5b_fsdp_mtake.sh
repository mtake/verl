#!/usr/bin/env bash
# SFT | GSM8K | FSDP engine | NVIDIA GPUs
# Covers plain SFT, Ulysses sequence parallel, Liger-kernel, and LoRA/PEFT via env vars.
#
# Examples:
#   # plain SFT
#   bash run_qwen2_5_0_5b_fsdp.sh
#
#   # sequence-parallel (Ulysses) = 2
#   SP_SIZE=2 bash run_qwen2_5_0_5b_fsdp.sh
#
#   # sequence-parallel + Liger kernel
#   SP_SIZE=2 USE_LIGER=1 bash run_qwen2_5_0_5b_fsdp.sh
#
#   # LoRA (PEFT)
#   USE_PEFT=1 bash run_qwen2_5_0_5b_fsdp.sh

set -xeuo pipefail

DATA_ROOT=${DATA_ROOT:-$HOME/data-verl}  # Originally $HOME/data
LOGGER=${LOGGER:-console}  # Originally '["console","wandb"]'
CKPTS_ROOT=${CKPTS_ROOT:-$HOME/ckpts-verl}  # default: checkpoints

# ---- user-adjustable ----
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}
SP_SIZE=${SP_SIZE:-1}
USE_LIGER=${USE_LIGER:-0}
USE_PEFT=${USE_PEFT:-0}
LORA_RANK=${LORA_RANK:-32}
LORA_ALPHA=${LORA_ALPHA:-16}
LORA_TARGETS=${LORA_TARGETS:-all-linear}
MICRO_BATCH_SIZE_PER_GPU=${MICRO_BATCH_SIZE_PER_GPU:-4}
LR=${LR:-1e-4}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
PROJECT_NAME=${PROJECT_NAME:-gsm8k-sft}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-gsm8k-sft-qwen2_5_0_5b}
# ---- end user-adjustable ----

# @@@ahoaho XXX
nproc_per_node=${NPROC_PER_NODE:-8}
save_path=${CKPTS_ROOT}/${PROJECT_NAME}/${EXPERIMENT_NAME}

extra_args=()
if [ "${USE_LIGER}" = "1" ]; then
    extra_args+=("model.use_liger=True")
fi
if [ "${USE_PEFT}" = "1" ]; then
    extra_args+=(
        "model.lora_rank=${LORA_RANK}"
        "model.lora_alpha=${LORA_ALPHA}"
        "model.target_modules=${LORA_TARGETS}"
    )
fi

torchrun --standalone --nnodes=1 --nproc_per_node=${nproc_per_node} \
    -m verl.trainer.sft_trainer \
    data.train_files=$DATA_ROOT/gsm8k_sft/train.parquet \
    data.val_files=$DATA_ROOT/gsm8k_sft/test.parquet \
    data.messages_key=messages \
    data.micro_batch_size_per_gpu=${MICRO_BATCH_SIZE_PER_GPU} \
    optim.lr=${LR} \
    engine=fsdp \
    engine.ulysses_sequence_parallel_size=${SP_SIZE} \
    model.path="${MODEL_PATH}" \
    model.use_remove_padding=true \
    trainer.default_local_dir="${save_path}" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.logger=${LOGGER} \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    "${extra_args[@]}" "$@"
