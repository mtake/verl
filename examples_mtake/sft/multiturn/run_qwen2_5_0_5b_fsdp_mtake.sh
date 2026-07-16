#!/usr/bin/env bash
# SFT | multiturn | FSDP engine | NVIDIA GPUs
# Toggle Ulysses sequence parallel via SP_SIZE env var.
#
# Examples:
#   # plain SFT
#   bash run_qwen2_5_0_5b_fsdp.sh
#
#   # sequence-parallel (Ulysses) = 2
#   SP_SIZE=2 bash run_qwen2_5_0_5b_fsdp.sh

set -xeuo pipefail

DATA_ROOT=${DATA_ROOT:-$HOME/data-verl}  # Originally $HOME/data
LOGGER=${LOGGER:-console}  # Originally '["console","wandb"]'
CKPTS_ROOT=${CKPTS_ROOT:-$HOME/ckpts-verl}  # default: checkpoints

# ---- user-adjustable ----
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}
SP_SIZE=${SP_SIZE:-1}
MICRO_BATCH_SIZE_PER_GPU=${MICRO_BATCH_SIZE_PER_GPU:-4}
####MICRO_BATCH_SIZE_PER_GPU=${MICRO_BATCH_SIZE_PER_GPU:-1}  # ERR
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-1}
####TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}  # ERR
PROJECT_NAME=${PROJECT_NAME:-multiturn-sft}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-multiturn-sft-qwen2_5_0_5b}
# ---- end user-adjustable ----

nproc_per_node=${NPROC_PER_NODE:-8}
save_path=${CKPTS_ROOT}/${PROJECT_NAME}/${EXPERIMENT_NAME}

torchrun --nnodes=1 --nproc_per_node=${nproc_per_node} \
    -m verl.trainer.sft_trainer \
    data.train_files=$DATA_ROOT/multiturn/train.parquet \
    data.val_files=$DATA_ROOT/multiturn/test.parquet \
    data.messages_key=messages \
    data.micro_batch_size_per_gpu=${MICRO_BATCH_SIZE_PER_GPU} \
    model.path="${MODEL_PATH}" \
    model.use_remove_padding=true \
    engine=fsdp \
    engine.ulysses_sequence_parallel_size=${SP_SIZE} \
    trainer.default_local_dir="${save_path}" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.logger=${LOGGER} \
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS} "$@"
####    trainer.total_epochs=${TOTAL_EPOCHS} "$@"  # ERR
