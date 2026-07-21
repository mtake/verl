#!/usr/bin/env bash

#
# WIP with 8 GPUs
#

#Traceback (most recent call last):
#  File "/proj/dmfexp/granite_ja/mtake/w/verl-command/verl/verl/trainer/sft_trainer.py", line 461, in main
#    run_sft(config)
#  File "/proj/dmfexp/granite_ja/mtake/w/verl-command/verl/verl/trainer/sft_trainer.py", line 453, in run_sft
#    trainer.fit()
#  File "/proj/dmfexp/granite_ja/mtake/w/verl-command/verl/verl/trainer/sft_trainer.py", line 341, in fit
#    start_epoch = global_step // self.steps_per_epoch
#                  ~~~~~~~~~~~~^^~~~~~~~~~~~~~~~~~~~~~
#   ZeroDivisionError: integer division or modulo by zero
#
#   self.steps_per_epoch = len(self.train_dataloader)
#
#   self.train_dataloader = StatefulDataLoader(
#       dataset=self.train_dataset,
#       batch_size=self.train_batch_size_per_dp,
#       sampler=self.train_sampler,
#       collate_fn=self.collate_fn,
#       num_workers=self.config.data.num_workers,
#       pin_memory=False,
#       drop_last=True,
#       pin_memory_device=device_name,
#   )
#
#  According to the [documentation](https://meta-pytorch.org/data/beta/torchdata.stateful_dataloader.html),
#  len(self.train_dataloader) can be zero for small datasets. In this example, train dataset includes only 2 records.
#
#  len(dataloader) heuristic is based on the length of the sampler used. When dataset is an IterableDataset, it instead returns an estimate based on len(dataset) / batch_size, with proper rounding depending on drop_last, regardless of multi-process loading configurations. This represents the best guess PyTorch can make because PyTorch trusts user dataset code in correctly handling multi-process loading to avoid duplicate data.
#  However, if sharding results in multiple workers having incomplete last batches, this estimate can still be inaccurate, because (1) an otherwise complete batch can be broken into multiple ones and (2) more than one batch worth of samples can be dropped when drop_last is set. Unfortunately, PyTorch can not detect such cases in general.
#  See Dataset Types for more details on these two types of datasets and how IterableDataset interacts with Multi-process data loading.
#

# for macOS
if command -v gdate &> /dev/null
then
    DATE_CMD=gdate
else
    DATE_CMD=date
fi

START_TIME="$(${DATE_CMD} +%s)"
START_TIME_STR="$(${DATE_CMD} -d @${START_TIME} +%Y%m%d-%H%M%S)"
BASENAME="$(basename "${BASH_SOURCE}" .sh)"
HOSTNAME_S="$(hostname -s)"
LOGFILE="${BASENAME}-${START_TIME_STR}-${HOSTNAME_S}.log"
echo "XXX LOGFILE ${LOGFILE}" | tee -a ${LOGFILE}
echo "XXX DATETIME ${START_TIME_STR}" | tee -a ${LOGFILE}

# count gpus
if command -v nvidia-smi >/dev/null 2>&1; then
    NPROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)
else
    NPROC_PER_NODE=0
fi
echo "XXX NPROC_PER_NODE: ${NPROC_PER_NODE}" | tee -a ${LOGFILE}

# @@@ahoaho XXX
#if (( NPROC_PER_NODE == 0 )); then
#    echo "ERROR: A GPU is required to run this command. Exiting..." | tee -a ${LOGFILE}
#    exit 1
#fi
if (( NPROC_PER_NODE < 8 )); then
    echo "ERROR: 8 GPUs are required to run this command. Exiting..." | tee -a ${LOGFILE}
    exit 1
fi

#VENV=../../.venv
VENV=.venv
if [[ -d "${VENV}" ]]; then
    source "${VENV}/bin/activate"
fi

# @@@ahoaho XXX
### NOTE start Ray if not running.
##unset _RAY_STARTED
##if ! ray status > /dev/null 2>&1; then
##    echo "XXX Starting Ray..."
##    ray start --head
##    _RAY_STARTED=1
##else
##    echo "XXX Ray is already running."
##fi

ENV=""
ENV="TOKENIZERS_PARALLELISM=false ${ENV}"
# @@@ahoaho XXX
# AssertionError: Expandable segments are not compatible with memory pool.
#ENV="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ${ENV}"  # deprecated
#ENV="PYTORCH_ALLOC_CONF=expandable_segments:True ${ENV}"
ENV="NCCL_DEBUG=INFO ${ENV}"

if true; then
ENV="CUDA_LAUNCH_BLOCKING=1 ${ENV}"
ENV="TORCH_USE_CUDA_DSA=1 ${ENV}"
fi

if false; then
ENV="TORCH_CPP_LOG_LEVEL=INFO ${ENV}"
ENV="TORCH_DISTRIBUTED_DEBUG=DETAIL ${ENV}"

ENV="NCCL_DEBUG_SUBSYS=ALL ${ENV}"

ENV="NCCL_ASYNC_ERROR_HANDLING=1 ${ENV}"  # deprecated

ENV="TORCH_NCCL_ASYNC_ERROR_HANDLING=1 ${ENV}"

#ENV="NCCL_P2P_DISABLE=1 ${ENV}"
#ENV="NCCL_SHM_DISABLE=1 ${ENV}"
#ENV="NCCL_IB_DISABLE=1 ${ENV}"
fi

# ENV="DATA_ROOT=${HOME}/data-verl ${ENV}"
# ENV="MODEL_PATH=ibm-granite/granite-4.1-3b ${ENV}"  # default: Qwen/Qwen2.5-0.5B-Instruct
ENV="NPROC_PER_NODE=${NPROC_PER_NODE} ${ENV}"
# ENV="TOTAL_EPOCHS=${TOTAL_EPOCHS:-1} ${ENV}"  # default: 1
# ENV="LOGGER=mlflow ${ENV}"
# ENV="INFER_BACKEND=vllm ${ENV}"

# ENV="CKPTS_ROOT=${HOME}/ckpts-verl ${ENV}"

echo "================== ENVIRONMENT VARIABLES ===================" | tee -a ${LOGFILE}
env 2>&1 | tee -a ${LOGFILE}
echo "============================================================" | tee -a ${LOGFILE}

# @@@ahoaho XXX
#cmd="${ENV}bash examples/sft/multiturn/run_qwen2_5_0_5b_fsdp.sh"
cmd="${ENV}bash examples_mtake/sft/multiturn/run_qwen2_5_0_5b_fsdp_mtake.sh"
echo "$cmd" | tee -a ${LOGFILE}
eval "$cmd" 2>&1 | tee -a ${LOGFILE}

# @@@ahoaho XXX
# https://verl.readthedocs.io/en/latest/start/quickstart.html#step-3-perform-ppo-training-with-the-instruct-model
#
# See trainer.default_local_dir in verl/trainer/config/ppo_trainer.yaml
#
# The checkpoint is saved at the following dir by default: checkpoints/${trainer.project_name}/${trainer.experiment_name}. You can merge the saved checkpoints to huggingface model using verl.model_merger module, for example:
#
# python3 -m verl.model_merger merge \
#     --backend fsdp \
#     --local_dir checkpoints/${trainer.project_name}/${trainer.experiment_name}/global_step_1/actor \
#     --target_dir checkpoints/${trainer.project_name}/${trainer.experiment_name}/global_step_1/actor/huggingface
#
#PROJECT_NAME=multiturn-sft
#EXPERIMENT_NAME=multiturn-sft-qwen2_5_0_5b
#EXPERIMENT_DIR="${CKPTS_ROOT}/${PROJECT_NAME}/${EXPERIMENT_NAME}"
#LATEST_CHECKPOINTED_ITERATION="$(cat ${EXPERIMENT_DIR}/latest_checkpointed_iteration.txt)"
#LATEST_CHECKPOINT_DIR="${EXPERIMENT_DIR}/global_step_${LATEST_CHECKPOINTED_ITERATION}"
#cmd="${ENV}python -m verl.model_merger merge --backend fsdp --local_dir ${LATEST_CHECKPOINT_DIR} --target_dir ${LATEST_CHECKPOINT_DIR}/huggingface"
#echo "$cmd" | tee -a ${LOGFILE}
#eval "$cmd" 2>&1 | tee -a ${LOGFILE}

# @@@ahoaho XXX
##if [[ -n "${_RAY_STARTED}" ]]; then
##    echo "XXX Stopping Ray..."
##    ray stop
##fi

END_TIME="$(${DATE_CMD} +%s)"
END_TIME_STR="$(${DATE_CMD} -d @${END_TIME} +%Y%m%d-%H%M%S)"
echo "XXX DATETIME ${END_TIME_STR}" | tee -a ${LOGFILE}
echo "XXX ELAPSED_SECS $((END_TIME - START_TIME))" | tee -a ${LOGFILE}
