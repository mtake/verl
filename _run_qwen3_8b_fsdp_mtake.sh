#!/usr/bin/env bash

#
# OK with 4 GPUs
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
    NDEVICES_PER_NODE=$(nvidia-smi --list-gpus | wc -l)
else
    NDEVICES_PER_NODE=0
fi
echo "XXX NDEVICES_PER_NODE: ${NDEVICES_PER_NODE}" | tee -a ${LOGFILE}

# @@@ahoaho XXX
#if (( NDEVICES_PER_NODE == 0 )); then
#    echo "ERROR: A GPU is required to run this command. Exiting..." | tee -a ${LOGFILE}
#    exit 1
#fi
if (( NDEVICES_PER_NODE < 8 )); then
    echo "ERROR: 8 GPUs are required to run this command. Exiting..." | tee -a ${LOGFILE}
    exit 1
fi

#VENV=../../.venv
VENV=.venv
if [[ -d "${VENV}" ]]; then
    source "${VENV}/bin/activate"
fi

# @@@ahoaho XXX
# NOTE start Ray if not running.
unset _RAY_STARTED
if ! ray status > /dev/null 2>&1; then
    echo "XXX Starting Ray..."
    ray start --head
    _RAY_STARTED=1
else
    echo "XXX Ray is already running."
fi

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

# ENV="DATA_DIR=${HOME}/data/gsm8k ${ENV}"
# @@@ahoaho XXX for functional test
#ENV="MODEL_PATH=ibm-granite/granite-4.1-3b ${ENV}"  # default: Qwen/Qwen3-8B
ENV="MODEL_PATH=Qwen/Qwen3-0.6B ${ENV}"  # default: Qwen/Qwen3-8B
ENV="NDEVICES_PER_NODE=${NDEVICES_PER_NODE} ${ENV}"
# @@@ahoaho XXX for functional test
ENV="TOTAL_EPOCHS=${TOTAL_EPOCHS:-1} ${ENV}"  # default: 15
# ENV="LOGGER=mlflow ${ENV}"
# ENV="INFER_BACKEND=vllm ${ENV}"

# ENV="CKPTS_ROOT=${HOME}/ckpts ${ENV}"

echo "================== ENVIRONMENT VARIABLES ===================" | tee -a ${LOGFILE}
env 2>&1 | tee -a ${LOGFILE}
echo "============================================================" | tee -a ${LOGFILE}

# @@@ahoaho XXX
#cmd="${ENV}bash examples/ppo_trainer/run_qwen3_8b_fsdp.sh"
cmd="${ENV}bash examples_mtake/ppo_trainer/run_qwen3_8b_fsdp_mtake.sh"
echo "$cmd" | tee -a ${LOGFILE}
eval "$cmd" 2>&1 | tee -a ${LOGFILE}

# @@@ahoaho XXX
if [[ -n "${_RAY_STARTED}" ]]; then
    echo "XXX Stopping Ray..."
    ray stop
fi

END_TIME="$(${DATE_CMD} +%s)"
END_TIME_STR="$(${DATE_CMD} -d @${END_TIME} +%Y%m%d-%H%M%S)"
echo "XXX DATETIME ${END_TIME_STR}" | tee -a ${LOGFILE}
echo "XXX ELAPSED_SECS $((END_TIME - START_TIME))" | tee -a ${LOGFILE}
