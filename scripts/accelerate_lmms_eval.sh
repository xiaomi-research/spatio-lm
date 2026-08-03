#!/bin/bash

set -e

PROJ_ROOT=$(realpath $(dirname $0)/..)
pushd $PROJ_ROOT

OUTPUT_DIR=${1:-"work_dirs/$MLP_TASK_NAME/$MLP_TASK_ID"}

# For ddp
export NNODES=${MLP_WORKER_NUM:-1}
export NODE_RANK=$MLP_ROLE_INDEX
export MASTER_ADDR=$MLP_WORKER_0_HOST
export MASTER_PORT=$MLP_WORKER_0_PORT
export NPROC_PER_NODE=${MLP_WORKER_GPU:-8}
export NUM_PROCESS=$(($NNODES * $NPROC_PER_NODE))


ckpts=("$OUTPUT_DIR"/checkpoint-*)
for (( i=${#ckpts[@]}-1; i>=0; i-- )); do
    [ -d "${ckpts[i]}" ] || continue

    echo "Evaluating ${ckpts[i]} ..."
    accelerate launch \
    --multi_gpu \
    --num_processes=${NUM_PROCESS} \
    --machine_rank=${NODE_RANK} \
    --main_process_ip=${MASTER_ADDR} \
    --main_process_port=${MASTER_PORT} \
    --num_machines=${NNODES} \
    lmms-eval/eval_internvl.py \
    -m "${ckpts[i]}" \
    --save_file "${ckpts[i]}" \
    "${@:2}"
done

popd

set +e
