#!/usr/bin/env bash
set -eu

CWD=$(dirname "$0")

CONFIG=$1
CHECKPOINT=$2
GPUS=${GPUS:-4}
PORT=${PORT:-9909}
SEED=${SEED:-2025}

torchrun \
    --master_port="$PORT" \
    --nproc_per_node="$GPUS" \
    "$CWD/tools/train.py" \
    --config "$CONFIG" \
    --resume "$CHECKPOINT" \
    --print-method rich \
    --seed "$SEED" \
    --test-only \
    --preloads "robust_au_od.rtdetrv2" \
    "${@:3}"
