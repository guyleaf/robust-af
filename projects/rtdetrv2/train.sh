#!/usr/bin/env bash
set -eu

CWD=$(dirname "$0")

CONFIG=$1
GPUS=${GPUS:-4}
PORT=${PORT:-9909}
SEED=${SEED:-2025}

# TODO: missing --use-amp
torchrun \
    --master-port "$PORT" \
    --nproc-per-node "$GPUS" \
    "$CWD/tools/train.py" \
    --config "$CONFIG" \
    --print-method rich \
    --seed "$SEED" \
    --preloads "robust_u2u_od.rtdetrv2" \
    "${@:2}"
