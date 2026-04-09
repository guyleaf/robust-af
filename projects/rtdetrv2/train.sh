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
    "$CWD/tools/base/train.py" \
    --config "$CONFIG" \
    --print-method rich \
    --seed "$SEED" \
    "${@:2}" 2>&1 | tee "log_$(date +%Y%m%d%H%M%S).txt"
