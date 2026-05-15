#!/usr/bin/env bash
set -euo pipefail

CWD=$(dirname "$0")

CONFIG=$1
GPUS=${GPUS:-4}
PORT=${PORT:-9909}
SEED=${SEED:-2025}

mkdir -p logs
# TODO: missing --use-amp
torchrun \
    --master-port "$PORT" \
    --nproc-per-node "$GPUS" \
    "$CWD/tools/base/train.py" \
    --config "$CONFIG" \
    --print-method rich \
    --seed "$SEED" \
    "${@:2}" 2>&1 | tee "logs/log_$(date +%Y%m%d%H%M%S).txt"
